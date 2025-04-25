import torch
import torch.nn as nn
import torch.nn.functional as F
from .pvtv2_b2 import pvt_v2_b2, pvt_v2_b0
from lib.visualizer import get_local
from lib.CrossAttention import MixtureCrossattention
get_local.activate()

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)

        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)
        return x


# Group-Reversal Attention (GRA) Block
class GRA(nn.Module):
    def __init__(self, channel, subchannel):
        super(GRA, self).__init__()
        self.group = channel//subchannel
        self.conv = nn.Sequential(
            nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x, y):
        if self.group == 1:
            x_cat = torch.cat((x, y), 1)
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y), 1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y), 1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y), 1)
        elif self.group == 16:
            xs = torch.chunk(x, 16, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y), 1)
        elif self.group == 32:
            xs = torch.chunk(x, 32, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
            xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
            xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y), 1)
        else:
            raise Exception("Invalid Channel")

        x = x + self.conv(x_cat)
        y = y + self.score(x)

        return x, y


class ReverseStage(nn.Module):
    def __init__(self, channel):
        super(ReverseStage, self).__init__()
        self.weak_gra = GRA(channel, channel)
        self.medium_gra = GRA(channel, 8)
        self.strong_gra = GRA(channel, 1)

    def forward(self, x, y):
        # reverse guided block
        y = -1 * (torch.sigmoid(y)) + 1

        # three group-reversal attention blocks
        x, y = self.weak_gra(x, y)
        x, y = self.medium_gra(x, y)
        _, y = self.strong_gra(x, y)

        return y


class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, imagenet_pretrained=True, all_channel=32, all_dim=11*11):
        super(Network, self).__init__()
        # ---- pvt Backbone ----
        self.encoder = pvt_v2_b2(pretrained=imagenet_pretrained)
        self.rfb2_1 = RFB_modified(128, channel)
        self.rfb3_1 = RFB_modified(320, channel)
        self.rfb4_1 = RFB_modified(512, channel)

        # ---- Partial Decoder ----
        self.NCD = NeighborConnectionDecoder(channel)
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.dim = all_dim
        # # ---- reverse stage ----
        self.RS5 = ReverseStage(channel)
        self.RS4 = ReverseStage(channel)
        self.RS3 = ReverseStage(channel)

        self.MixtureCrossattention = MixtureCrossattention()

        self.upsample1 = nn.Upsample(size=12, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(size=11, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)


    def forward(self, input_1, input_2):
        # Feature Extraction
        origin_shape = input_1.shape
        #
        if len(origin_shape) == 5:
            x = input_1.view(-1, *origin_shape[2:])
            y = input_2.view(-1, *origin_shape[2:])
        else:
            x = input_1.view(-1, 3, 352, 352)
            y = input_2.view(-1, 3, 352, 352)

        #pvt-based
        x1, x2, x3, x4 = self.encoder(x)
        y1, y2, y3, y4 = self.encoder(y)
        
        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)        # channel -> 32
        #
        y2_rfb = self.rfb2_1(y2)  # channel -> 32
        y3_rfb = self.rfb3_1(y3)  # channel -> 32
        y4_rfb = self.rfb4_1(y4)  # channel -> 32

        x5_rfb = self.upsample1(x4_rfb)
        y5_rfb = self.upsample1(y4_rfb)
        x5_rfb, y5_rfb = self.MixtureCrossattention(x5_rfb, y5_rfb)
        x5_rfb = self.upsample2(x5_rfb)
        y5_rfb = self.upsample2(y5_rfb)

        # Neighbourhood Connected Decoder
        S_g = self.NCD(x5_rfb, self.upsample3(x4_rfb), self.upsample3(x3_rfb))
        S_g_pred = F.interpolate(S_g, scale_factor=8, mode='bilinear')  # Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- reverse stage 5 ----
        guidance_g = F.interpolate(S_g, scale_factor=0.25, mode='bilinear')
        ra3_feat = self.RS5(x4_rfb, guidance_g)
        S_5 = ra3_feat + guidance_g
        S_5_pred = F.interpolate(S_5, scale_factor=32, mode='bilinear')  # Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse stage 4 ----
        guidance_5 = F.interpolate(S_5, scale_factor=2, mode='bilinear')
        ra2_feat = self.RS4(x3_rfb, guidance_5)
        S_4 = ra2_feat + guidance_5
        S_4_pred = F.interpolate(S_4, scale_factor=16, mode='bilinear')  # Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse stage 3 ----
        guidance_4 = F.interpolate(S_4, scale_factor=2, mode='bilinear')
        ra1_feat = self.RS3(x2_rfb, guidance_4)
        S_3 = ra1_feat + guidance_4
        S_3_pred = F.interpolate(S_3, scale_factor=8, mode='bilinear')  # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        Y_g = self.NCD(y5_rfb, self.upsample3(y4_rfb), self.upsample3(y3_rfb))
        Y_g_pred = F.interpolate(Y_g, scale_factor=8, mode='bilinear')  # Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        guidance_y = F.interpolate(Y_g, scale_factor=0.25, mode='bilinear')
        rb3_feat = self.RS5(y4_rfb, guidance_y)
        Y_5 = rb3_feat + guidance_y
        Y_5_pred = F.interpolate(Y_5, scale_factor=32, mode='bilinear')  # Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        guidance_5 = F.interpolate(Y_5, scale_factor=2, mode='bilinear')
        rb2_feat = self.RS4(y3_rfb, guidance_5)
        Y_4 = rb2_feat + guidance_5
        Y_4_pred = F.interpolate(Y_4, scale_factor=16, mode='bilinear')  # Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        guidance_4 = F.interpolate(Y_4, scale_factor=2, mode='bilinear')
        rb1_feat = self.RS3(y2_rfb, guidance_4)
        Y_3 = rb1_feat + guidance_4
        Y_3_pred = F.interpolate(Y_3, scale_factor=8, mode='bilinear')  # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return [S_g_pred, S_5_pred, S_4_pred, S_3_pred], [Y_g_pred, Y_5_pred, Y_4_pred, Y_3_pred]


if __name__ == '__main__':
    import numpy as np
    from time import time
    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 352, 352)
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)