import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

def generate_sent_masks(length):
    """ Generate sentence masks for encoder hidden states.
        returns enc_masks (Tensor): Tensor of sentence masks of shape (b, max_seq_length),where max_seq_length = max source length """
    enc_masks = torch.ones([length, length], dtype=torch.bool)
    for id, i in enumerate(enc_masks):
        i[12:] = 0 if id > 11 else 1
    return enc_masks


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class CrossAttention(nn.Module):
    def __init__(self, ):
        super(CrossAttention, self).__init__()

    def forward(self, x, matrix):
        x = torch.bmm(x, matrix).contiguous()
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1)) 
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, original):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1).contiguous()
        # input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        # input: B, C, H, W -> v: B, C, H * W
        v = self.value(original).view(batch_size, -1, height * width)
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k) 
        attn_matrix = self.softmax(attn_matrix) 
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1)) 
        out = out.view(*input.shape)
        return self.gamma * out + input


class MixtureCrossattention(nn.Module):
    def __init__(self, channel=32, lamda_1=0.7):
        super(MixtureCrossattention, self).__init__()
        self.CrossAttention = CrossAttention()
        self.SelfAttention = SelfAttention(in_channels=32)
        self.lamda_1 = lamda_1
        self.lamda_2 = 1 - lamda_1
        self.x_patch_embbeding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=3, p2=3),
        )

        self.complement_patch_embbeding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=3, p2=3),
        )

    def forward(self, x1, x2):
        x1 = self.x_patch_embbeding(x1)
        x2 = self.complement_patch_embbeding(x2) 

        A_B = torch.cat((x1, x2), 2) 
        B_A = torch.cat((x2, x1), 2) 

        B_A = torch.transpose(B_A, 1, 2)
        A = torch.bmm(B_A, A_B)  #[576,576]
        B = torch.transpose(A.clone(), 1, 2) 

        A = F.softmax(A, dim=1) 
        B[0,288:,:288]=B[0,288:,:288].transpose(0, 1)
        B = F.softmax(B, dim=1)

        from einops import rearrange
        x3 = self.CrossAttention(torch.cat((self.lamda_1 * x1, self.lamda_2 * x2), 2), B)[:, :, :288]
        x4 = self.CrossAttention(torch.cat((self.lamda_1 * x2, self.lamda_2 * x1), 2), A)[:, :, :288]
        x3 = rearrange(x3, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', c=32, p1=3, p2=3, h=4, w=4)
        x4 = rearrange(x4, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', c=32, p1=3, p2=3, h=4, w=4)
        

        return x3, x4
