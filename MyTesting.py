import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from lib.MAST_Network import Network
# from lib.MAST_Network_resnet import Network
from utils.data_val import test_dataset
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--data_path', type=str, default='./data/MAST/SUN-SEG/')
parser.add_argument('--gpu_id', type=str, default="1", help='gpu id')
parser.add_argument('--pth_path', type=str, default='./weight/Net_epoch_best.pth')
parser.add_argument('--save_path', type=str, default='./data/')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU {}'.format(opt.gpu_id))

for _data_name in ['TestEasyDataset/Unseen', 'TestHardDataset/Unseen']:
    pred_data_path = os.path.join(opt.data_path,_data_name)
    save_path = os.path.join(opt.save_path,_data_name)
    
    model = Network(imagenet_pretrained=True)

    model.load_state_dict(torch.load(opt.pth_path,weights_only=True))
    print("Load Parameter Successfully!")
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Frame/'.format(pred_data_path)
    gt_root='{}/GT/'.format(pred_data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        image = image.cuda()
        res1, res2 = model(image, image)
        res = res1
        res = F.upsample(res[3], size=(352,352), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name[-1]))
        if not os.path.exists(os.path.join(save_path,name[0])):
            os.makedirs(os.path.join(save_path,name[0]))

        cv2.imwrite(os.path.join(save_path,'/'.join(name)), res*255)
