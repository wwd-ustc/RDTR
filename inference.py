from model import RDTR

import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io as io
import numpy as np
import cv2
import os
from PIL import Image
import argparse
import warnings

warnings.filterwarnings('ignore')


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.RDTR = RDTR()  # 矫正

    def forward(self, x):
        bm = self.RDTR(x)
        bm = bm / 127.5 - 1

        return bm


def reload_rec_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


def rec(rec_model_path, distorrted_path, save_path, opt):
    print(torch.__version__)

    # distorted images list
    img_list = os.listdir(distorrted_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    net = Net(opt).cuda()

    # reload rec model
    reload_rec_model(net.RDTR, rec_model_path)

    net.eval()

    for img_path in img_list:
        name = img_path.split('.')[-2]  # image name
        img_path = distorrted_path + img_path  # image path
        im_ori = np.array(Image.open(img_path))[:, :, :3] / 255.  # read image 0-255 to 0-1
        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (256, 256))
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float().unsqueeze(0)

        with torch.no_grad():
            bm_pred = net(im.cuda())
            bm_pred = bm_pred.cpu()

            bm0 = cv2.resize(bm_pred[0, 0].numpy(), (w, h))  # x flow
            bm1 = cv2.resize(bm_pred[0, 1].numpy(), (w, h))  # y flow
            bm0 = cv2.blur(bm0, (3, 3))
            bm1 = cv2.blur(bm1, (3, 3))
            lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)

            # save rectified image
            out = F.grid_sample(torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float(), lbl, align_corners=True)
            io.imsave(save_path + name + '_rec' + '.png', ((out[0] * 255).permute(1, 2, 0).numpy()).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rec_model_path', default='./pretrained_net/')
    parser.add_argument('--distorrted_path', default='./distorted/')
    parser.add_argument('--index', type=int, default=0)
    opt = parser.parse_args()
    
    pth_list = sorted(os.listdir(opt.rec_model_path))
    print(pth_list)

    for i in pth_list:
        rec(rec_model_path=opt.rec_model_path + i,
            distorrted_path=opt.distorrted_path,
            save_path='./result/',
            opt=opt)


if __name__ == "__main__":
    main()
