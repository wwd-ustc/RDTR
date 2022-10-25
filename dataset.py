import glob
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self):

        self.distorted_list = sorted(glob.glob('./dataset_bar/distorted/train/*.jpg'))
        self.bflow_list = sorted(glob.glob('./dataset_bar/bflow/train/*.npy'))
        
        assert len(self.distorted_list) == len(self.bflow_list)


    def __getitem__(self, index):
    
        im = np.array(Image.open(self.distorted_list[index]))[:, :, :3] / 255.0
        im = torch.from_numpy(im).permute(2, 0, 1).float()
        
        bflow = np.load(self.bflow_list[index])
        bflow = torch.from_numpy(bflow).permute(2, 1, 0).float()
        
        return im, bflow

    def __len__(self):
        return len(self.distorted_list)


class Dataset2(data.Dataset):
    def __init__(self):

        self.distorted_list = sorted(glob.glob('./dataset_bar/distorted/test/*.jpg'))
        self.bflow_list = sorted(glob.glob('./dataset_bar/bflow/test/*.npy'))
        
        assert len(self.distorted_list) == len(self.bflow_list)


    def __getitem__(self, index):
    
        im = np.array(Image.open(self.distorted_list[index]))[:, :, :3] / 255.0
        im = torch.from_numpy(im).permute(2, 0, 1).float()
        
        bflow = np.load(self.bflow_list[index])
        bflow = torch.from_numpy(bflow).permute(2, 1, 0).float()

        return im, bflow

    def __len__(self):
        return len(self.distorted_list)
        