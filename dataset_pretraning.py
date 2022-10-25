import glob
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import linecache


class Dataset(data.Dataset):
    def __init__(self, mode):
        self.mode = mode

        self.distorted_list_bar = sorted(glob.glob('./dataset_bar/distorted/train/*.jpg'))
        self.distorted_list_pin = sorted(glob.glob('./dataset_pin/distorted/train/*.jpg'))
        self.distorted_list = self.distorted_list_bar + self.distorted_list_pin
        self.task_map = {"data": 0, "pincushion": 1}


    def __getitem__(self, index):
    
        im = np.array(Image.open(self.distorted_list[index]))[:, :, :3]
        im = im / 255.0
        
        cls = self.task_map[self.distorted_list[index].split('/')[-3]]
        
        if cls == 0:  # barrel
            para_line = linecache.getline('./dataset_bar/para_train_bar.txt', index + 1)
            para = torch.tensor(list(map(float, para_line[9:-2].split(', '))))
            para[0] = (para[0] + 1.5e-5) / (1.5e-5 - 1e-7)
            para[1] = (para[1] + 1.5e-10) / (1.5e-10 - 1e-12)
            para[2] = (para[2] + 1.5e-15) / (1.5e-15 - 1e-17)
            para[3] = (para[3] + 1.5e-20) / (1.5e-20 - 1e-22)
            
        else:
            para_line = linecache.getline('./dataset_pin/para_train_pin.txt', index - 49999)
            para = torch.tensor(list(map(float, para_line[9:-2].split(', '))))
            para[0] = (para[0] - 1e-7) / (1.5e-5 - 1e-7)
            para[1] = (para[1] - 1e-12) / (1.5e-10 - 1e-12)
            para[2] = (para[2] - 1e-17) / (1.5e-15 - 1e-17)
            para[3] = (para[3] - 1e-22) / (1.5e-20 - 1e-22)

        im = torch.from_numpy(im).permute(2, 0, 1).float()

        return im, cls, para

    def __len__(self):
        return len(self.distorted_list)
        
        
        
        
class Dataset2(data.Dataset):
    def __init__(self, mode):
        self.mode = mode

        self.distorted_list_bar = sorted(glob.glob('./dataset_bar/distorted/test/*.jpg'))
        self.distorted_list_pin = sorted(glob.glob('./dataset_pin/distorted/test/*.jpg'))
        self.distorted_list = self.distorted_list_bar + self.distorted_list_pin
        self.task_map = {"data": 0, "pincushion": 1}


    def __getitem__(self, index):
    
        im = np.array(Image.open(self.distorted_list[index]))[:, :, :3]
        im = im / 255.0
        
        cls = self.task_map[self.distorted_list[index].split('/')[-3]]
        
        if cls == 0:  # barrel
            para_line = linecache.getline('./dataset_bar/para_test_bar.txt', index + 1)
            para = torch.tensor(list(map(float, para_line[9:-2].split(', '))))
            para[0] = (para[0] + 1.5e-5) / (1.5e-5 - 1e-7)
            para[1] = (para[1] + 1.5e-10) / (1.5e-10 - 1e-12)
            para[2] = (para[2] + 1.5e-15) / (1.5e-15 - 1e-17)
            para[3] = (para[3] + 1.5e-20) / (1.5e-20 - 1e-22)
            
        else:
            para_line = linecache.getline('./dataset_bar/para_test_bar.txt', index - 4999)
            para = torch.tensor(list(map(float, para_line[9:-2].split(', '))))
            para[0] = (para[0] - 1e-7) / (1.5e-5 - 1e-7)
            para[1] = (para[1] - 1e-12) / (1.5e-10 - 1e-12)
            para[2] = (para[2] - 1e-17) / (1.5e-15 - 1e-17)
            para[3] = (para[3] - 1e-22) / (1.5e-20 - 1e-22)
        
        im = torch.from_numpy(im).permute(2, 0, 1).float()

        return im, cls, para

    def __len__(self):
        return len(self.distorted_list)
        