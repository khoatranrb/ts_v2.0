import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import torch
import cv2
import os

class MNIST_Data(Dataset):
    def __init__(self, train=None, data=None):
        data = torchvision.datasets.MNIST('/content',download=True)
        self.train = train
        if train:
            self.X = data.data[:10000,...]
            self.y = data.targets[:10000]
        else:
            self.X = data.data[50000:,...]
            self.y = data.targets[50000:]
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        img = self.X[idx][np.newaxis,...]
        img = img/127.5-1
        return img, np.array(self.y[idx], dtype=np.int64)
      
class Denoise_Data(Dataset):
    def __init__(self, path, train=True):
        self.train = train
        self.path = os.path.join(path, 'val')
        if train:
            self.path = os.path.join(path, 'train')

    def __len__(self):
        if self.train:
            return len(os.listdir(self.path))
        return len(os.listdir(os.path.join(self.path, 'inp')))
    
    def __getitem__(self, idx):
        idx += 1
        if self.train:
            path = os.path.join(self.path, '%s.jpg'%(idx))
            out = cv2.imread(path)
            out = cv2.resize(out, (128,128))
            noise = np.random.normal(0,40,out.shape)
            inp = out.astype(float) + noise
            inp = np.clip(inp,0,255)
        else:
            inppath = os.path.join(self.path, 'inp', '%s.jpg'%(idx))
            outpath = os.path.join(self.path, 'out', '%s.jpg'%(idx))
            out = cv2.imread(outpath)
            inp = cv2.imread(inppath)
            out = cv2.resize(out, (128,128))
            inp = cv2.resize(inp, (128,128))
            

        inp = inp.astype(np.uint8).transpose((2,0,1))
        out = out.transpose((2,0,1))

        inp = inp/127.5-1
        out = out/127.5-1

        return inp.astype(np.float32), out.astype(np.float32)
