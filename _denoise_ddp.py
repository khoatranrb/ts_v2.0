import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import cv2
import os

class Data(Dataset):
    def __init__(self, path, train=True):
        self.train = train
        self.path = os.path.join(path, 'val')
        if train:
            self.path = os.path.join(path, 'train')

    def __len__(self):
        return 80
        if self.train:
            return len(os.listdir(self.path))
        return len(os.listdir(os.path.join(self.path, 'inp')))
    
    def __getitem__(self, idx):
        idx -= 1
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

bs = 8

trainset = Data('data', train= True)
trainloader = DataLoader(trainset, shuffle=True, batch_size=bs)

valset = Data('data', train=not True)
valloader = DataLoader(valset, batch_size=bs)

import torch.nn as nn
from ddp import Step

class DPED(nn.Module):
    def __init__(self, out_channels=64):
        super(DPED, self).__init__()
        self.conv1 = Step(nn.Conv2d(3, out_channels, 9, padding=4))
        
        self.block1 = ConvBlock(64, 64, 3)
        self.block2 = ConvBlock(64, 64, 3)
        self.block3 = ConvBlock(64, 64, 3)
        self.block4 = ConvBlock(64, 64, 3)
        
        self.conv2 = Step(nn.Conv2d(64, 64, 3, padding=1))
        self.conv3 = Step(nn.Conv2d(64, 64, 3, padding=1))
        self.conv4 = Step(nn.Conv2d(64, 3, 9, padding=4))
        # self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        # self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        # self.conv4 = nn.Conv2d(64, 3, 9, padding=4)
        self.activation = nn.Tanh()
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        
    def forward(self, x=None, update=False, opt=None):
        out = self.conv1(x, update, opt)
        out = self.relu1(out)
        
        out = self.block1(out, update, opt)
        out = self.block2(out, update, opt)
        out = self.block3(out, update, opt)
        out = self.block4(out, update, opt)
        
        out = self.conv2(out, update, opt)
        out = self.relu2(out)
        
        out = self.conv3(out, update, opt)
        out = self.relu3(out)

        out = self.conv4(out, update, opt)
        out = self.activation(out)
        
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_size):
        super(ConvBlock, self).__init__()
        self.conv_size = conv_size
        
        self.conv1 = Step(nn.Conv2d(in_channels, out_channels, conv_size, 1, padding=1))
        self.conv2 = Step(nn.Conv2d(in_channels, out_channels, conv_size, 1, padding=1))
        
        self.relu = nn.ReLU()
        
    def forward(self, x=None, update=False, opt=None):
        out = self.conv1(x, update, opt)
        out = self.relu(out)
        
        out = self.conv2(out, update, opt)
        out = self.relu(out)
        
        out = out + x
        return out

model = DPED().cuda()

from ddp import DDPNOPT

opt = DDPNOPT(model, lr=1e-4, lrddp=1e-4)

loss_func = nn.MSELoss(reduction='sum')

from tqdm.auto import tqdm
epochs = 50

for epoch in range(0, epochs):
    torch.cuda.empty_cache()
    train_iter = iter(trainloader)
    model.train()
    total_loss = 0
    count = 0
    print('Epoch', epoch)
    for _ in tqdm(range(len(trainloader))):
        opt.zero_grad()
        try:
            inp, out = next(train_iter)
        except Exception as e:
            # print(e)
            continue
        inp = inp.cuda()
        out = out.cuda()

        pred = model(inp)
        loss = loss_func(pred, out)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        count += 1
    print(total_loss/(count*bs))
    # torch.save(model, 'saved/ddpnopt/%s.pth'%(epoch))
    # torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': opt.state_dict(),
    #         }, 'saved/ddpnopt/%s.pth'%(epoch))
    model.eval()
    val_iter = iter(valloader)
    total_loss = 0
    count = 0
    for _ in tqdm(range(len(valloader))):
        try:
            inp, out = next(val_iter)
        except:
            continue
        inp = inp.cuda()
        out = out.cuda()

        with torch.no_grad(): pred = model(inp)
        loss = loss_func(pred, out)
        total_loss += loss.item()
        count += 1
    print(total_loss/(count*bs))
