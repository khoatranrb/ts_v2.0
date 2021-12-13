import torch
import numpy as np
import torch.nn.functional as F
from ddp import Step
import torch.nn as nn

class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(288, 128)
        self.fc4 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(-1)
    def backbone(self, inp):
        out = self.tanh(self.conv1(inp))
        out = self.tanh(self.conv2(self.maxpool(out)))
        out = self.maxpool(out)
        out = self.tanh(self.conv3(out))
        out = self.maxpool(out)
        return torch.flatten(out, start_dim=1)
    def forward(self, inp):
        out = self.backbone(inp)
        out = self.fc4(self.tanh(self.fc1(out)))
        return out
     
class MNIST_DDP(nn.Module):
    def __init__(self):
        super(MNIST_DDP, self).__init__()
        self.conv1 = Step(nn.Conv2d(1, 32, 3, 1, 1))
        self.conv2 = Step(nn.Conv2d(32, 32, 3, 1, 1))
        self.conv3 = Step(nn.Conv2d(32, 32, 3, 1, 1))
        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.Tanh()
        self.fc1 = Step(nn.Linear(288, 128))
        self.fc4 = Step(nn.Linear(128, 10))
        self.softmax = nn.Softmax(-1)
    def backbone(self, inp, update=False, opt=None):
        out = self.relu(self.conv1(inp, update, opt))
        out = self.relu(self.conv2(self.maxpool(out), update, opt))
        out = self.maxpool(out)
        out = self.relu(self.conv3(out, update, opt))
        out = self.maxpool(out)
        return torch.flatten(out, start_dim=1)
    def forward(self, inp=None, update=False, opt=None):
        out = self.backbone(inp, update, opt)
        out = self.relu(self.fc1(out, update, opt))
        out = self.fc4(out, update, opt)
        return out
      
class DPED(nn.Module):
    def __init__(self, out_channels=64):
        super(DPED, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channels, 9, padding=4)
        
        self.block1 = ConvBlock(64, 64, 3)
        self.block2 = ConvBlock(64, 64, 3)
        self.block3 = ConvBlock(64, 64, 3)
        self.block4 = ConvBlock(64, 64, 3)
        
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, 9, padding=4)
        self.activation = nn.Tanh()
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.relu3(out)

        out = self.conv4(out)
        out = self.activation(out)
        
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_size):
        super(ConvBlock, self).__init__()
        self.conv_size = conv_size
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, conv_size, 1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, conv_size, 1, padding=1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        
        out = out + x
        return out
  
class DPED_DDP(nn.Module):
    def __init__(self, out_channels=64):
        super(DPED_DDP, self).__init__()
        self.conv1 = Step(nn.Conv2d(3, out_channels, 9, padding=4))
        
        self.block1 = ConvBlock_DDP(64, 64, 3)
        self.block2 = ConvBlock_DDP(64, 64, 3)
        self.block3 = ConvBlock_DDP(64, 64, 3)
        self.block4 = ConvBlock_DDP(64, 64, 3)
        
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

class ConvBlock_DDP(nn.Module):
    def __init__(self, in_channels, out_channels, conv_size):
        super(ConvBlock_DDP, self).__init__()
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
