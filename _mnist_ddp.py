import torch.nn as nn

import torchvision

image_data = torchvision.datasets.MNIST('/content',download=True)

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

class Data(Dataset):
    def __init__(self, train=None, data=None):
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


import torch
import numpy as np
import torch.nn.functional as F
from ddp import Step
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
net = Net().cuda()

import numpy as np

from ddp import DDPNOPT
opt = DDPNOPT(net, lr=1e-4, lrddp=1e-4)
ce = nn.CrossEntropyLoss()

from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

device = 'cuda'
datatrain = Data(True, image_data)
train_loader = DataLoader(datatrain, batch_size=32, shuffle=True)
dataval = Data(False, image_data)
val_loader = DataLoader(dataval, batch_size=32)

for epoch in range(100):
    print('Epoch',epoch+1)
    torch.cuda.empty_cache()
    train_iter = iter(train_loader)
    net.train()
    for i in tqdm(range(len(train_loader))):
        opt.zero_grad()
        x, y = next(train_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logit = net(x)
        loss = ce(logit, y)
        loss.backward(retain_graph=True)
        opt.step()
        break

    net.eval()
    gts, preds = [], []
    val_iter = iter(val_loader)
    for i in range(len(val_loader)):
        x, y = next(val_iter)
        x = x.to(device, non_blocking=True)
        with torch.no_grad(): logit = net(x)
        pred = logit.argmax(axis=-1)
        y = list(y.numpy())
        pred = list(pred.cpu().detach().numpy())
        gts+=y
        preds+=pred
    print(f1_score(gts,preds,average='macro'))
