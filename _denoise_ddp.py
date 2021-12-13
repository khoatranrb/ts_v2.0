import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import cv2
import os
from dataloaders import Denoise_Data
import torch.nn as nn
from models import DPED_DDP

bs = 8

trainset = Denoise_Data('data', train= True)
trainloader = DataLoader(trainset, shuffle=True, batch_size=bs)

valset = Denoise_Data('data', train=not True)
valloader = DataLoader(valset, batch_size=bs)

model = DPED_DDP().cuda()

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
