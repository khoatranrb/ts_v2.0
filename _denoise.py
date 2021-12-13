import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import cv2
import os
from dataloaders import Denoise_Data
import torch.nn as nn
from models import DPED

bs = 8

trainset = Denoise_Data('data', train= True)
trainloader = DataLoader(trainset, shuffle=True, batch_size=bs)

valset = Denoise_Data('data', train=not True)
valloader = DataLoader(valset, batch_size=bs)

model = DPED().cuda()

from torch.optim import SGD, Adam, RMSprop

opt = RMSprop(model.parameters(), lr=1e-4)

loss_func = nn.MSELoss(reduction='sum')

from tqdm.auto import tqdm
epochs = 50

for epoch in range(epochs):
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
#     torch.save(model, 'saved/RMSprop/%s.pth'%(epoch))
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
