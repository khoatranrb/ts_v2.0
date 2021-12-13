import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import torch
import numpy as np
import torch.nn.functional as F
from models import MNIST

net = MNIST().cuda()

from torch.optim import Adam, RMSprop, SGD

opt = Adam(net.parameters(), lr=1e-4)

ce = nn.CrossEntropyLoss()

from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from dataloaders import MNIST_Data

device = 'cuda'
datatrain = MNIST_Data(True)
train_loader = DataLoader(datatrain, batch_size=32, shuffle=True)
dataval = MNIST_Data(False)
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
