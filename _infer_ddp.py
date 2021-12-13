import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import torch.nn as nn
from models import DPED_DDP

inp_path = 'data/val/inp/1.jpg'
model_path = 'saved/model.pth'

model = DPED_DDP()

checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

model.load_state_dict(checkpoint['model_state_dict'], strict=False)

def read_img(path):
    img = cv2.imread(path)
    name = path.split('/')[-1]
    img = cv2.resize(img, (128,128))
    img = img.transpose((2,0,1))
    img = img/127.5-1
    return torch.tensor([img], dtype=torch.float)

inp = cv2.imread(inp_path)
inp = cv2.resize(inp, (128,128))
plt.imshow(inp[:,:,[2,1,0]])
plt.show()

inp = read_img(inp_path)
with torch.no_grad(): pred = model(inp)

img = pred[0].detach().numpy().transpose((1,2,0))
img = 127.5*(img+1)
img = img.astype(np.uint8)
plt.imshow(img[:,:,[2,1,0]])
plt.show()
