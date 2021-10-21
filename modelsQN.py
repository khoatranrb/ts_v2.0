import torch.nn as nn

class Net(nn.Module):
    def __init__(self, seq):
        super(Net, self).__init__()
        self.seq = nn.Sequential(*seq)
    def forward(self, x):
        return self.seq(x)
    def step(self, x):
        for i, module in enumerate(self.seq.children()):
            x = module.update(x)

from keras.datasets import mnist
from modulesQN import Step

(train_X, train_y), (test_X, test_y) = mnist.load_data()

# seq = [Step(nn.Linear(784, 32), nn.Sigmoid()),
#        Step(nn.Linear(32, 32), nn.Sigmoid()),
#        Step(nn.Linear(32, 32), nn.Sigmoid()),
#        Step(nn.Linear(32, 32), nn.Sigmoid()),
#        Step(nn.Linear(32, 32), nn.Sigmoid()),
#        Step(nn.Linear(32, 32), nn.Sigmoid()),
#        Step(nn.Linear(32, 32), nn.Sigmoid()),
#        Step(nn.Linear(32, 32), nn.Sigmoid()),
#        Step(nn.Linear(32, 32), nn.Sigmoid()),
#        Step(nn.Linear(32, 32), nn.Sigmoid()),
#        Step(nn.Linear(32, 32), nn.Sigmoid()),
#        Step(nn.Linear(32,10), nn.Softmax(-1))]

seq = [Step(nn.Conv2d(1, 64, 3, 1, 1), nn.Sigmoid()),
       Step(nn.Conv2d(64, 32, 3, 1, 1), nn.Sigmoid()),
       Step(nn.Conv2d(32, 32, 3, 1, 1), nn.Sigmoid()),
       Step(nn.Conv2d(32, 32, 3, 1, 1), nn.Sigmoid()),
       Step(nn.Conv2d(32, 32, 3, 1, 1), nn.Sigmoid()),
       Step(nn.Conv2d(32, 32, 3, 1, 1), nn.Sigmoid()),
       Step(nn.Conv2d(32, 32, 3, 1, 1), nn.Sigmoid()),
       Step(nn.Conv2d(32, 32, 3, 1, 1), nn.Sigmoid()),
       Step(nn.Conv2d(32, 32, 3, 1, 1), nn.Sigmoid()),
       Step(nn.Conv2d(32, 32, 3, 1, 1), nn.Sigmoid()),
       Step(nn.Conv2d(32, 32, 3, 1, 1), nn.Sigmoid()),
       Step(nn.Conv2d(32, 3, 3, 1, 1), nn.Sigmoid())]

net = Net(seq)

from hack_grads import *
import numpy as np

add_hooks(net)

from torch.optim import Adam

adam = Adam(net.parameters(), lr=1e-3)

# x = train_X[0:2].reshape(2,784)/255.0
#
# x = torch.tensor(x.astype(np.float32))
# y = torch.tensor(np.array([[0,0,1,0,0,0,0,0,0,0],
#                 [0,0,0,0,1,0,0,0,0,0]], dtype=np.float32))

x = train_X[0:2]/255.0
x = x[:,np.newaxis,...]
y = torch.tensor(x.astype(np.float32)).repeat((1,3,1,1))
x = torch.tensor(x.astype(np.float32))

mse = nn.MSELoss()

for _ in range(20):
    adam.zero_grad()
    pred = net(x)
    loss = mse(pred,y)
    loss.backward(retain_graph=True)
    # adam.step()
    compute_grad1(net)
    net.step(x)
    clear_backprops(net)
    print(loss)
disable_hooks()
