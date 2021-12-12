import numpy as np
import torch.nn as nn
import torch

def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__

class Step(nn.Module):
    def __init__(self, module):
        super(Step, self).__init__()
        self.x = None
        self.x_old = None
        self.mod = module
        self.type = _layer_type(module)
    def __str__(self):
        return 'DDP'
    def forward(self, inp=None, update=False, opt=None):
        if not update:
            self.x_old = inp
            self.x = nn.Parameter(inp)
            return (self.mod(self.x)+self.mod(inp))/2
        else:
            try: return self.update(inp, opt)
            except: return self.update(self.x, opt)

    def forward_wo_train(self, inp):
        return self.mod(inp)

    @torch.no_grad()
    def update(self, inp, opt):
        Q_u = self.mod.weight.grad
        Q_uu = opt.state[self.mod.weight]['square_avg']
        Q_uu = Q_uu.sqrt().add_(opt.eps)
        Q_x = self.x.grad.mean(dim=0)
        if self.type=='Linear':
            Q_ux = calc_q_ux_fc(Q_u, Q_x.unsqueeze(-1))
            big_k = calc_big_k_fc(Q_uu, Q_ux)
            term2 = torch.einsum('xy,zt->xt', big_k, (inp - self.x_old).mean(dim=0).unsqueeze(-1)).squeeze(-1).reshape(
                self.mod.weight.shape)
        else:
            Q_ux = calc_q_ux_conv(self.mod.weight, Q_u)
            big_k = calc_big_k_conv(Q_uu, Q_ux)
            term2 = calc_term2_conv(big_k, inp - self.x_old).reshape(self.mod.weight.shape)
        self.mod.weight += opt.lr*opt.lrddp*(term2)
        out = self.forward_wo_train(inp)
        return out


def calc_q_ux_fc(q_u, q_x):
    return torch.einsum('xy,zt->xt', q_u.flatten(start_dim=0).unsqueeze(-1), torch.transpose(q_x,0,1))

def calc_q_ux_conv(W, q_u):
    return torch.einsum('cdef,cdef->cdef',W,q_u)

def calc_small_k(q_uu, q_u):
    return -q_u/q_uu

def calc_big_k_conv(q_uu, q_ux):
    return -q_ux/q_uu
        
def calc_big_k_fc(q_uu, q_ux):
    return -torch.einsum('x,xt->xt', 1/q_uu.flatten(start_dim=0), q_ux)

def calc_term2_conv(big_k, delta_x):
    conv_size = big_k.shape[-1]
    b, c, h, w = delta_x.shape
    t = torch.as_strided(delta_x, size=(b, h - conv_size+1, w - conv_size+1, c, conv_size,conv_size), stride=(c * h * h, w, 1, h * w, w, 1)).mean(dim=0)
    t = t.flatten(start_dim=0, end_dim=1).flatten(start_dim=-2, end_dim=-1)
    H = t.shape[0]
    big_k = big_k.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True).repeat((1,1,1,H))
    out = torch.einsum('ncxh,hcd->ncdx', big_k, t)
    return out
    

import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor
from typing import List, Optional

def update(params: List[Tensor],
            grads: List[Tensor],
            square_avgs: List[Tensor],
            *,
            lr: float,
            alpha: float,
            eps: float):

    for i, param in enumerate(params):
        grad = grads[i]
        square_avg = square_avgs[i]

        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

        avg = square_avg.sqrt().add_(eps)

        param.addcdiv_(grad, avg, value=-lr)


class DDPNOPT(Optimizer):
    def __init__(self, model, lr=1e-2, lrddp = 1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        self.lr = lr
        self.lrddp = lrddp
        self.model = model
        self.eps = eps
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(DDPNOPT, self).__init__(model.parameters(), defaults)

    def __setstate__(self, state):
        super(DDPNOPT, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            square_avgs = []
            grad_avgs = []

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avgs.append(state['square_avg'])

                state['step'] += 1


        update(params_with_grad,
                  grads,
                  square_avgs,
                  lr=self.lr,
                  alpha=group['alpha'],
                  eps=group['eps'])
        if self.lrddp: self.model(update=True, opt=self)
