import numpy as np
import torch.nn as nn
import torch

def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__

class Step(nn.Module):
    def __init__(self, module, act, lr=1e-3, alpha=0.99, eps=1e-8):
        super(Step, self).__init__()
        self.x = None
        self.mod = module
        self.type = _layer_type(module)
        self.act = act

        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.second_derv = {'weight':0, 'bias':0}
    def forward(self, inp):
        self.x = nn.Parameter(inp)
        return self.act((self.mod(self.x)+self.mod(inp))/2)
    def forward_wo_train(self, inp):
        return self.act(self.mod(inp))
    def update_second_derv(self):
        self.second_derv['weight'] = self.alpha*self.second_derv['weight'] \
                                     + (1-self.alpha)*self.mod.weight.grad*self.mod.weight.grad
        self.second_derv['bias'] = self.alpha * self.second_derv['bias'] \
                                     + (1 - self.alpha) * self.mod.bias.grad * self.mod.bias.grad

    @torch.no_grad()
    def update(self, inp):
        inp *= 1
        Q_u = self.mod.weight.grad1
        Q_uu = self.alpha * self.second_derv['weight'] + (1 - self.alpha) * Q_u * Q_u
        Q_uu = torch.sqrt(Q_uu)+self.eps
        Q_x = self.x.grad
        if self.type=='Linear':
            Q_ux = calc_q_ux_fc(Q_u, Q_x.unsqueeze(-1))
            big_k = calc_big_k(Q_uu, Q_ux)
            term2 = torch.einsum('bxy,bzt->bxt', big_k, (inp - self.x).unsqueeze(-1)).squeeze(-1).mean(dim=0).reshape(
                self.mod.weight.shape)
        else:
            Q_ux = calc_q_ux_conv(self.mod.weight, Q_u)
            big_k = calc_big_k(Q_uu, Q_ux)
            term2 = calc_term2_conv(big_k, inp - self.x).reshape(self.mod.weight.shape)
        small_k = calc_small_k(Q_uu, Q_u)
        term1 = small_k.mean(dim=0)
        term2 *= 1/100
        self.mod.weight += self.lr*(term1+term2)
        self.update_second_derv()
        self.mod.bias -= (self.lr * self.mod.bias.grad) / (torch.sqrt(self.second_derv['bias']) + self.eps)
        out = self.forward_wo_train(inp)
        return out


def calc_q_ux_fc(q_u, q_x):
    return torch.einsum('bxy,bzt->bxt', q_u.flatten(start_dim=1).unsqueeze(-1), torch.transpose(q_x,1,2))

def calc_q_ux_conv(W, q_u):
    return torch.einsum('cdef,bcdef->bcdef',W,q_u)

def calc_small_k(q_uu, q_u):
    return -q_u/q_uu

def calc_big_k(q_uu, q_ux):
    try:
        return -torch.einsum('bx,bxt->bxt', 1/q_uu.flatten(start_dim=1), q_ux)
    except:
        return -q_ux/q_uu

def calc_term2_conv(big_k, delta_x):
    b, c, h, w = delta_x.shape
    t = torch.as_strided(delta_x, size=(b, h - 2, w - 2, c, 3, 3), stride=(c * h * h, w, 1, h * w, w, 1))
    t = t.flatten(start_dim=1, end_dim=2).flatten(start_dim=-2, end_dim=-1)
    H = t.shape[1]
    big_k = big_k.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True).repeat((1,1,1,1,H))
    out = torch.einsum('bncxh,bhcd->bncdx', big_k, t)
    return out.mean(dim=0)