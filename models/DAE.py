# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/4/2 20:56
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import misc


class DAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DAE, self).__init__()

        self.fc1 = nn.Linear(in_channels, 256)
        self.fch = nn.Linear(256, 128)
        self.fc2_mean = nn.Linear(128, out_channels)
        self.fc2_logvar = nn.Linear(128, out_channels)

    def encode(self, x):
        h0 = F.relu(self.fc1(x))
        h1 = F.relu(self.fch(h0))
        return self.fc2_mean(h1), self.fc2_logvar(h1)

    def reparametrization(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(mu.device)
        z = mu + std * esp
        return z

    def forward(self, x):
        x = x.mean(-1)

        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)

        return z, mu, logvar.mul(0.5).exp_()

if __name__ == '__main__':
    x = torch.randn((8, 1024, 10))
    model = DAE(1024, 1)
    y = model(x)
    print(f'{misc.count_params(model):,d}')