# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/4/3 10:27
import torch
from torch import nn

from utils import misc
from graph.graph import Graph


class GATLayer(nn.Module):
    """GAT layer"""

    def __init__(self, in_channels, out_channels, mask=None, num_groups=1, gate=False):
        super(GATLayer, self).__init__()

        self.mask = mask

        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=(1,), groups=num_groups),
            nn.Identity() if gate else nn.ReLU()
        )

        self.q = nn.Conv1d(in_channels, 1, kernel_size=(1,), bias=False)
        self.k = nn.Conv1d(in_channels, 1, kernel_size=(1,), bias=False)

        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        mask = None if self.mask == None else self.mask.to(x.device)

        q, k = self.q(x), self.k(x).transpose(1, 2)
        att = self.leakyrelu(q + k) if mask == None else self.leakyrelu(q + k) * mask
        att = self.softmax(att)

        agg = torch.einsum('b c n, b m n -> b c m', x, att)
        out = self.mlp(agg)

        return out


class DirectGraph(nn.Module):
    """Direct graph"""

    def __init__(self, in_channels, num_groups=1):
        super(DirectGraph, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=(1,), groups=num_groups),
            nn.ReLU()
        )

    def forward(self, x):
        in_fea = self.mlp(x)
        out_fea = in_fea.transpose(2, 1)

        graph = in_fea - out_fea
        graph = torch.where(graph > 0 * graph, graph, graph * 0)
        
        return graph

class GCNLayer(nn.Module):
    """GCN layer"""

    def __init__(self, in_channels, out_channels, A, mask=None, num_groups=1,
                 gate=False, direct=False):
        super(GCNLayer, self).__init__()

        self.A = A
        self.mask = torch.zeros_like(A) + 1 if mask == None else mask
        self.graph = DirectGraph(in_channels) if direct else None

        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=(1,), groups=num_groups),
            nn.Identity() if gate else nn.ReLU()
        )

    def forward(self, x):
        if self.graph == None:
            A = self.A.to(x.device) * self.mask.to(x.device)
            agg = torch.einsum('b c n, n m -> b c m', x, A)
        else:
            A = self.A.to(x.device) * self.mask.to(x.device) * self.graph(x)
            agg = torch.einsum('b c n, b n m -> b c m', x, A)

        out = self.mlp(agg)

        return out


class DiffAgg(nn.Module):
    """GCN aggregation"""

    def __init__(self, in_channels, A, mask=None):
        super(DiffAgg, self).__init__()

        self.gcn = GCNLayer(in_channels, 1, A, mask, 1, True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        att = self.gcn(x).squeeze(1)
        att = self.softmax(att)
        out = torch.einsum('b c n, b n -> b c', x, att)

        return out


class GateGCNLayer(nn.Module):
    """Gate GCN layer"""

    def __init__(self, in_channels, out_channels, A, mask=None, num_groups=1):
        super(GateGCNLayer, self).__init__()

        self.gcn1 = GCNLayer(in_channels, out_channels, A, mask, num_groups)
        # self.gate = GATLayer(in_channels, out_channels, None, num_groups)
        # self.gate = GCNLayer(in_channels, out_channels, A, num_groups, True)

        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=(1,), groups=num_groups),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.mlp(x) + self.gcn1(x)

        return out


class GCNRegressor(nn.Module):
    """ GCN regressor """

    def __init__(self, in_channels, out_channels, num_node=10, num_k=2, num_groups=(4, 4, 4)):
        super(GCNRegressor, self).__init__()

        graph = Graph(num_node, num_k)
        A = nn.Parameter(torch.Tensor(graph.norm_adj), requires_grad=True)
        mask = torch.Tensor(graph.adj)

        self.fusion = nn.Sequential(
            GateGCNLayer(in_channels, 512, A, mask, num_groups=num_groups[0]),
            GateGCNLayer(512, 256, A, mask, num_groups=num_groups[1]),
            GateGCNLayer(256, 128, A, mask, num_groups=num_groups[2]),
            DiffAgg(128, A, mask),
        )

        self.fc2_mean = nn.Linear(128, out_channels)
        self.fc2_logvar = nn.Linear(128, out_channels)

    def forward(self, x):
        mid = self.fusion(x)

        mu, logvar = self.fc2_mean(mid), self.fc2_logvar(mid)

        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(x.device)
        out = mu + std * esp

        return out, mu, std


if __name__ == '__main__':
    x = torch.randn((8, 1024, 10))
    model = GCNRegressor(1024, 1)
    y = model(x)
    print(f'{misc.count_params(model):,d}')


