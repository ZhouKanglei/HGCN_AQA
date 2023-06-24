# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/4/16 8:48
import math

import torch
import torch.nn as nn

from graph import Graph
from utils import misc


class attAdj(nn.Module):
    """Graph adj with self-attention"""

    def __init__(self, in_channels, out_channels, num_nodes, num_groups=4, softmax=True):
        super(attAdj, self).__init__()
        out_channels = out_channels
        self.out_channels = out_channels
        # self-attention
        self.conv_q = nn.Conv1d(in_channels, out_channels, kernel_size=(1,), groups=num_groups, bias=False)
        self.conv_k = nn.Conv1d(in_channels, out_channels, kernel_size=(1,), groups=num_groups, bias=False)
        self.softmax = nn.Softmax(-1) if softmax else nn.Identity()

        self.C = nn.Parameter(torch.eye(num_nodes), requires_grad=True)
        self.alpha = nn.Parameter(torch.eye(1), requires_grad=True)

    def forward(self, x):
        q, k = self.conv_q(x), self.conv_k(x)
        B = torch.einsum('b c n, b c m -> b n m', q, k) / math.sqrt(self.out_channels)
        C = self.C.to(x.device).unsqueeze(0).repeat_interleave(x.shape[0], dim=0)
        B = self.softmax(self.alpha * B + C)

        return B


class clipRefine(nn.Module):
    """Clip refinement"""

    def __init__(self, in_channels, out_channels, num_groups=1, num_clips=10, num_k=3, factor=2):
        super(clipRefine, self).__init__()
        mid_channels = in_channels // factor
        # information state generation
        self.state_gen = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=(1,), groups=num_groups),
            nn.ReLU()
        )
        # graph initialization
        graph = Graph(num_clips, 2)
        self.A = torch.Tensor(graph.adj)
        self.I = torch.eye(num_clips)

        # graph construction
        self.graph_gen = nn.Tanh()
        # information transfer
        self.info_trans = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=(1,), groups=num_groups),
            nn.ReLU()
        )

    def forward(self, x):
        A = self.A.to(x.device)
        I = self.I.to(x.device)
        state = self.state_gen(x)
        in_state, out_state = state.unsqueeze(-1), state.unsqueeze(-2)
        A_sta = (self.graph_gen((in_state - out_state)).mean(1) + I) * A
        h = torch.einsum('b c n, b n m -> b c m', x, A_sta)
        y = self.info_trans(h)

        return y


class shotGCN(nn.Module):
    """shot graph convolutional network"""

    def __init__(self, in_channels, out_channels, A, num_groups=4, factor=2):
        super(shotGCN, self).__init__()
        self.mid_channels = mid_channels = in_channels // factor
        # graph
        self.A = A
        self.adp_adj = attAdj(in_channels, mid_channels, A.shape[0], num_groups=num_groups, softmax=False)
        self.softmax = nn.Softmax(-1)
        # aggregation
        self.shot_agg = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=(1,), groups=num_groups),
            nn.ReLU()
        )

    def forward(self, x):
        A = self.A.to(x.device)
        B = self.adp_adj(x)
        A = self.softmax(A + B)
        h = torch.einsum('b c n, b n m -> b c m', x, A)
        y = self.shot_agg(h)

        return y


class diffGCN(nn.Module):
    """Diff graph convolutional network"""

    def __init__(self, in_channels, out_channels, A, num_groups=1, factor=1):
        super(diffGCN, self).__init__()
        self.mid_channels = mid_channels = in_channels // factor
        # self-attention
        self.A = A
        self.adp_adj = attAdj(in_channels, mid_channels, A.shape[0], num_groups=num_groups, softmax=False)
        self.softmax = nn.Softmax(-1)
        # aggregation
        self.shot_agg = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=(1,), groups=num_groups),
            nn.ReLU(),
        )

        self.S = nn.Parameter(torch.ones((out_channels, A.shape[0])), requires_grad=True)
        self.alpha = nn.Parameter(torch.eye(1), requires_grad=True)

    def forward(self, x):
        A = self.A.to(x.device)
        B = self.adp_adj(x)
        A = self.softmax(A + B)
        h = torch.einsum('b c n, b n m -> b c m', x, A)
        T = self.shot_agg(h)
        S = self.S.to(x.device)
        out = S + self.alpha * T

        return out


class sceneConst(nn.Module):
    """Scene construction"""

    def __init__(self, in_channels, out_channels, num_groups=4,
                 num_shots=10, num_k=3, num_scenes=3, factor=4):
        super(sceneConst, self).__init__()
        # undirected graph generation
        graph = Graph(num_shots, num_k)
        self.A = A = torch.Tensor(graph.adj)

        self.shot_gcn = shotGCN(in_channels, out_channels, A, num_groups, factor)
        self.trans_gcn = diffGCN(in_channels, num_scenes, A, num_groups=1, factor=factor)

        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        h = self.shot_gcn(x)

        T = self.trans_gcn(x)
        y = torch.einsum('b c n, b m n -> b c m', h, T)

        return y


class motionGCN(nn.Module):
    """Motion graph aggregation"""

    def __init__(self, in_channels, out_channels, num_groups=4,
                 num_scenes=10, factor=4):
        super(motionGCN, self).__init__()
        mid_channels = in_channels // factor
        # graph initialization
        self.A = torch.zeros((num_scenes, num_scenes))
        for i in range(num_scenes):
            for j in range(num_scenes):
                self.A[i, j] = 1 if j <= i else 0

        self.adp_A = attAdj(in_channels, mid_channels, num_scenes, num_groups=num_groups, softmax=False)
        self.softmax = nn.Softmax(-1)
        # aggregation
        self.mot_agg = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=(1,), groups=num_groups),
            nn.ReLU()
        )

    def forward(self, x):
        A = self.A.to(x.device)
        B = self.adp_A(x)
        A_sum = self.softmax(A * B)

        h = torch.einsum('b c n, b n m -> b c m', x, A_sum)
        y = self.mot_agg(h)

        return y


class mlp(nn.Module):
    """mlp"""

    def __init__(self, in_channels, out_channels, num_groups):
        super(mlp, self).__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=(1,), groups=num_groups),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)


class hgcn(nn.Module):
    """Motion scene graph aggregation"""

    def __init__(self, in_channels=1024, out_channels=1, num_groups=8, num_clips=10,
                 num_scenes=7, num_k=3, factor=4, a=True, b=True, c=False):
        """
        :param in_channels:
        :param out_channels:
        :param num_groups:
        :param num_clips:
        :param num_scenes:
        :param num_k:
        :param factor:
        :param a: shot refinement T or F
        :param b: scene graph construction T or F
        :param c: motion graph aggregation T or F
        """
        super(hgcn, self).__init__()
        self.num_scenes = num_scenes
        # Clip refinement
        self.shot_gen = clipRefine(in_channels, 512, num_groups, num_clips, num_k, factor) if a \
            else mlp(in_channels, 512, num_groups)
        # Scene construction
        self.scene_gen = sceneConst(512, 256, num_groups, num_shots=num_clips,
                                    num_k=num_k, num_scenes=num_scenes, factor=factor) if b \
            else mlp(512, 256, num_groups)
        # Motion graph aggregation
        self.mot_agg = motionGCN(256, 128, num_groups, num_scenes, factor) if c \
            else mlp(256, 128, num_groups)
        self.c = c
        # output
        self.fc2_mean = nn.Conv1d(128, out_channels, kernel_size=(1,))
        self.fc2_logvar = nn.Conv1d(128, out_channels, kernel_size=(1,))

    def reparameter(self, v):
        # Score distribution regression
        mu, logvar = self.fc2_mean(v), self.fc2_logvar(v)

        # Re-parameterization
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(v.device)

        return mu, std, esp

    def forward(self, x, phase='train'):
        shot = self.shot_gen(x)
        scene = self.scene_gen(shot)
        v = self.mot_agg(scene)  # video-level representation

        mu, std, esp = self.reparameter(v)
        scores = mu + std * esp
        out = scores.mean(-1) * self.num_scenes

        if phase == 'test':
            return out, mu, std, shot.mean(-1), scene.mean(-1), v, scores
        else:
            return out, mu, std


if __name__ == '__main__':
    x = torch.randn((8, 1024, 10))
    model = hgcn(1024, 1, num_groups=8)
    y = model(x)
    print(f'{misc.count_params(model):,d}')
