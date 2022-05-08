# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/4/27 8:17
import os

import numpy as np


def stat_hist(exp_path):
    print(exp_path)
    titles, rhos, rl2s = [], [], []
    for i in range(1, 7):
        hist_path = os.path.join(exp_path, f'{i}/weights/history.npz')
        if os.path.exists(hist_path):
            history = np.load(hist_path)
            hist_test = history['test']
            rhos.append(hist_test[:, 1].max())
            rl2s.append(hist_test[:, -1].min())

            print(f'{i:7d} &', end='')

    def tex_print(var):
        for idx, v in enumerate(var):
            print(f'{v:7.4f}', end=' & ')

        print(f'{sum(var) / len(var):.4f}')

    print('   avg')

    tex_print(rhos)
    tex_print(rl2s)


if __name__ == '__main__':
    exp_paths = ['/home/zkl/Documents/Codes/AQA/exps/Seven/ours',
                 '/home/zkl/Documents/Codes/AQA/exps/Seven/ours_',
                 '/home/zkl/Documents/Codes/AQA/exps/Seven/ours_group_4',
                 '/home/zkl/Documents/Codes/AQA/exps/Seven/ours_clip_add',
                 '/home/zkl/Documents/Codes/AQA/exps/Seven/ours_scene_5',
                 '/home/zkl/Documents/Codes/AQA/exps/Seven/ours_scene_9',
                 '/home/zkl/Documents/Codes/AQA/exps/Seven/ours_scene_10',
                 '/home/zkl/Documents/Codes/AQA/exps/Seven/ours_full',
                 '/home/zkl/Documents/Codes/AQA/exps/Seven/ours_group_16',
                 '/home/zkl/Documents/Codes/AQA/exps/Seven/ours_lr_5e4',
                 '/home/zkl/Documents/Codes/AQA/exps/Seven/ours_lr_5e5',
                 '/home/zkl/Documents/Codes/AQA/exps/Seven/ours_decay_0',
                 '/home/zkl/Documents/Codes/AQA/exps/Seven/ours_decay_1e3',
                 '/home/zkl/Documents/Codes/AQA/exps/Seven/ours_decay_1e4',
                 '/home/zkl/Documents/Codes/AQA/exps/Seven/ours_decay_1e2',
                 '/home/zkl/Documents/Codes/AQA/exps/Seven/ours_lr_5e5_decay_0',
                 ]

    for exp_path in exp_paths:
        stat_hist(exp_path)
