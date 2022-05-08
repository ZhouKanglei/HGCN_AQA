# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/4/7 16:34

import os
import math
import numpy as np
import seaborn as sns
import pandas as pd

from matplotlib import pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['savefig.dpi'] = 300
plt.rcParams["savefig.format"] = 'pdf'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['text.usetex'] = True



class vis(object):
    """visualizer"""

    def __init__(self, fig_path):
        super(vis, self).__init__()

        self.fig_path = fig_path
        os.makedirs(self.fig_path, exist_ok=True)

    def print_log(self, file):
        print(f'Save fig to {file}.')

    def vis_hist(self, root_dir, affix=''):
        true = np.load(os.path.join(root_dir, 'true.npy'))
        pred = np.load(os.path.join(root_dir, 'pred.npy'))

        plt.figure(figsize=(9, 3))

        x = np.arange(len(true))
        width = 0.35  # the width of the bars

        plt.bar(x - width / 2, true.reshape(-1, ), width, label='True')
        plt.bar(x + width / 2, pred.reshape(-1, ), width, label='Pred')

        plt.legend()
        plt.grid()

        fig_file = os.path.join(self.fig_path, f'hist{affix}.pdf')
        plt.savefig(fig_file)
        self.print_log(fig_file)

    def vis_err_bar(self, root_dir, affix=''):
        true = np.load(os.path.join(root_dir, 'true.npy'))
        pred = np.load(os.path.join(root_dir, 'pred.npy'))

        plt.figure(figsize=(12, 3))

        x = np.arange(len(true))
        err = (pred - true).reshape(-1, )
        plt.bar(x, err)

        for idx, e in enumerate(err):
            if abs(e) >= 20:
                print(idx, pred[idx], true[idx])
                plt.bar(idx, e, label=f'sample \#{idx}')

        plt.grid()
        plt.legend()

        plt.xlabel('\# of sample')
        plt.ylabel('Error ($\hat{s} - s$)')

        fig_file = os.path.join(self.fig_path, f'err_bar{affix}.pdf')
        plt.savefig(fig_file)
        self.print_log(fig_file)

    def vis_corr(self, root_dir, affix=''):
        true = np.load(os.path.join(root_dir, 'true.npy'))
        pred = np.load(os.path.join(root_dir, 'pred.npy'))

        plt.figure(figsize=(3, 3))

        colors = np.random.rand(len(true))
        plt.scatter(true, pred, c=colors, label='($s$, $\hat{s}$)', marker='.')
        parameter = np.polyfit(true.reshape(-1, ), pred.reshape(-1, ), 1)
        plt.plot(true, true * parameter[0] + parameter[1], '-.',
                 label='$\hat{s}=' + f'{parameter[0]:.2f}' +
                       's + ' + f'{parameter[1]:.2f}' + '$', color='C0')
        plt.plot(true, true, label='$\hat{s}=s$', color='C1')

        print(parameter[0])

        plt.xlabel('Ground-truth score $s$')
        plt.ylabel('Predicted score $\hat{s}$')

        plt.grid()
        plt.legend()

        fig_file = os.path.join(self.fig_path, f'corr{affix}.pdf')
        plt.savefig(fig_file)
        self.print_log(fig_file)

    def vis_history(self, history):
        self.vis_cmp(history['train'][:, 1], history['test'][:, 1], 'Epoch',
                     'Spearman rank coefficient $\\rho$', 'rho.pdf')
        self.vis_cmp(history['train'][:, 2], history['test'][:, 2], 'Epoch',
                     '$p$-value', 'p_value.pdf')
        self.vis_cmp(history['train'][:, 3], history['test'][:, 3], 'Epoch',
                     'MSE', 'mse.pdf')
        self.vis_cmp(history['train'][:, 4], history['test'][:, 4], 'Epoch',
                     '$\mathrm{R}$-$\ell_2$', 'RL2.pdf')

    def vis_cmp(self, train, test, xlabel, ylabel, fig_name, c0='C0', c1='C1'):
        plt.figure(figsize=(3, 3))

        plt.plot(train, '-.', label='Train', color=c0)
        plt.plot(test, label='Test', color=c1)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.legend()
        plt.grid()

        fig_file = os.path.join(self.fig_path, fig_name)
        plt.savefig(fig_file)
        self.print_log(fig_file)

    def save_fig(self, video, affix=''):
        # rho
        plt.figure(figsize=(3, 3))

        frame_idx = [0, 10, 20, 30, 40, 50, 60, 70, 80, 87]
        video = np.uint8(video)

        for i in frame_idx:
            v = video[i]
            if v.shape[0] > v.shape[1]:
                offset = int(v.shape[0] - v.shape[1]) // 2
                v = v[offset:v.shape[0] - offset, :, :]
            else:
                offset = int(v.shape[1] - v.shape[0]) // 2
                v = v[:, offset:v.shape[1] - offset, :]

            img = Image.fromarray(v)

            fig_file = os.path.join(self.fig_path, f'{affix}{i}.jpg')
            img.save(fig_file)
            self.print_log(fig_file)

    def norm_distribution(self, mu, sigma, affix=''):
        plt.figure(figsize=(4, 4), facecolor='#8FBBD9')

        if sigma > 1.0e-03:
            x1 = np.linspace(mu - 10 * sigma, mu + 10 * sigma, 100)
        else:
            x1 = np.linspace(mu - 0.01, mu + 0.01, 100)

        y1 = np.zeros_like(x1)
        y2 = np.exp(-1 * ((x1 - mu) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)

        plt.plot(x1, y2, 'C0', label=f'$\mu = {mu:.4f} \\\\ \sigma = {sigma * 1000:.2f} \\times 10^' + '{-3}$')
        plt.fill_between(x1, y1, y2, color='C0', alpha=0.5)

        plt.xlabel('Score')
        plt.ylabel('Probability Density')

        plt.legend()
        plt.grid()

        plt.legend()
        plt.grid()

        fig_file = os.path.join(self.fig_path, f'mu_sigma{affix}.pdf')
        plt.savefig(fig_file)
        self.print_log(fig_file)

    def vis_heatmap(self, att_map, mask=None, affix=''):
        plt.figure()
        font_size = 18

        x_tick = [i + 1 for i in range(att_map.shape[0])]
        y_tick = [i + 1 for i in range(att_map.shape[1])]
        data = {}

        print('--------')
        for i in range(len(x_tick)):
            data[x_tick[i]] = att_map[i]

        print(data)
        pd_data = pd.DataFrame(data, index=y_tick, columns=x_tick)

        ax = sns.heatmap(pd_data, cmap=plt.cm.Oranges, annot=True, annot_kws={'fontsize': font_size - 2},
                    cbar=True, square=True, linewidths=0.1, linecolor='black', fmt='.2f', mask=mask)

        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=font_size)

        plt.xticks(fontsize=font_size + 2, rotation=0)
        plt.yticks(fontsize=font_size + 2)
        plt.xlabel('\# of scene', fontsize=font_size + 2)
        plt.ylabel('\# of scene', fontsize=font_size + 2)

        fig_file = os.path.join(self.fig_path, f'att{affix}.pdf')
        plt.savefig(fig_file)
        self.print_log(fig_file)

    def vis_tsne(self, X, y, affix=''):
        plt.figure(figsize=(4, 4))

        X_embed = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X)

        for i in sorted(set(y)):
            score_range = f'{i * 10:d}--{i * 10 + 10:d}'
            score_range = f'{i}'
            print(score_range)
            plt.scatter(X_embed[y == i, 0], X_embed[y == i, 1], label=f'{score_range}', marker='.')

        plt.xlabel('$x$')
        plt.ylabel('$y$')

        plt.grid()
        plt.legend()

        fig_file = os.path.join(self.fig_path, f'tsne{affix}.pdf')
        plt.savefig(fig_file)
        self.print_log(fig_file)

    def vis_cmp_corr(self, root_dirs, methods, styles, affix=''):

        def plot_error(i, root_dir, label='', style=''):
            print(root_dir)

            true = np.load(os.path.join(root_dir, 'true.npy'))
            pred = np.load(os.path.join(root_dir, 'pred.npy'))

            err = np.array(abs(true - pred))
            idx = []
            rate = []

            for i in np.linspace(0, 20, 21):
                rate.append(sum(err <= i) / len(err))
                idx.append(i)

            if label == 'Ours w/ DD':
                x = np.linspace(0, 20, 21)
                y1 = x * 0
                y2 = np.array(rate).reshape(y1.shape)
                plt.fill_between(x, y1, y2, color='C3', alpha=0.5)

            plt.plot(idx, rate, style, label=label)

        plt.figure(figsize=(4, 4))
        for i in range(len(root_dirs)):
            plot_error(i, root_dirs[i], methods[i], styles[i])

        plt.xlabel('Error threshold')
        plt.ylabel('Cumulative score (\%)')
        plt.grid()
        plt.legend()

        fig_file = os.path.join(self.fig_path, f'cmp{affix}.pdf')
        plt.savefig(fig_file)
        self.print_log(fig_file)



if __name__ == '__main__':
    root_dir = '/home/zkl/Documents/Codes/AQA/exps/MTL/msgcn_shot_add_trans_adp_3/weights'
    out_dir = '/home/zkl/Documents/Codes/AQA/exps/figs'

    vis = vis(fig_path=out_dir)


    root_dirs = ['/home/zkl/Documents/Codes/AQA/exps/MTL/gcn_gate/weights',
                 '/home/zkl/Documents/Codes/AQA/exps/MTL/gcn_mean/weights',
                 '/home/zkl/Documents/Codes/AQA/exps/MTL/gcn_ada_gate/weights',
                 '/home/zkl/Documents/Codes/AQA/exps/MTL/msgcn_shot_add_trans_adp_5/weights']

    # methods = ['I3D + MLP w/ DD', 'CoRe w/ DD', 'Ours', 'Ours w/ DD']
    # styles = ['--o', '-x', '--^', '-+']
    #
    # vis.vis_corr(root_dir, '_ours')
    # vis.vis_cmp_corr(root_dirs, methods, styles)

    # feature_dir = '/home/zkl/Documents/Codes/AQA/exps/mix'
    # data = np.load(os.path.join(feature_dir, 'features.npz'))
    # X = data['I3D'].mean(-1)
    # shot = data['shot']
    # scene = data['scene']
    # diff = np.int32(data['diff'])
    # v = data['v']
    # y = data['y'].reshape(-1, )
    # y = np.int32(y // 10)
    # y = np.int32(y / diff)
    # vis.vis_tsne(X, y, '_I3D')
    # vis.vis_tsne(shot, y, '_shot')
    # vis.vis_tsne(scene, y, '_scene')
    # vis.vis_tsne(v, y, '_v')

    # vis.vis_hist(root_dir)
    vis.vis_err_bar(root_dir)
    #
    # history = {}
    # history['train'] = np.random.rand(100, 5)
    # history['test'] = np.random.rand(100, 5)
    # history = np.load(os.path.join(root_dir, 'history.npz'))
    # vis.vis_history(history)

    # vis.norm_distribution(91.2726, 2.4874e-04, affix='_13')
    # vis.norm_distribution(92.4048, 1.1275e-04, affix='_224')
    # vis.norm_distribution(49.5849, 2.4273e-03, affix='_332')
    # vis.norm_distribution(27.5098, 4.2281e-02, affix='_340')

    # vis.save_fig(video[-1].cpu().numpy())

    # vis.vis_heatmap(np.random.randn(5, 5))
