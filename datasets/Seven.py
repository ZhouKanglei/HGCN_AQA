# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/4/26 9:53
import torch
import scipy.io
import os
import random
from utils import misc
from PIL import Image


class Seven_Dataset(torch.utils.data.Dataset):
    """AQA-7 dataset"""
    def __init__(self, args, subset, transform):
        random.seed(args.seed)
        self.subset = subset
        self.transforms = transform

        classes_name = ['diving', 'gym_vault', 'ski_big_air', 'snowboard_big_air', 'sync_diving_3m', 'sync_diving_10m']
        self.sport_class = classes_name[args.class_idx - 1]

        self.class_idx = args.class_idx  # sport class index(from 1 begin)
        self.score_range = args.score_range
        # file path
        self.data_root = args.data_root
        self.data_path = os.path.join(self.data_root, '{}-out'.format(self.sport_class))
        self.split_path = os.path.join(self.data_root, 'Split_4', f'split_4_{self.subset}_list.mat')
        self.split = scipy.io.loadmat(self.split_path)[f'consolidated_{self.subset}_list']
        self.split = self.split[self.split[:, 0] == self.class_idx].tolist()
        self.dataset = self.split.copy()

        # setting
        self.length = args.frame_length


    def load_video(self, idx):
        video_path = os.path.join(self.data_path, '%03d' % idx)
        video = [Image.open(os.path.join(video_path, 'img_%05d.jpg' % (i + 1))) for i in range(self.length)]
        return self.transforms(video)


    def __getitem__(self, index):
        sample_1 = self.dataset[index]
        assert int(sample_1[0]) == self.class_idx
        idx = int(sample_1[1])

        data = {}

        data['video'] = self.load_video(idx)
        data['final_score'] = misc.normalize(sample_1[2], self.class_idx, self.score_range)

        return data, index


    def __len__(self):
        return len(self.dataset)