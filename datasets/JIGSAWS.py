# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/5/7 20:26
import glob
import os
import pickle
import random

import torch
from PIL import Image

class JIGSAWS_Dataset(torch.utils.data.Dataset):
    """JIGSAWS dataset"""

    def __init__(self, args, subset, transform):
        # dataset
        self.subset = subset
        self.transform = transform

        self.length = args.frame_length

        classes_name = ['Needle_Passing', 'Suturing', 'Knot_Tying']
        self.cls = classes_name[args.class_idx - 1]

        # file path
        self.data_root = args.data_root
        self.label_dict = self.read_pickle(args.label_path)
        self.cv_file = self.read_pickle(args.split_path)

        # load fold
        self.load_fold(args.fold)

    def load_fold(self, fold):
        folds = [0, 1, 2, 3]
        if self.subset == 'train':
            folds.pop(fold)
        else:
            folds = [fold]

        self.name_list = []
        all_list = self.cv_file[self.cls]
        for fold in folds:
            for vid in all_list[fold]:
                self.name_list.append(vid + '_capture1')  # only loads left view
                self.name_list.append(vid + '_capture2')  # only loads right view

    def read_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)

        return pickle_data

    def load_video(self, video_file_name, phase):
        image_list = sorted(
            (glob.glob(os.path.join(self.data_root, f'{video_file_name}', '*.jpg')))
        )
        video = []
        true_length = len(image_list)
        for i in range(self.length):
            if i < true_length:
                video.append(Image.open(image_list[i]))
            else:
                video.append(Image.open(image_list[true_length - 1]))
        # video = [Image.open(image_list[i]) for i in range(self.length)]

        return self.transform(video)

    def __getitem__(self, index):

        sample = self.name_list[index]

        data = {}
        data['video'] = self.load_video(sample, self.subset)
        data['final_score'] = sum(self.label_dict[sample[:-9]])

        return data, index

    def __len__(self):
        return len(self.name_list)
