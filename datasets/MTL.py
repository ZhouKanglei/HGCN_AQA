# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/4/2 16:19
import glob
import os
import pickle
import random

import torch
from PIL import Image


class MTL_Dataset(torch.utils.data.Dataset):
    """MTL-AQA dataset"""
    def __init__(self, args, subset, transform):
        # dataset
        self.subset = subset
        self.transform = transform
        # file path
        self.label_path = args.label_path
        self.label_dict = self.read_pickle(self.label_path)

        if self.subset == 'test':
            self.split_path_test = args.test_split
            self.dataset = self.read_pickle(self.split_path_test)
        else:
            self.split_path = args.train_split
            self.dataset = self.read_pickle(self.split_path)

        self.data_root = args.data_root
        # setting
        self.temporal_shift = [args.temporal_shift_min, args.temporal_shift_max]
        self.length = args.frame_length

    def read_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)

        return pickle_data


    def load_video(self, video_file_name, phase):
        image_list = sorted(
            (glob.glob(os.path.join(self.data_root, f'{video_file_name[0]:02d}', '*.jpg')))
        )
        end_frame = self.label_dict.get(video_file_name).get('end_frame')
        if phase == 'train':
            temporal_aug_shift = random.randint(self.temporal_shift[0], self.temporal_shift[1])
            if end_frame + temporal_aug_shift > self.length or end_frame + temporal_aug_shift < len(image_list):
                end_frame = end_frame + temporal_aug_shift
        start_frame = end_frame - self.length

        video = [Image.open(image_list[start_frame + i]) for i in range(self.length)]
        return self.transform(video)


    def __getitem__(self, index):
        sample = self.dataset[index]

        data = {}
        data['video'] = self.load_video(sample, self.subset)
        data['final_score'] = self.label_dict.get(sample).get('final_score')
        data['difficulty'] = self.label_dict.get(sample).get('difficulty')
        data['judge_scores'] = self.label_dict.get(sample).get('judge_scores')

        return data, index

    def __len__(self):
        return len(self.dataset)
