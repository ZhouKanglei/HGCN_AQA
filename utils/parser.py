import os

import torch
import yaml
import argparse
import shutil

from utils import misc
from utils.misc import str2bool


class Parser(object):
    """Args parser"""

    def __init__(self):

        self.get_args()

        self.setup()

        self.check_args()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--benchmark', type=str, choices=['MTL', 'Seven', 'JIGSAWS'], help='dataset', default='MTL')
        parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
        parser.add_argument('--fix_bn', type=str2bool, default=True)
        parser.add_argument('--resume', type=str2bool, default=False,
                            help='autoresume training from exp dir (interrupted by accident)')
        parser.add_argument('--ckpts', type=str, default=None, help='test used ckpt path')
        parser.add_argument('--class_idx', type=int, default=1, choices=[1, 2, 3, 4, 5, 6], help='class idx in Seven')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='device')
        parser.add_argument('--device', type=list, default=[0, 1, 2, 3], help='device')
        parser.add_argument('--seed', type=int, default=1024, help='random seed')
        parser.add_argument('--model', type=str, default='models.DAE', help='random seed')
        parser.add_argument('--phase', type=str, default='train', help='train or test')
        parser.add_argument('--config', type=str, default=None, help='config file')
        parser.add_argument('--weight_decay', type=float, default=0.00001)
        parser.add_argument('--base_lr', type=float, default=0.0001)
        parser.add_argument('--fold', type=int, default=None, help='cross-validation folds')
        parser.add_argument('--bs_train', type=int, default=None, help='batchsize of train')
        parser.add_argument('--bs_test', type=int, default=None, help='batchsize of test')
        parser.add_argument('--num_groups', type=int, default=4, help='number of group size')

        self.args = parser.parse_args()

    def setup(self):
        if self.args.config is None:
            self.args.config = f'configs/{self.args.benchmark}.yaml'
        else:
            self.args.config = f'configs/{self.args.config}.yaml'

        self.get_config()

        self.merge_config()

    def get_config(self):
        print(f'----------------------------\n'
              f'Load yaml from {self.args.config}.\n'
              f'----------------------------\n')

        with open(self.args.config) as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

    def merge_config(self):
        for k, v in self.config.items():
            if k not in vars(self.args).keys():
                setattr(self.args, k, v)
            elif vars(self.args)[k] == None:
                setattr(self.args, k, v)


    def check_args(self):
        self.args.exp_path = os.path.join('./exps', self.args.benchmark, self.args.exp_name)
        if self.args.benchmark == 'Seven':
            print(f'Using CLASS idx {self.args.class_idx}')
            self.args.exp_path = os.path.join(self.args.exp_path, str(self.args.class_idx))

        if self.args.benchmark == 'JIGSAWS':
            print(f'Using CLASS idx {self.args.class_idx}')
            self.args.exp_path = os.path.join(self.args.exp_path, str(self.args.class_idx)
                                              + '_' + str(self.args.fold))

        if self.args.benchmark == 'MTL':
            if not self.args.usingDD:
                self.args.score_range = 100

        if torch.cuda.is_available():
            self.args.use_gpu = True
        else:
            self.args.use_gpu = False

        self.args.device = [int(i) for i in self.args.device]
        self.args.output_device = self.args.device[0]

        misc.init_seed(self.args.seed)

        if self.args.resume:
            cfg_path = os.path.join(self.args.exp_path, f'configs/{self.args.benchmark}.yaml')
            print(f'----------------------------\n'
                  f'Resume yaml from {cfg_path}.\n'
                  f'----------------------------\n')


        self.args.model_args['num_groups'] = self.args.num_groups
