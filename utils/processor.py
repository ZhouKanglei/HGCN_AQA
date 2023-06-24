# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/3/27 10:51
import logging
import numpy as np
import os
import pprint
import shutil
from collections import OrderedDict

import torch
import yaml
from scipy import stats
from torch import nn, optim
from torchvideotransforms import video_transforms, volume_transforms
from tqdm import tqdm

from utils import misc
from utils.misc import count_params


class Processor(object):
    """Processor for AQA"""

    def __init__(self, args):
        self.args = args

        # initialize the log
        self.init_log()

        # save the files
        self.save_files()

        # load data
        self.load_data()

        # load model
        self.load_model()

        # load loss
        self.load_loss()

        # build optimizer & scheduler
        self.build_opti_sche()

        # load pre-train
        self.load_pretrain()

    def init_log(self):
        # logger: CRITICAL > ERROR > WARNING > INFO > DEBUG
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # stream handler
        log_sh = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(message)s', "%Y-%m-%d %H:%M:%S")
        log_sh.setFormatter(formatter)

        logger.addHandler(log_sh)

        # file handler
        log_dir = os.path.join(self.args.exp_path, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f'{self.args.phase}.log')
        if (self.args.resume and self.args.phase == 'train') or self.args.phase == 'test':
            log_fh = logging.FileHandler(log_file, mode='a')
        else:
            log_fh = logging.FileHandler(log_file, mode='w')

        log_fh.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s | %(message)s', "%Y-%m-%d %H:%M:%S")
        log_fh.setFormatter(formatter)

        logger.addHandler(log_fh)

        self.logger = logger

    def save_files(self):
        # save config file
        args_dict = vars(self.args)

        config_dir = os.path.join(self.args.exp_path, 'configs')
        os.makedirs(config_dir, exist_ok=True)

        config_file = os.path.join(config_dir, f'{self.args.benchmark}.yaml')
        with open(config_file, 'w') as f:
            yaml.dump(args_dict, f)

        self.logger.info(f'Save args to {config_file}.')

        # save model
        model_dir = os.path.join(self.args.exp_path, 'models')
        os.makedirs(model_dir, exist_ok=True)

        if self.args.phase == 'train':
            shutil.copytree('./models', model_dir, dirs_exist_ok=True)
            self.logger.info(f'Back-up copy models to {model_dir}.')

    def load_data(self):
        self.logger.info(f'Load data...')

        Dataset = misc.import_class("datasets." + self.args.benchmark)
        self.dataloader = {}

        train_trans = video_transforms.Compose([
            video_transforms.RandomHorizontalFlip(),
            video_transforms.Resize((455, 256)),
            video_transforms.RandomCrop(224),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = Dataset(self.args, transform=train_trans, subset='train')

        if self.args.phase == 'train':
            self.dataloader['train'] = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.args.bs_train,
                shuffle=True,
                pin_memory=True,
                num_workers=int(self.args.workers),
                worker_init_fn=misc.worker_init_fn
            )

        test_trans = video_transforms.Compose([
            video_transforms.Resize((455, 256)),
            video_transforms.CenterCrop(224),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_dataset = Dataset(self.args, transform=test_trans, subset='test')
        self.dataloader['test'] = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.args.bs_test,
            shuffle=False,
            pin_memory=True,
            num_workers=int(self.args.workers)
        )

    def load_model(self):
        self.logger.info(f'Load model...')

        Backbone = misc.import_class(self.args.backbone)
        self.base_model = Backbone(**self.args.backbone_args).cuda(
            self.args.output_device) if self.args.use_gpu else Backbone(**self.args.backbone_args)

        Regressor = misc.import_class(self.args.model)
        self.regressor = Regressor(**self.args.model_args).cuda(
            self.args.output_device) if self.args.use_gpu else Regressor(**self.args.model_args)

        if len(self.args.device) > 1:
            self.base_model = nn.DataParallel(self.base_model, device_ids=self.args.device,
                                              output_device=self.args.output_device)
            self.regressor = nn.DataParallel(self.regressor, device_ids=self.args.device,
                                             output_device=self.args.output_device)

            self.base_model = self.base_model.cuda(self.args.output_device)
            self.regressor = self.regressor.cuda(self.args.output_device)
            self.logger.info(f'{len(self.args.device)} GPUs available, using DataParallel.')

    def load_loss(self):
        self.mse = nn.MSELoss(reduction='sum').cuda(self.args.output_device)

    def compute_loss(self, epoch, label, pred, mu):
        loss = self.mse(label, pred)
        if self.args.super_loss and epoch % 10 == 0:
            loss = 0.4 * self.mse(label, mu) + 0.6 * loss

        return loss

    def compute_matric(self, pred_scores, true_scores):
        rho, p = stats.spearmanr(pred_scores, true_scores)

        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)

        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = 100 * np.power((pred_scores - true_scores) /
                       (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]

        return rho, p, L2, RL2

    def build_opti_sche(self):
        self.logger.info(f'Build optimizer and scheduler...')

        if self.args.optimizer == 'Adam':

            self.optimizer = optim.Adam(
                [{'params': self.base_model.parameters(), 'lr': self.args.base_lr * self.args.lr_factor},
                 {'params': self.regressor.parameters()}],
                lr=self.args.base_lr, weight_decay=self.args.weight_decay
            )

            self.optimizer = optim.Adam(
                [*self.base_model.parameters()] + [*self.regressor.parameters()],
                lr=self.args.base_lr, weight_decay=self.args.weight_decay
            )

        else:
            raise NotImplementedError()

        self.lr_scheduler = None

    def load_pretrain(self):
        self.history = {}
        self.history['train'] = []
        self.history['test'] = []

        if self.args.resume is not True and self.args.phase == 'train':
            # initial params
            self.start_epoch = 0
            self.epoch_best = 0
            self.rho_best = 0
            self.L2_min, self.RL2_min = 1000, 1000

            return

        if self.args.ckpts is None:
            self.args.ckpts = os.path.join(self.args.exp_path, 'weights/best.pth') if self.args.phase == 'test' \
                else os.path.join(self.args.exp_path, 'weights/last.pth')

        if not os.path.exists(self.args.ckpts):
            self.logger.info(f'No checkpoint file from path {self.args.ckpts}...')
            # initial params
            self.start_epoch = 0
            self.epoch_best = 0
            self.rho_best = 0
            self.L2_min, self.RL2_min = 1000, 1000

            return

        self.logger.info(f'Loading weights from {self.args.ckpts}...')

        # load state dict
        state_dict = torch.load(self.args.ckpts, map_location='cpu')

        # parameter resume of models
        self.base_model.load_state_dict(state_dict['base_model'])
        self.regressor.load_state_dict(state_dict['regressor'])

        # optimizer
        self.optimizer.load_state_dict(state_dict['optimizer'])

        # initial params
        self.start_epoch = state_dict['epoch'] + 1
        self.epoch_best = state_dict['epoch_best']
        self.rho_best = state_dict['rho_best']
        self.L2_min = state_dict['L2_min']
        self.RL2_min = state_dict['RL2_min']

    def save_checkpoint(self, epoch, rho, L2, RL2, exp_name):
        checkpoint_dir = os.path.join(self.args.exp_path, 'weights')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, exp_name + '.pth')

        torch.save({
            'base_model': self.base_model.state_dict(),
            'regressor': self.regressor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch, 'rho': rho, 'L2': L2, 'RL2': RL2,
            'epoch_best': self.epoch_best,
            'rho_best': self.rho_best,
            'L2_min': self.L2_min,
            'RL2_min': self.RL2_min,
        }, checkpoint_file)

        self.logger.info(f'Save checkpoint to {checkpoint_file}.')

    def save_best(self, epoch, rho, L2, RL2, pred_scores, true_scores):
        self.L2_min = L2
        self.RL2_min = RL2
        self.rho_best = rho
        self.epoch_best = epoch

        self.logger.info('----- New best found -----')
        self.print_best()

        self.save_checkpoint(epoch, rho, L2, RL2, 'best')

        save_path_pred = os.path.join(self.args.exp_path, 'weights/pred.npy')
        save_path_true = os.path.join(self.args.exp_path, 'weights/true.npy')
        np.save(save_path_pred, pred_scores)
        np.save(save_path_true, true_scores)

    def save_history(self):
        history_file = os.path.join(self.args.exp_path, 'weights/history.npz')
        np.savez(history_file, train=self.history['train'], test=self.history['test'])

        self.logger.info(f'Save history to {history_file}.')

    def print_best(self):
        self.logger.info(f' Best epoch: {self.epoch_best + 1:d}')
        self.logger.info(f'Correlation: {self.rho_best:.4f}')
        self.logger.info(f'         L2: {self.L2_min:.4f}')
        self.logger.info(f'        RL2: {self.RL2_min:.4f}')

    def train_step(self, epoch):
        self.base_model.eval()
        self.regressor.train()

        self.base_model.apply(misc.fix_bn)

        loader = self.dataloader['train']
        true_scores = []
        pred_scores = []

        process = tqdm(loader, dynamic_ncols=True)
        for idx, (data, index) in enumerate(process):
            # data
            true_scores.extend(data['final_score'].numpy().reshape((-1, 1)))
            video = data['video'].float().cuda(self.args.output_device)
            label = data['final_score'].float().reshape((-1, 1)).cuda(self.args.output_device)

            # forward
            feature = self.base_model(video)
            pred, mu, _ = self.regressor(feature, self.args.phase)

            # loss
            if self.args.usingDD:
                diff = data['difficulty'].float().reshape((-1, 1)).cuda(self.args.output_device)
                pred = pred * diff
                mu = mu * diff

            loss = self.compute_loss(epoch, label, pred, mu)

            # pred
            pred_scores.extend([i for i in pred.cpu().detach().numpy()])

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # process bar updating
            process.set_description(f'(BS {self.args.bs_train}) loss: {loss:.2f}')
            process.update()

        process.close()

        # analysis on results
        rho, p, L2, RL2 = self.compute_matric(pred_scores, true_scores)
        self.history['train'].append((epoch, rho, p, L2, RL2))
        self.logger.info(f'[TRAIN {epoch + 1:d}] Correlation: {rho:.4f}, '
                         f'L2: {L2:.4f}, RL2: {RL2:.4f}')

        # scheduler lr
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def eval_step(self, epoch):
        self.base_model.eval()
        self.regressor.eval()

        loader = self.dataloader['test']
        true_scores = []
        pred_scores = []

        process = tqdm(range(len(loader)), dynamic_ncols=True)
        with torch.no_grad():
            for idx, (data, index) in enumerate(loader):
                # data
                true_scores.extend(data['final_score'].numpy().reshape((-1, 1)))
                video = data['video'].float().cuda(self.args.output_device)

                # forward
                feature = self.base_model(video)
                pred, mu, std = self.regressor(feature, self.args.phase)

                # loss
                if self.args.usingDD:
                    diff = data['difficulty'].float().reshape((-1, 1)).cuda(self.args.output_device)
                    pred = pred * diff

                # pred
                pred_scores.extend([i for i in pred.cpu().detach().numpy()])

                # process bar updating
                process.set_description(f'(BS {self.args.bs_test})')
                process.update()

            process.close()

            # analysis on results
            rho, p, L2, RL2 = self.compute_matric(pred_scores, true_scores)
            self.history['test'].append((epoch, rho, p, L2, RL2))
            self.logger.info(f' [EVAL {epoch + 1:d}] Correlation: {rho:.4f} ({self.rho_best:.4f}), '
                             f'L2: {L2:.4f}, RL2: {RL2:.4f}')

            # save checkpoint
            if self.args.phase == 'train':
                self.save_checkpoint(epoch, rho, L2, RL2, 'last')

            # save best
            if rho > self.rho_best and self.args.phase == 'train':
                self.save_best(epoch, rho, L2, RL2, pred_scores, true_scores)

    def test_step(self, epoch):
        self.base_model.eval()
        self.regressor.eval()

        loader = self.dataloader['test']
        true_scores = []
        pred_scores = []
        I3D_features, shots, scenes, vs = [], [], [], []
        diffs = []

        process = tqdm(range(len(loader)), dynamic_ncols=True)
        with torch.no_grad():
            for idx, (data, index) in enumerate(loader):
                # data
                true_scores.extend(data['final_score'].numpy().reshape((-1, 1)))
                video = data['video'].float().cuda(self.args.output_device)

                # forward
                feature = self.base_model(video)
                I3D_features.extend(feature.cpu().detach().numpy())
                pred, mu, std, shot, scene, v, scores = self.regressor(feature, self.args.phase)
                shots.extend(shot.cpu().detach().numpy())
                scenes.extend(scene.cpu().detach().numpy())
                vs.extend(v.cpu().detach().numpy())

                # loss
                if self.args.usingDD:
                    diff = data['difficulty'].float().reshape((-1, 1)).cuda(self.args.output_device)
                    pred = pred * diff
                    diffs.extend(data['difficulty'].float().reshape((-1, )))

                # pred
                pred_scores.extend([i for i in pred.cpu().detach().numpy()])

                # process bar updating
                process.set_description(f'(BS {self.args.bs_test})')
                process.update()

            process.close()

            # save feature
            fiile_name = './exps/mix/features.npz'
            os.makedirs(os.path.dirname(fiile_name), exist_ok=True)
            np.savez(fiile_name, I3D=np.array(I3D_features), y=np.array(true_scores), shot=np.array(shots),
                     scene=np.array(scenes), v=np.array(vs), diff=np.array(diffs))

            # analysis on results
            rho, p, L2, RL2 = self.compute_matric(pred_scores, true_scores)
            self.history['test'].append((epoch, rho, p, L2, RL2))
            self.logger.info(f' [EVAL {epoch + 1:d}] Correlation: {rho:.4f} ({self.rho_best:.4f}), '
                             f'L2: {L2:.4f}, RL2: {RL2:.4f}')

            # save checkpoint
            if self.args.phase == 'train':
                self.save_checkpoint(epoch, rho, L2, RL2, 'last')

            # save best
            if rho > self.rho_best and self.args.phase == 'train':
                self.save_best(epoch, rho, L2, RL2, pred_scores, true_scores)


    def start(self):
        if self.args.resume == False and self.args.phase == 'train':
            self.logger.info(f'Parameters:\n{pprint.pformat(vars(self.args))}\n')

        num_param1, num_param2 = count_params(self.base_model), count_params(self.regressor)
        self.logger.info(f'Model total number of params: {num_param1:,d} + {num_param2:,d}')

        if self.args.phase == 'train':
            for epoch in range(self.start_epoch, self.args.max_epoch):
                self.logger.info(f'+----------------------------'
                                 f'[EPOCH {epoch + 1}]'
                                 f'----------------------------+')
                self.train_step(epoch)
                self.eval_step(epoch)

            self.logger.info(f'------------------------------------')
            self.print_best()
            self.save_history()
            self.logger.info(f'------------------------------------')

        else:
            self.test_step(0)
