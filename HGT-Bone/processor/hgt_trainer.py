import sys
import argparse
import yaml
import math
import numpy as np
import os

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor


def weights_init(m):
    """
    Xavier/Kaiming Initialization for GCN
    """
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class FT_Processor(Processor):
    """
    Processor for HGT-Bone (Paper 1)

    Academic Goal:
    Evaluate the effectiveness of High-Order Geometric Features and
    Physically-Guided Hypergraph Convolution.

    Mechanism:
    - Supervised Training (CrossEntropy)
    - Physical Regularization (Orthogonality Loss)
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
                eps=1e-4)
        else:
            raise ValueError()

    def adjust_lr(self):
        lr_decay_rate = getattr(self.arg, 'lr_decay_rate', 0.1)

        # Warm-up Strategy
        if hasattr(self.arg, 'warm_up_epoch') and self.arg.warm_up_epoch > 0:
            if self.meta_info['epoch'] < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (self.meta_info['epoch'] + 1) / self.arg.warm_up_epoch
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.lr = lr
                return

        # Multi-step Decay
        if self.arg.step:
            lr = self.arg.base_lr * (
                    lr_decay_rate ** np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def show_best(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100 * sum(hit_top_k) * 1.0 / len(hit_top_k)
        accuracy = round(accuracy, 5)
        self.current_result = accuracy
        if self.best_result <= accuracy:
            self.best_result = accuracy
            # Optional: Save best model weights here

        self.io.print_log('\tBest Top{}: {:.2f}%'.format(k, self.best_result))

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()

        loader = self.data_loader['train']
        loss_value = []
        phy_loss_value = []

        for data, label in loader:
            self.global_step += 1
            # data: (N, 8, T, V=21, M) already from feeder_hgt
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            # [HGT-Bone Logic]
            # Feeder handles feature extraction. No extra manual diff needed here.

            output = self.model(data)

            # 1. Classification Loss (Cross Entropy)
            loss_ce = self.loss(output, label)

            # 2. Physical Constraint Loss (Orthogonality)
            loss_phy = torch.tensor(0.0).to(self.dev)

            # Handle DataParallel
            model_core = self.model.module if hasattr(self.model, 'module') else self.model

            # Retrieve loss from HGTBoneNet
            if hasattr(model_core, 'get_orthogonality_loss'):
                loss_phy = model_core.get_orthogonality_loss()

            # Total Loss
            # loss = L_ce + lambda * L_phy
            loss = loss_ce + (self.arg.lambda_physical * loss_phy)

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            if self.arg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_clip_norm)

            self.optimizer.step()

            # Statistics & Logging
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['loss_ce'] = loss_ce.data.item()
            self.iter_info['loss_phy'] = loss_phy.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)

            loss_value.append(self.iter_info['loss'])
            phy_loss_value.append(self.iter_info['loss_phy'])

            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.epoch_info['mean_phy'] = np.mean(phy_loss_value)

        self.show_epoch_info()
        self.train_writer.add_scalar('loss', self.epoch_info['mean_loss'], epoch)
        self.train_writer.add_scalar('loss_physical', self.epoch_info['mean_phy'], epoch)

    def test(self, epoch):
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            with torch.no_grad():
                output = self.model(data)

            result_frag.append(output.data.cpu().numpy())

            if self.arg.phase in ['train', 'test']:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if self.arg.phase in ['train', 'test']:
            self.label = np.concatenate(label_frag)

        self.eval_info['eval_mean_loss'] = np.mean(loss_value)
        self.show_eval_info()

        for k in self.arg.show_topk:
            self.show_topk(k)

        self.show_best(1)
        self.eval_log_writer(epoch)

    @staticmethod
    def get_parser(add_help=False):
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='HGT-Bone Recognition Processor')

        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--stream', type=str, default='bone', help='the stream of input')

        # [HGT-Bone Specific Arguments]
        parser.add_argument('--lambda_physical', type=float, default=0.1,
                            help='Weight for Physical Constraint (Orthogonality) loss')

        parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Gradient clipping norm')
        parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
        parser.add_argument('--warm_up_epoch', type=int, default=0, help='warm up epochs')

        return parser