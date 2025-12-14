import sys
import argparse
import yaml
import os
import shutil
import numpy as np
import random
import math

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F  # [ADDED] Needed for cosine similarity
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .io import IO
from tensorboardX import SummaryWriter

def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Processor(IO):
    """
        Base Processor
    """

    def __init__(self, argv=None):

        self.load_arg(argv)
        self.init_environment()

        if self.arg.phase == 'train':
            if os.path.isdir(self.arg.work_dir + '/train'):
                print('log_dir: ', self.arg.work_dir, 'already exist')
                # 只有当确实存在且不是关键目录时才删除，这里保留原有逻辑
                shutil.rmtree(self.arg.work_dir + '/train')
                shutil.rmtree(self.arg.work_dir + '/val')
                print('Dir removed: ', self.arg.work_dir + '/train')
                print('Dir removed: ', self.arg.work_dir + '/val')
            self.train_writer = SummaryWriter(os.path.join(self.arg.work_dir, 'train'), 'train')
            self.val_writer = SummaryWriter(os.path.join(self.arg.work_dir, 'val'), 'val')
        else:
            self.train_writer = self.val_writer = SummaryWriter(os.path.join(self.arg.work_dir, 'test'), 'test')

        self.load_model()
        self.load_weights()
        self.gpu()
        self.load_data()
        self.load_optimizer()

        self.global_step = 0

    def train_log_writer(self, epoch):
        self.train_writer.add_scalar('batch_loss', self.iter_info['loss'], self.global_step)
        self.train_writer.add_scalar('lr', self.lr, self.global_step)
        self.train_writer.add_scalar('epoch', epoch, self.global_step)

    def eval_log_writer(self, epoch):
        self.val_writer.add_scalar('eval_loss', self.eval_info['eval_mean_loss'], epoch)
        self.val_writer.add_scalar('current_result', self.current_result, epoch)
        self.val_writer.add_scalar('best_result', self.best_result, epoch)

    def init_environment(self):
        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.eval_info = dict()
        self.meta_info = dict(epoch=0, iter=0)

    def load_optimizer(self):
        pass

    def load_data(self):
        self.data_loader = dict()

        # 1. 加载训练集
        if self.arg.train_feeder_args:
            train_feeder = import_class(self.arg.train_feeder)

            # 计算 num_workers
            num_workers = self.arg.num_worker
            if self.arg.use_gpu and len(self.gpus) > 1:
                num_workers = self.arg.num_worker * len(self.gpus)

            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=train_feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=num_workers,
                drop_last=True,
                worker_init_fn=init_seed)

        # [FIX] 从训练集获取统计量 (mean_map, std_map)
        train_mean_map = None
        train_std_map = None
        if 'train' in self.data_loader:
            if hasattr(self.data_loader['train'].dataset, 'mean_map'):
                train_mean_map = self.data_loader['train'].dataset.mean_map
                train_std_map = self.data_loader['train'].dataset.std_map
                # print(f"Train stats loaded. Mean shape: {train_mean_map.shape}")

        # 2. 加载测试集
        if self.arg.test_feeder_args:
            test_feeder = import_class(self.arg.test_feeder)

            num_workers = self.arg.num_worker
            if self.arg.use_gpu and len(self.gpus) > 1:
                num_workers = self.arg.num_worker * len(self.gpus)

            # [FIX] 将训练集的统计量注入到测试集参数中
            # 这样测试集就会使用训练集的分布进行归一化，而不是使用测试集自己的
            test_args = self.arg.test_feeder_args.copy()  # 复制配置，避免修改原始参数
            if train_mean_map is not None:
                test_args['mean_map'] = train_mean_map
                test_args['std_map'] = train_std_map
                # print("Injected training stats into test feeder.")

            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=test_feeder(**test_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=num_workers,
                drop_last=False,
                worker_init_fn=init_seed)

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_eval_info(self):
        for k, v in self.eval_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('eval', self.meta_info['iter'], self.eval_info)

    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.arg.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def train(self, epoch):
        """
        [MODIFIED for Scheme B]
        Updated training loop with Joint Regression Loss and Bone Direction Loss.
        Supports both Bone Stream (Bone-to-Bone) and Joint Stream (Joint-to-Joint) alignment.
        """
        self.model.train()
        self.adjust_lr()
        self.io.print_log('Training epoch: {}'.format(epoch))
        loader = self.data_loader['train']
        loss_value = []

        # 获取真实的 Model 对象（处理 DataParallel 的情况）以便访问 graph 属性
        real_model = self.model.module if hasattr(self.model, 'module') else self.model

        # [Safety Fallback] SHREC 22 关节的标准骨骼连接 (以防 graph 属性缺失)
        # 格式: (Parent, Child)
        SHREC_BONE_PAIRS = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),  # Thumb
            (1, 6), (6, 7), (7, 8), (8, 9),  # Index
            (1, 10), (10, 11), (11, 12), (12, 13),  # Middle
            (1, 14), (14, 15), (15, 16), (16, 17),  # Ring
            (1, 18), (18, 19), (19, 20), (20, 21)  # Pinky
        ]

        # [MODIFIED] Unpack 4 items: data, label, index, gt_joints
        for batch_idx, (data, label, index, gt_joints_raw) in enumerate(loader):
            self.global_step += 1

            # 1. 移动数据到 GPU
            with torch.no_grad():
                data = data.float().to(self.dev)
                label = label.long().to(self.dev)
                gt_joints_raw = gt_joints_raw.float().to(self.dev)

            # 2. 前向传播
            # 方案B: 训练模式下 Model.forward 返回 (logits, predicted_joints)
            output, predicted_joints = self.model(data)

            # 3. 准备 Ground Truth
            # Input data shape: (N, C, T, V, M)
            # GT Raw from Feeder: (N, 3, T, V, M) -> 转换以匹配模型输出 (N*M, 3, T, V)
            N, C_gt, T, V, M = gt_joints_raw.shape
            gt_joints = gt_joints_raw.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C_gt, T, V)

            # 4. 计算 Loss

            # 4.1 分类损失 (Classification Loss)
            if hasattr(self, 'loss') and self.loss is not None:
                L_cls = self.loss(output, label)
            else:
                L_cls = nn.CrossEntropyLoss()(output, label)

            # 4.2 辅助任务损失 (Regression & Bone Direction)
            L_reg = torch.tensor(0.0, device=self.dev)
            L_bone_cos = torch.tensor(0.0, device=self.dev)

            # 智能获取骨骼连接对
            bone_pairs = []
            if hasattr(real_model.graph, 'source_edges'):
                # Case 1: HGT-Bone (Line Graph) 通常有此属性
                bone_pairs = real_model.graph.source_edges
            elif hasattr(real_model.graph, 'inward_ori_index'):
                # Case 2: Standard Graph (Joint)
                bone_pairs = real_model.graph.inward_ori_index
            else:
                # Case 3: Fallback (防止 Crash)
                bone_pairs = SHREC_BONE_PAIRS

            # 根据数据流类型 (Bone vs Joint) 采用不同的对齐策略
            if self.arg.stream == 'bone':
                # === Bone Stream: Bone-to-Bone Alignment ===
                # 目标：构建 GT Bone Vectors (因为 Bone 流模型预测的是向量)
                gt_bone_list = []
                # 过滤越界对
                valid_pairs = [p for p in bone_pairs if p[0] < V and p[1] < V]

                for v1, v2 in valid_pairs:
                    vec = gt_joints[..., v1] - gt_joints[..., v2]
                    gt_bone_list.append(vec)

                if gt_bone_list:
                    target_data = torch.stack(gt_bone_list, dim=-1)  # (N*M, 3, T, Num_Bones)
                    pred_data = predicted_joints  # 模型输出即为骨骼向量估计

                    # 维度对齐 (时间 T)
                    if pred_data.shape[2] != target_data.shape[2]:
                        pred_data = F.interpolate(
                            pred_data,
                            size=(target_data.size(2), target_data.size(3)),
                            mode='bilinear', align_corners=False
                        )

                    # L_reg: 向量数值回归
                    L_reg = nn.MSELoss()(pred_data, target_data)

                    # L_bone_cos: 向量方向一致性
                    # pred_data 和 target_data 都是向量，直接算 Cosine
                    # eps=1e-6 防止除零
                    cos_sim = F.cosine_similarity(pred_data, target_data, dim=1, eps=1e-6)
                    L_bone_cos = 1.0 - cos_sim.mean()
            else:
                # === Joint Stream: Joint-to-Joint Alignment ===
                target_data = gt_joints
                pred_data = predicted_joints

                # 维度对齐 (时间 T 和 空间 V)
                if pred_data.shape[2:] != target_data.shape[2:]:
                    pred_data = F.interpolate(
                        pred_data,
                        size=(target_data.size(2), target_data.size(3)),
                        mode='bilinear', align_corners=False
                    )

                # L_reg: 绝对坐标回归
                L_reg = nn.MSELoss()(pred_data, target_data)

                # L_bone_cos: 先计算向量，再算方向一致性
                if len(bone_pairs) > 0:
                    src_idx = [p[0] for p in bone_pairs if p[0] < V and p[1] < V]
                    dst_idx = [p[1] for p in bone_pairs if p[0] < V and p[1] < V]

                    if len(src_idx) > 0:
                        pred_vec = pred_data[..., src_idx] - pred_data[..., dst_idx]
                        gt_vec = target_data[..., src_idx] - target_data[..., dst_idx]

                        cos_sim = F.cosine_similarity(pred_vec, gt_vec, dim=1, eps=1e-6)
                        L_bone_cos = 1.0 - cos_sim.mean()

            # 4.3 获取超图正则化 Loss
            L_entropy = torch.tensor(0.0, device=self.dev)
            L_ortho = torch.tensor(0.0, device=self.dev)

            if hasattr(real_model, 'get_hypergraph_loss'):
                # 兼容返回 (ent, ortho) 或 (ent, ortho, phy)
                hg_losses = real_model.get_hypergraph_loss()
                if len(hg_losses) >= 2:
                    L_entropy = hg_losses[0]
                    L_ortho = hg_losses[1]

            # 4.4 总 Loss 加权
            # 优先从 args 读取，如果没有则使用默认值
            lambda_reg = getattr(self.arg, 'lambda_reg', 0.1)
            # [Critical] 使用 lambda_physical 控制 L_bone_cos
            lambda_bone = getattr(self.arg, 'lambda_physical', 0.05)
            lambda_ent = getattr(self.arg, 'lambda_entropy', 0.001)
            lambda_ort = getattr(self.arg, 'lambda_ortho', 0.1)

            loss = L_cls + \
                   (lambda_reg * L_reg) + \
                   (lambda_bone * L_bone_cos) + \
                   (lambda_ent * L_entropy) + \
                   (lambda_ort * L_ortho)

            # 5. 反向传播
            if hasattr(self, 'optimizer'):
                self.optimizer.zero_grad()
                loss.backward()
                # [Recommended] 梯度裁剪
                if self.arg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_clip_norm)
                self.optimizer.step()

            # 6. 记录日志
            self.iter_info['loss'] = loss.item()
            self.iter_info['L_cls'] = L_cls.item()
            self.iter_info['L_reg'] = L_reg.item()
            self.iter_info['L_bone'] = L_bone_cos.item()
            self.iter_info['L_ent'] = L_entropy.item()

            if hasattr(self, 'optimizer'):
                self.iter_info['lr'] = self.optimizer.param_groups[0]['lr']

            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()

    def test(self, epoch):
        self.model.eval()
        self.io.print_log('Eval epoch: {}'.format(epoch))
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        # [MODIFIED] Handle potentially 4 return values from loader, though test loader might behave differently depending on Feeder implementation
        # Assuming Feeder returns same structure for test set
        for batch_idx, batch_data in enumerate(loader):
            # Unpack dynamically to handle different Feeder returns (3 or 4 values)
            data = batch_data[0].float().to(self.dev)
            label = batch_data[1].long().to(self.dev)
            # index = batch_data[2]
            # gt_joints = batch_data[3] if len(batch_data) > 3 else None

            with torch.no_grad():
                # Inference Mode: Model returns only logits (controlled by model.training flag)
                output = self.model(data)

                # Safety check: if model accidentally returns tuple in eval mode
                if isinstance(output, tuple):
                    output = output[0]

                if hasattr(self, 'loss') and self.loss is not None:
                    loss = self.loss(output, label)
                else:
                    loss = nn.CrossEntropyLoss()(output, label)

                loss_value.append(loss.item())
                result_frag.append(output.data.cpu().numpy())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        labels = np.concatenate(label_frag)

        self.eval_info['eval_mean_loss'] = np.mean(loss_value)
        self.show_eval_info()

        # 计算 Top-1 和 Top-5
        rank = self.result.argsort()
        hit1_list = [l in rank[i, -1:] for i, l in enumerate(labels)]
        hit5_list = [l in rank[i, -5:] for i, l in enumerate(labels)]

        self.current_result = np.sum(hit1_list) / len(hit1_list) * 100.0
        self.io.print_log('\tTop1: {:.2f}%'.format(self.current_result))
        self.io.print_log('\tTop5: {:.2f}%'.format(np.sum(hit5_list) / len(hit5_list) * 100.0))

    def print_networks(self, net, print_flag=False):
        self.io.print_log('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        if print_flag:
            self.io.print_log(net)
        self.io.print_log('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        self.io.print_log('-----------------------------------------------')

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        self.print_networks(self.model)

        # training phase
        if self.arg.phase == 'train':
            self.global_step = self.arg.start_epoch * len(self.data_loader['train'])
            self.meta_info['iter'] = self.global_step
            self.best_result = 0.0

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch + 1

                # training
                self.train(epoch + 1)

                # save model
                if self.arg.save_interval == -1:
                    pass
                elif ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    filename = 'epoch{}_model.pt'.format(epoch + 1)
                    self.io.save_model(self.model, filename)

                # evaluation
                if self.arg.eval_interval == -1:
                    pass
                elif ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    self.test(epoch + 1)
                    self.io.print_log("current %.2f%%, best %.2f%%" %
                                      (self.current_result, self.best_result))

                    # save best model
                    filename = 'epoch%.3d_acc%.2f_model.pt' % (epoch + 1, self.current_result)
                    self.io.save_model(self.model, filename)
                    if self.current_result >= self.best_result:
                        filename = 'best_model.pt'
                        self.io.save_model(self.model, filename)
                        # save the output of model
                        if self.arg.save_result:
                            result_dict = dict(
                                zip(self.data_loader['test'].dataset.sample_name,
                                    self.result))
                            self.io.save_pkl(result_dict, 'test_result.pkl')

        # test phase
        elif self.arg.phase == 'test':

            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))

            self.best_result = 0.0

            # evaluation
            self.io.print_log('Evaluation Start:')
            self.test(1)
            self.io.print_log('Done.\n')

            # save the output of model
            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name,
                        self.result))
                # ✅ 直接保存为固定文件名
                self.io.save_pkl(result_dict, 'test_result.pkl')

    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(add_help=add_help, description='Base Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

        # processor
        parser.add_argument('--phase', default='train', help='must be train or test')
        parser.add_argument('--save_result', type=str2bool, default=True,
                            help='if ture, the output of the model will be stored')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+',
                            help='the indexes of GPUs for training or testing')

        # visualize and debug
        parser.add_argument('--log_interval', type=int, default=100,
                            help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=10, help='the interval for storing models (#epoch)')
        parser.add_argument('--eval_interval', type=int, default=5, help='the interval for evaluating models (#epoch)')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--pavi_log', type=str2bool, default=False, help='logging on pavi or not')

        # feeder
        parser.add_argument('--train_feeder', default='feeder.feeder', help='train data loader will be used')
        parser.add_argument('--test_feeder', default='feeder.feeder', help='test data loader will be used')
        parser.add_argument('--num_worker', type=int, default=4, help='the number of worker per gpu for data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(),
                            help='the arguments of data loader for training')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(),
                            help='the arguments of data loader for test')
        parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
        parser.add_argument('--debug', action="store_true", help='less data, faster loading')

        # model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--weights', default=None, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+',
                            help='the name of weights which will be ignored in the initialization')

        # loss (ADDED)
        parser.add_argument('--loss', default=None, help='the loss will be used')
        parser.add_argument('--loss_args', action=DictAction, default=dict(), help='the arguments of loss')

        # optimizer (ADDED - 必须保留)
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--optimizer_args', action=DictAction, default=dict(), help='the arguments of optimizer')

        return parser