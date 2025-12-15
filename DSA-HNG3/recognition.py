import sys
import argparse
import yaml
import math
import numpy as np
import os

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor


def weights_init(m):
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
    Processor for Paper A Supervised Training
    Updated: Uses Entropy Regularization instead of Hard Sparsity.
    Supports Physically Guided DSA-HGN (Scheme B).
    ✅ [FIXED] Added temperature annealing support.
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

        if hasattr(self.arg, 'warm_up_epoch') and self.arg.warm_up_epoch > 0:
            if self.meta_info['epoch'] < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (self.meta_info['epoch'] + 1) / self.arg.warm_up_epoch
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.lr = lr
                return

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
            # Export topology when achieving best result
            self.export_topology(self.meta_info['epoch'])

        self.io.print_log('\tBest Top{}: {:.2f}%'.format(k, self.best_result))

    def export_topology(self, epoch):
        """
        Exports the learned virtual topology (Adjacency Matrix) for visualization.
        """
        save_path = os.path.join(self.arg.work_dir, f'topology_best_epoch_{epoch}.npy')
        model_core = self.model.module if hasattr(self.model, 'module') else self.model
        target_module = None

        # Search for the first module with 'last_h_dynamic' (New Module) or 'last_h' (Old Module)
        for name, m in model_core.named_modules():
            # Check for new PhysicallyGuidedDSAHypergraph
            if hasattr(m, 'last_h_dynamic') and m.last_h_dynamic is not None:
                target_module = m
                H = target_module.last_h_dynamic[0:1]  # Use dynamic part for visualization
                break
            # Fallback to old DifferentiableSparseHypergraph
            elif hasattr(m, 'last_h') and m.last_h is not None:
                target_module = m
                H = target_module.last_h[0:1]
                break

        if target_module is not None:
            # H shape: (1, V, M)
            # Calculate Virtual Adjacency A = H * H.T (approx)
            # Shape: (1, V, V)
            A_virtual = torch.matmul(H, H.transpose(-1, -2)).squeeze(0).detach().cpu().numpy()

            np.save(save_path, A_virtual)

    # ============================================================
    # ✅ [CRITICAL FIX] Temperature Annealing Support
    # ============================================================

    def _update_hypergraph_temperature(self, epoch):
        """
        ✅ [核心修复] 递归更新所有超图模块的温度
        这是训练稳定性的关键 - 原代码遗漏了这个调用！

        Args:
            epoch: 当前训练轮次 (0-based)
        """
        real_model = self.model.module if hasattr(self.model, 'module') else self.model

        updated_count = 0
        for module in real_model.modules():
            # 更新所有具有 set_epoch 方法的模块（主要是超图）
            if hasattr(module, 'set_epoch') and callable(module.set_epoch):
                try:
                    module.set_epoch(epoch)
                    updated_count += 1
                except Exception as e:
                    self.io.print_log(f'[WARNING] Failed to update temperature for module: {e}')

        # 每10轮打印一次温度信息，用于监控退火过程
        if epoch % 10 == 0 and updated_count > 0:
            for m in real_model.modules():
                if hasattr(m, 'temperature') and hasattr(m, 'min_temperature'):
                    temp = m.temperature.item() if torch.is_tensor(m.temperature) else m.temperature
                    min_temp = m.min_temperature.item() if torch.is_tensor(m.min_temperature) else m.min_temperature
                    self.io.print_log(
                        f'[Temperature Annealing] Epoch {epoch + 1}: T={temp:.4f} '
                        f'(Target={min_temp:.2f}, Modules={updated_count})'
                    )
                    break

    def train(self, epoch):
        """
        ✅ [MODIFIED] 完整修复的训练循环
        主要改进：
        1. 添加温度退火调用
        2. 优化损失计算逻辑
        3. 添加数值稳定性保护
        """
        self.model.train()
        self.adjust_lr()

        # ✅ [核心修复1] 更新超图温度 (epoch从1开始，传入0-based)
        self._update_hypergraph_temperature(epoch - 1)

        # 安全检查
        is_multi_stream_model = 'DualBranch' in self.arg.model or 'MultiStream' in self.arg.model
        if is_multi_stream_model:
            if self.arg.stream != 'joint' or self.arg.train_feeder_args.get('bone', False):
                self.io.print_log(
                    f'[WARNING] Detected {self.arg.model}. Forcing stream="joint".')
                self.arg.stream = 'joint'
                self.arg.train_feeder_args['bone'] = False
                self.arg.train_feeder_args['vel'] = False

        loader = self.data_loader['train']
        loss_dict = {k: [] for k in ['loss', 'loss_ce', 'loss_ent', 'loss_orth',
                                     'loss_phy', 'loss_reg', 'loss_bone']}

        real_model = self.model.module if hasattr(self.model, 'module') else self.model

        # Fallback骨骼连接
        SHREC_BONE_PAIRS = [
            (0, 1),
            (1, 2), (2, 3), (3, 4), (4, 5),
            (1, 6), (6, 7), (7, 8), (8, 9),
            (1, 10), (10, 11), (11, 12), (12, 13),
            (1, 14), (14, 15), (15, 16), (16, 17),
            (1, 18), (18, 19), (19, 20), (20, 21)
        ]

        for data, label, index, gt_bone_raw in loader:
            self.global_step += 1

            # 移动到设备
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            gt_bone_raw = gt_bone_raw.float().to(self.dev, non_blocking=True)

            # gt_bone_raw已经是Bone向量格式 (N, 3, T, 21, M)
            N, C_gt, T, V_gt, M = gt_bone_raw.shape
            gt_bone = gt_bone_raw.permute(0, 4, 1, 2, 3).contiguous().view(
                N * M, C_gt, T, V_gt
            )  # (N*M, 3, T, 21)

            # 前向传播
            output, predicted_bone = self.model(data)

            # 分类损失
            loss_ce = self.loss(output, label)

            # ============================================================
            # Bone流回归损失
            # ============================================================
            if self.arg.stream == 'bone':
                target_data = gt_bone  # (N*M, 3, T, 21)
                pred_data = predicted_bone  # (N*M, 3, T', 21)

                # 时序维度对齐
                if pred_data.shape[2] != target_data.shape[2]:
                    pred_data = F.interpolate(
                        pred_data,
                        size=(target_data.size(2), target_data.size(3)),
                        mode='bilinear', align_corners=False
                    )

                # 回归损失
                loss_reg = nn.MSELoss()(pred_data, target_data)

                # ✅ [核心修复2] 方向一致性损失 - 添加数值稳定性保护
                pred_norm = F.normalize(pred_data, p=2, dim=1, eps=1e-8)
                target_norm = F.normalize(target_data, p=2, dim=1, eps=1e-8)
                cos_sim = (pred_norm * target_norm).sum(dim=1).mean()
                cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # 防止数值误差
                loss_bone_cos = 1.0 - cos_sim

            else:
                # Joint流(保留原逻辑)
                target_data = gt_bone
                pred_data = predicted_bone

                if pred_data.shape[2:] != target_data.shape[2:]:
                    pred_data = F.interpolate(
                        pred_data,
                        size=(target_data.size(2), target_data.size(3)),
                        mode='bilinear', align_corners=False
                    )

                loss_reg = nn.MSELoss()(pred_data, target_data)

                joint_pairs = SHREC_BONE_PAIRS
                if hasattr(real_model.graph, 'inward_ori_index'):
                    joint_pairs = real_model.graph.inward_ori_index
                elif hasattr(real_model.graph, 'source_edges'):
                    joint_pairs = real_model.graph.source_edges

                valid_pairs = [p for p in joint_pairs if p[0] < V_gt and p[1] < V_gt]
                if len(valid_pairs) > 0:
                    src_idx = [p[0] for p in valid_pairs]
                    dst_idx = [p[1] for p in valid_pairs]

                    pred_vec = pred_data[..., src_idx] - pred_data[..., dst_idx]
                    gt_vec = target_data[..., src_idx] - target_data[..., dst_idx]

                    cos_sim = F.cosine_similarity(pred_vec, gt_vec, dim=1, eps=1e-6)
                    loss_bone_cos = 1.0 - cos_sim.mean()
                else:
                    loss_bone_cos = torch.tensor(0.0).to(self.dev)

            # ============================================================
            # 超图正则化损失
            # ============================================================
            loss_entropy = torch.tensor(0.0).to(self.dev)
            loss_ortho = torch.tensor(0.0).to(self.dev)
            loss_physical = torch.tensor(0.0).to(self.dev)

            if hasattr(real_model, 'get_hypergraph_loss'):
                hg_losses = real_model.get_hypergraph_loss()
                if len(hg_losses) == 3:
                    loss_entropy, loss_ortho, loss_physical = hg_losses
                elif len(hg_losses) == 2:
                    loss_entropy, loss_ortho = hg_losses
            else:
                for m in real_model.modules():
                    if hasattr(m, 'get_loss') and callable(m.get_loss):
                        res = m.get_loss()
                        if isinstance(res, tuple):
                            if len(res) >= 2:
                                loss_entropy += res[0]
                                loss_ortho += res[1]
                            if len(res) >= 3:
                                loss_physical += res[2]

            # ✅ [核心修复3] 数值稳定性检查
            for name, loss_val in [('entropy', loss_entropy), ('ortho', loss_ortho),
                                   ('physical', loss_physical), ('reg', loss_reg),
                                   ('bone_cos', loss_bone_cos)]:
                if not torch.isfinite(loss_val):
                    self.io.print_log(f'[WARNING] Non-finite {name} loss detected: {loss_val.item():.6f}')
                    loss_val = torch.tensor(0.0, device=self.dev)

            # 获取权重参数
            lambda_reg = getattr(self.arg, 'lambda_reg', 0.1)
            lambda_bone = getattr(self.arg, 'lambda_bone', 0.05)
            lambda_ent = getattr(self.arg, 'lambda_entropy', 0.001)
            lambda_ort = getattr(self.arg, 'lambda_ortho', 0.1)
            lambda_phy = getattr(self.arg, 'lambda_physical', 0.1)

            # 总损失
            loss = loss_ce + \
                   (lambda_reg * loss_reg) + \
                   (lambda_bone * loss_bone_cos) + \
                   (lambda_ent * loss_entropy) + \
                   (lambda_ort * loss_ortho) + \
                   (lambda_phy * loss_physical)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            if self.arg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.arg.grad_clip_norm
                )

            self.optimizer.step()

            # 日志记录
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['loss_ce'] = loss_ce.data.item()
            self.iter_info['loss_reg'] = loss_reg.data.item()
            self.iter_info['loss_bone'] = loss_bone_cos.data.item()
            self.iter_info['loss_ent'] = loss_entropy.data.item()
            self.iter_info['loss_orth'] = loss_ortho.data.item()
            self.iter_info['loss_phy'] = loss_physical.data.item()
            if hasattr(self, 'optimizer'):
                self.iter_info['lr'] = self.optimizer.param_groups[0]['lr']

            for k in loss_dict.keys():
                if k in self.iter_info:
                    loss_dict[k].append(self.iter_info[k])

            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        # Epoch统计
        for k in loss_dict.keys():
            if len(loss_dict[k]) > 0:
                self.epoch_info[f'mean_{k.replace("loss_", "")}'] = np.mean(loss_dict[k])
                self.train_writer.add_scalar(
                    k,
                    self.epoch_info[f'mean_{k.replace("loss_", "")}'],
                    epoch
                )

        self.show_epoch_info()

    def test(self, epoch):
        self.model.eval()

        # 安全检查
        is_multi_stream_model = 'DualBranch' in self.arg.model or 'MultiStream' in self.arg.model
        if is_multi_stream_model:
            if self.arg.stream != 'joint' or self.arg.test_feeder_args.get('bone', False):
                self.arg.stream = 'joint'
                self.arg.test_feeder_args['bone'] = False
                self.arg.test_feeder_args['vel'] = False

        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for batch_data in loader:
            data = batch_data[0].float().to(self.dev, non_blocking=True)
            label = batch_data[1].long().to(self.dev, non_blocking=True)

            with torch.no_grad():
                output = self.model(data)

            # 安全检查
            if isinstance(output, tuple):
                output = output[0]

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
            description='Spatial Temporal Graph Convolution Network')

        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--stream', type=str, default='joint', help='the stream of input')

        parser.add_argument('--lambda_entropy', type=float, default=0.001,
                            help='Weight for Entropy loss')
        parser.add_argument('--lambda_ortho', type=float, default=0.1,
                            help='Weight for Orthogonality loss')
        parser.add_argument('--lambda_physical', type=float, default=0.1,
                            help='Weight for Physical Orthogonality loss')
        parser.add_argument('--lambda_reg', type=float, default=0.1,
                            help='Weight for Regression loss')
        parser.add_argument('--lambda_bone', type=float, default=0.05,
                            help='Weight for Bone direction consistency loss')
        parser.add_argument('--lambda_sparsity', type=float, default=0.0,
                            help='Deprecated alias for entropy weight')

        parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                            help='Gradient clipping norm')

        parser.add_argument('--mining_epoch', type=int, default=1e6,
                            help='the starting epoch of nearest neighbor mining')
        parser.add_argument('--topk', type=int, default=1,
                            help='topk samples in nearest neighbor mining')

        parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                            help='decay rate for learning rate')
        parser.add_argument('--warm_up_epoch', type=int, default=0,
                            help='warm up epochs')

        return parser