"""
recognition.py - Final Training Processor v2.2 (Patch 2 Fixed)
============================================================
综合Claude+Grok二轮审核 + 补丁修复

修复内容:
1. ✅ 修复缺失的 show_topk 方法
2. ✅ 保留 Mixup, Early Stopping, Overfit Monitor 等核心功能
============================================================
"""

import sys
import argparse
import yaml
import math
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor


def weights_init(m):
    """权重初始化"""
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
    ✅ [最终版 v2.2] 训练处理器 - 完整修复版

    支持:
    - Label Smoothing
    - Mixup训练
    - 早停机制 (Top1 + val_loss)
    - 过拟合监控 (阈值5x)
    - 温度退火
    - 详细超图损失日志
    """

    def load_model(self):
        """加载模型和损失函数"""
        self.model = self.io.load_model(self.arg.model, **(self.arg.model_args))
        self.model.apply(weights_init)

        # Label Smoothing
        label_smoothing = getattr(self.arg, 'label_smoothing', 0.0)
        if label_smoothing > 0:
            self.io.print_log(f'[CONFIG] Label Smoothing: {label_smoothing}')
            self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.loss = nn.CrossEntropyLoss()

        # ✅ Mixup参数
        self.use_mixup = getattr(self.arg, 'use_mixup', False)
        self.mixup_alpha = getattr(self.arg, 'mixup_alpha', 0.2)
        if self.use_mixup:
            self.io.print_log(f'[CONFIG] Mixup enabled: alpha={self.mixup_alpha}')

        # 监控变量
        self.last_eval_loss = float('inf')
        self.train_val_ratios = []
        self.best_val_loss = float('inf')

    def load_optimizer(self):
        """加载优化器"""
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
            self.io.print_log(f'[CONFIG] AdamW optimizer, lr={self.arg.base_lr}, wd={self.arg.weight_decay}')
        else:
            raise ValueError(f"Unknown optimizer: {self.arg.optimizer}")

    def adjust_lr(self):
        """
        ✅ 改进版学习率调度器
        - 优化Warmup策略 (Cosine)
        - 添加更详细的日志
        """
        lr_decay_rate = getattr(self.arg, 'lr_decay_rate', 0.1)

        # ============================================================
        # 1. Warmup阶段
        # ============================================================
        if hasattr(self.arg, 'warm_up_epoch') and self.arg.warm_up_epoch > 0:
            if self.meta_info['epoch'] < self.arg.warm_up_epoch:
                # ✅ 改进：使用平滑的warmup曲线（余弦）
                progress = (self.meta_info['epoch'] + 1) / self.arg.warm_up_epoch
                # 平滑warmup
                lr = self.arg.base_lr * (0.5 * (1 + np.cos(np.pi * (1 - progress))))

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.lr = lr

                # ✅ Warmup期间每5个epoch打印一次学习率
                if self.meta_info['epoch'] % 5 == 0:
                    self.io.print_log(f'[LR] Warmup: Epoch {self.meta_info["epoch"]}, lr={lr:.6f}')

                return

        # ============================================================
        # 2. Decay阶段
        # ============================================================
        if self.arg.step:
            # 计算当前应该处于第几个decay阶段
            decay_stage = np.sum(self.meta_info['epoch'] >= np.array(self.arg.step))
            lr = self.arg.base_lr * (lr_decay_rate ** decay_stage)

            # ✅ 检测学习率变化，打印日志
            if hasattr(self, 'lr') and abs(self.lr - lr) > 1e-8:
                self.io.print_log(
                    f'[LR] Step Decay: Epoch {self.meta_info["epoch"]}, '
                    f'lr: {self.lr:.6f} -> {lr:.6f} '
                    f'(stage {decay_stage}/{len(self.arg.step)})'
                )

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        """
        ✅ [修复] 显示Top-K准确率
        """
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100 * sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, accuracy))
        return accuracy

    def show_best(self, k):
        """显示最佳结果"""
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100 * sum(hit_top_k) * 1.0 / len(hit_top_k)
        accuracy = round(accuracy, 5)
        self.current_result = accuracy
        if self.best_result <= accuracy:
            self.best_result = accuracy
            self.export_topology(self.meta_info['epoch'])

        self.io.print_log('\tBest Top{}: {:.2f}%'.format(k, self.best_result))

    def export_topology(self, epoch):
        """导出学习到的超图拓扑"""
        save_path = os.path.join(self.arg.work_dir, f'topology_best_epoch_{epoch}.npy')
        model_core = self.model.module if hasattr(self.model, 'module') else self.model
        target_module = None

        for name, m in model_core.named_modules():
            if hasattr(m, 'last_h_dynamic') and m.last_h_dynamic is not None:
                target_module = m
                H = target_module.last_h_dynamic[0:1]
                break
            elif hasattr(m, 'last_h') and m.last_h is not None:
                target_module = m
                H = target_module.last_h[0:1]
                break

        if target_module is not None:
            A_virtual = torch.matmul(H, H.transpose(-1, -2)).squeeze(0).detach().cpu().numpy()
            np.save(save_path, A_virtual)

    def _update_hypergraph_temperature(self, epoch):
        """更新超图模块温度"""
        real_model = self.model.module if hasattr(self.model, 'module') else self.model

        updated_count = 0
        for module in real_model.modules():
            if hasattr(module, 'set_epoch') and callable(module.set_epoch):
                try:
                    module.set_epoch(epoch)
                    updated_count += 1
                except Exception as e:
                    self.io.print_log(f'[WARNING] Temperature update failed: {e}')

        # ✅ 每5个epoch打印一次
        if epoch % 5 == 0 and updated_count > 0:
            for m in real_model.modules():
                if hasattr(m, 'temperature') and hasattr(m, 'target_entropy'):
                    temp = m.temperature.item() if torch.is_tensor(m.temperature) else m.temperature
                    target_ent = m.target_entropy.item() if torch.is_tensor(m.target_entropy) else m.target_entropy

                    # ✅ 打印更详细的超图状态
                    current_ent = m.loss_components.get('current_entropy', 0) if hasattr(m, 'loss_components') else 0
                    max_off = m.loss_components.get('max_off_diag', 0) if hasattr(m, 'loss_components') else 0
                    synergy_mean = m.loss_components.get('synergy_mean', 0) if hasattr(m, 'loss_components') else 0
                    synergy_max = m.loss_components.get('synergy_max', 0) if hasattr(m, 'loss_components') else 0

                    self.io.print_log(
                        f'[HyperGraph] Epoch {epoch}: T={temp:.4f}, '
                        f'Entropy={current_ent:.3f}/{target_ent:.3f}, '
                        f'MaxOffDiag={max_off:.4f}, '
                        f'Synergy={synergy_mean:.3f}/{synergy_max:.3f}'
                    )
                    break

    def _mixup_data(self, x, y, alpha=0.2):
        """✅ Mixup数据增强"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def _mixup_criterion(self, pred, y_a, y_b, lam):
        """✅ Mixup损失计算"""
        return lam * self.loss(pred, y_a) + (1 - lam) * self.loss(pred, y_b)

    def train(self, epoch):
        """训练一个epoch"""
        self.model.train()
        self.adjust_lr()
        self._update_hypergraph_temperature(epoch - 1)
        loader = self.data_loader['train']

        # 渐进式增强设置
        if hasattr(loader.dataset, 'set_epoch'):
            loader.dataset.set_epoch(epoch - 1)

        # 安全检查
        is_multi_stream_model = 'DualBranch' in self.arg.model or 'MultiStream' in self.arg.model
        if is_multi_stream_model:
            if self.arg.stream != 'joint' or self.arg.train_feeder_args.get('bone', False):
                self.io.print_log(f'[WARNING] Multi-stream detected. stream="joint".')
                self.arg.stream = 'joint'
                self.arg.train_feeder_args['bone'] = False

        loss_dict = {k: [] for k in ['loss', 'loss_ce', 'loss_ent', 'loss_orth',
                                     'loss_phy', 'loss_reg', 'loss_bone']}

        real_model = self.model.module if hasattr(self.model, 'module') else self.model

        SHREC_BONE_PAIRS = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (1, 6), (6, 7), (7, 8), (8, 9),
            (1, 10), (10, 11), (11, 12), (12, 13),
            (1, 14), (14, 15), (15, 16), (16, 17),
            (1, 18), (18, 19), (19, 20), (20, 21)
        ]

        for batch_data in loader:
            self.global_step += 1

            # 解包数据 (支持Mixup返回5个参数的情况)
            if len(batch_data) == 5:
                data, label_a, label_b, lam, index, gt_bone_raw = batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4], batch_data[5]
                # 注意：如果collate_fn改变了，需要适配。
                # 假设标准Mixup collate: data, label_a, label_b, lam, index, bone
                pass # 在feeder中处理，这里简化处理

            # 使用标准的 batch unpacking
            if len(batch_data) == 4:
                data, label, index, gt_bone_raw = batch_data
            else:
                data, label, index, gt_bone_raw = batch_data[0], batch_data[1], batch_data[2], batch_data[3]

            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            gt_bone_raw = gt_bone_raw.float().to(self.dev, non_blocking=True)

            N, C_gt, T, V_gt, M = gt_bone_raw.shape
            gt_bone = gt_bone_raw.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C_gt, T, V_gt)

            # ✅ Mixup应用
            if self.use_mixup:
                data, label_a, label_b, lam = self._mixup_data(data, label, self.mixup_alpha)
            else:
                label_a, label_b, lam = label, label, 1.0

            # 前向传播
            output, predicted_bone = self.model(data)

            # ============================================================
            # 分类损失
            # ============================================================
            if self.use_mixup and lam < 1.0:
                loss_ce = self._mixup_criterion(output, label_a, label_b, lam)
            else:
                loss_ce = self.loss(output, label)

            # ============================================================
            # 回归损失
            # ============================================================
            if self.arg.stream == 'bone':
                target_data = gt_bone
                pred_data = predicted_bone

                if pred_data.shape[2] != target_data.shape[2]:
                    pred_data = F.interpolate(
                        pred_data,
                        size=(target_data.size(2), target_data.size(3)),
                        mode='bilinear', align_corners=False
                    )

                loss_reg = nn.MSELoss()(pred_data, target_data)

                pred_norm = F.normalize(pred_data, p=2, dim=1, eps=1e-8)
                target_norm = F.normalize(target_data, p=2, dim=1, eps=1e-8)
                cos_sim = (pred_norm * target_norm).sum(dim=1).mean()
                cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
                loss_bone_cos = 1.0 - cos_sim
            else:
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
                if hasattr(real_model, 'graph') and hasattr(real_model.graph, 'inward_ori_index'):
                    joint_pairs = real_model.graph.inward_ori_index

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
                                loss_entropy = loss_entropy + res[0]
                                loss_ortho = loss_ortho + res[1]
                            if len(res) >= 3:
                                loss_physical = loss_physical + res[2]

            # 数值稳定性检查
            for name, loss_val in [('entropy', loss_entropy), ('ortho', loss_ortho),
                                   ('physical', loss_physical), ('reg', loss_reg),
                                   ('bone_cos', loss_bone_cos)]:
                if not torch.isfinite(loss_val):
                    self.io.print_log(f'[WARNING] Non-finite {name} loss')
                    if name == 'entropy':
                        loss_entropy = torch.tensor(0.0, device=self.dev)
                    elif name == 'ortho':
                        loss_ortho = torch.tensor(0.0, device=self.dev)
                    elif name == 'physical':
                        loss_physical = torch.tensor(0.0, device=self.dev)
                    elif name == 'reg':
                        loss_reg = torch.tensor(0.0, device=self.dev)
                    else:
                        loss_bone_cos = torch.tensor(0.0, device=self.dev)

            # ============================================================
            # 总损失
            # ============================================================
            lambda_reg = getattr(self.arg, 'lambda_reg', 0.1)
            lambda_bone = getattr(self.arg, 'lambda_bone', 0.05)
            lambda_ent = getattr(self.arg, 'lambda_entropy', 0.001)
            lambda_ort = getattr(self.arg, 'lambda_ortho', 0.1)
            lambda_phy = getattr(self.arg, 'lambda_physical', 0.1)

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

            # 日志
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
                self.train_writer.add_scalar(k, self.epoch_info[f'mean_{k.replace("loss_", "")}'], epoch)

        self.show_epoch_info()

        # ✅ 过拟合监控 (阈值从10x调到5x)
        train_ce = self.epoch_info.get('mean_ce', 0)
        if self.last_eval_loss < float('inf') and train_ce > 0:
            ratio = self.last_eval_loss / train_ce
            self.train_val_ratios.append(ratio)

            overfit_threshold = getattr(self.arg, 'overfit_ratio_threshold', 5.0)

            if ratio > overfit_threshold * 2:
                self.io.print_log(f'[WARNING] Severe overfitting! Ratio: {ratio:.1f}x (>{overfit_threshold * 2})')
            elif ratio > overfit_threshold:
                self.io.print_log(f'[WARNING] Overfitting detected. Ratio: {ratio:.1f}x (>{overfit_threshold})')
            elif ratio > overfit_threshold * 0.6:
                self.io.print_log(f'[INFO] Train/Val ratio: {ratio:.1f}x - Monitor closely')

    def test(self, epoch):
        """测试一个epoch"""
        self.model.eval()

        is_multi_stream_model = 'DualBranch' in self.arg.model or 'MultiStream' in self.arg.model
        if is_multi_stream_model:
            if self.arg.stream != 'joint' or self.arg.test_feeder_args.get('bone', False):
                self.arg.stream = 'joint'
                self.arg.test_feeder_args['bone'] = False

        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for batch_data in loader:
            data = batch_data[0].float().to(self.dev, non_blocking=True)
            label = batch_data[1].long().to(self.dev, non_blocking=True)

            with torch.no_grad():
                output = self.model(data)

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
        self.last_eval_loss = self.eval_info['eval_mean_loss']

        self.show_eval_info()

        # ✅ 显式调用 show_topk
        for k in self.arg.show_topk:
            self.show_topk(k)

        self.show_best(1)
        self.eval_log_writer(epoch)

    def start(self):
        """训练主循环"""
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        self.print_networks(self.model)

        if self.arg.phase == 'train':
            self.global_step = self.arg.start_epoch * len(self.data_loader['train'])
            self.meta_info['iter'] = self.global_step
            self.best_result = 0.0

            # 早停参数
            patience = getattr(self.arg, 'patience', 25)
            min_epoch_before_stop = getattr(self.arg, 'min_epoch_before_stop', 40)
            no_improve_count = 0
            best_epoch = 0

            self.io.print_log(f'[CONFIG] Early stopping: patience={patience}, min_epoch={min_epoch_before_stop}')
            self.io.print_log(f'[CONFIG] Mixup: {self.use_mixup}, alpha={self.mixup_alpha}')

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch + 1

                self.io.print_log(f'Training epoch: {epoch + 1}')
                self.train(epoch + 1)

                # 保存
                if self.arg.save_interval == -1:
                    pass
                elif ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch):
                    filename = 'epoch{}_model.pt'.format(epoch + 1)
                    self.io.save_model(self.model, filename)

                # 评估
                if self.arg.eval_interval == -1:
                    pass
                elif ((epoch + 1) % self.arg.eval_interval == 0) or (epoch + 1 == self.arg.num_epoch):
                    self.io.print_log(f'Eval epoch: {epoch + 1}')
                    self.test(epoch + 1)
                    self.io.print_log("current %.2f%%, best %.2f%%" % (self.current_result, self.best_result))

                    filename = 'epoch%.3d_acc%.2f_model.pt' % (epoch + 1, self.current_result)
                    self.io.save_model(self.model, filename)

                    # 早停判断
                    improved = False

                    if self.current_result > self.best_result:
                        self.best_result = self.current_result
                        best_epoch = epoch + 1
                        improved = True

                        self.io.save_model(self.model, 'best_model.pt')
                        if self.arg.save_result:
                            result_dict = dict(zip(self.data_loader['test'].dataset.sample_name, self.result))
                            self.io.save_pkl(result_dict, 'test_result.pkl')

                    if self.last_eval_loss < self.best_val_loss:
                        self.best_val_loss = self.last_eval_loss
                        improved = True

                    if improved:
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                        if epoch + 1 >= min_epoch_before_stop:
                            self.io.print_log(
                                f"[Early Stop] No improvement for {no_improve_count}/{patience} epochs "
                                f"(best={self.best_result:.2f}% @ epoch {best_epoch})"
                            )

                            if no_improve_count >= patience:
                                self.io.print_log(
                                    f"[Early Stop] Stopping at epoch {epoch + 1}. "
                                    f"Best: {self.best_result:.2f}% @ epoch {best_epoch}"
                                )
                                break

            self.io.print_log(f'\n[FINAL] Best accuracy: {self.best_result:.2f}% at epoch {best_epoch}')

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))

            self.best_result = 0.0
            self.io.print_log('Evaluation Start:')
            self.test(1)
            self.io.print_log('Done.\n')

            if self.arg.save_result:
                result_dict = dict(zip(self.data_loader['test'].dataset.sample_name, self.result))
                self.io.save_pkl(result_dict, 'test_result.pkl')

    @staticmethod
    def get_parser(add_help=False):
        """参数解析器"""
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='DSA-HGN Training Processor v2.2')

        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+')
        parser.add_argument('--base_lr', type=float, default=0.01)
        parser.add_argument('--step', type=int, default=[], nargs='+')
        parser.add_argument('--lr_decay_rate', type=float, default=0.1)
        parser.add_argument('--warm_up_epoch', type=int, default=0)
        parser.add_argument('--nesterov', type=str2bool, default=True)
        parser.add_argument('--weight_decay', type=float, default=0.0001)
        parser.add_argument('--stream', type=str, default='joint')

        # 损失权重
        parser.add_argument('--lambda_entropy', type=float, default=0.001)
        parser.add_argument('--lambda_ortho', type=float, default=0.1)
        parser.add_argument('--lambda_physical', type=float, default=0.1)
        parser.add_argument('--lambda_reg', type=float, default=0.1)
        parser.add_argument('--lambda_bone', type=float, default=0.05)
        parser.add_argument('--lambda_sparsity', type=float, default=0.0)

        parser.add_argument('--grad_clip_norm', type=float, default=1.0)

        # 新增参数
        parser.add_argument('--label_smoothing', type=float, default=0.0)
        parser.add_argument('--patience', type=int, default=25)
        parser.add_argument('--min_epoch_before_stop', type=int, default=40)
        parser.add_argument('--overfit_ratio_threshold', type=float, default=5.0)

        # ✅ Mixup参数
        parser.add_argument('--use_mixup', type=str2bool, default=False)
        parser.add_argument('--mixup_alpha', type=float, default=0.2)

        # 废弃参数
        parser.add_argument('--mining_epoch', type=int, default=1e6)
        parser.add_argument('--topk', type=int, default=1)

        return parser