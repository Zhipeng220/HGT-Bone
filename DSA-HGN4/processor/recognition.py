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
        else:
            pass

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()

        # [SAFETY GUARD] 多流模型强制检查
        is_multi_stream_model = 'DualBranch' in self.arg.model or 'MultiStream' in self.arg.model
        if is_multi_stream_model:
            if self.arg.stream != 'joint' or self.arg.train_feeder_args.get('bone', False):
                self.io.print_log(
                    f'[WARNING] Detected {self.arg.model}. Forcing stream="joint" to prevent Double-Diff.')
                self.arg.stream = 'joint'
                self.arg.train_feeder_args['bone'] = False
                self.arg.train_feeder_args['vel'] = False

        loader = self.data_loader['train']
        loss_dict = {k: [] for k in ['loss', 'loss_ce', 'loss_ent', 'loss_orth', 'loss_phy', 'loss_reg', 'loss_bone']}

        # [MODIFIED] Unpack 4 values: data, label, index, gt_joints_raw
        for data, label, index, gt_joints_raw in loader:
            self.global_step += 1
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            gt_joints_raw = gt_joints_raw.float().to(self.dev, non_blocking=True)

            # -----------------------------------------------------------
            # 0. Data Preprocessing (Bone Stream On-the-fly Calculation)
            # -----------------------------------------------------------
            if self.arg.stream == 'bone':
                # 如果是 8 通道输入 (HGT模式) 或 Feeder 已开启 Bone，跳过计算
                if self.arg.model_args.get('in_channels', 3) == 8:
                    pass
                elif self.arg.train_feeder_args.get('bone', False):
                    pass
                else:
                    try:
                        from net.utils.graph import Graph
                        layout = self.arg.model_args['graph_args'].get('layout', 'ntu-rgb+d')
                        graph = Graph(layout)
                        bone = torch.zeros_like(data)
                        for v1, v2 in graph.Bones:
                            # v1, v2 are 1-based in some graph defs, ensuring 0-based
                            bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]
                        data = bone
                    except Exception:
                        pass

            # -----------------------------------------------------------
            # 1. Forward Pass
            # -----------------------------------------------------------
            # output: Logits (N, Num_Class)
            # predicted_val: Regression Output (N, 3, T, V_model)
            #   - Joint Stream: V_model = 22 (Joints)
            #   - Bone Stream:  V_model = 21 (Bones)
            output, predicted_val = self.model(data)

            # Classification Loss
            loss_ce = self.loss(output, label)

            # -----------------------------------------------------------
            # 2. Regression Loss & Bone Direction Loss (Scheme B: Physically Aligned)
            # -----------------------------------------------------------
            # 原始 GT 关节数据: (N, 3, T, V_gt, M) -> (N*M, 3, T, V_gt)
            # V_gt 通常为 22 (SHREC) 或 25 (NTU)
            N, C_gt, T, V_gt, M = gt_joints_raw.shape
            gt_data = gt_joints_raw.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C_gt, T, V_gt)

            # 动态获取模型
            real_model = self.model.module if hasattr(self.model, 'module') else self.model

            if self.arg.stream == 'bone':
                # =======================================================
                # Bone Stream Strategy: Bone-to-Bone Alignment
                # Target: Ground Truth Bone Vectors (V=21)
                # Pred:   Model Output Vectors (V=21)
                # =======================================================

                # 1. 获取骨骼连接定义
                if hasattr(real_model.graph, 'source_edges'):
                    bone_pairs = real_model.graph.source_edges  # [(0,1), (1,2)...]
                else:
                    # Fallback: SHREC'17 Standard Layout
                    bone_pairs = [
                        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
                        (1, 6), (6, 7), (7, 8), (8, 9),
                        (1, 10), (10, 11), (11, 12), (12, 13),
                        (1, 14), (14, 15), (15, 16), (16, 17),
                        (1, 18), (18, 19), (19, 20), (20, 21)
                    ]

                # 2. 实时计算 Ground Truth Bone Vectors
                # logic: vector = parent - child or child - parent depending on definition.
                # Here we strictly follow: vec = J[v1] - J[v2]
                gt_bone_list = []
                for v1, v2 in bone_pairs:
                    # 确保索引在 GT 范围内
                    if v1 < V_gt and v2 < V_gt:
                        vec = gt_data[..., v1] - gt_data[..., v2]
                        gt_bone_list.append(vec)

                # Stack to (N*M, 3, T, 21)
                if len(gt_bone_list) > 0:
                    target_data = torch.stack(gt_bone_list, dim=-1)
                else:
                    # Fallback if pairing fails
                    target_data = torch.zeros_like(predicted_val)

                # 3. 预测值 (N*M, 3, T, 21)
                pred_data = predicted_val

                # [Safety] 时间维度对齐 (处理 Pooling 导致的 T 不一致)
                if pred_data.shape[2] != target_data.shape[2]:
                    pred_data = F.interpolate(pred_data, size=(target_data.size(2), target_data.size(3)),
                                              mode='bilinear', align_corners=False)

                # 计算 MSE (回归骨骼向量)
                loss_reg = nn.MSELoss()(pred_data, target_data)

                # 计算 Cosine Direction Loss (直接向量相似度)
                # dim=1 is Channel (x,y,z)
                cos_sim = F.cosine_similarity(pred_data, target_data, dim=1)
                loss_bone_cos = 1.0 - cos_sim.mean()

            else:
                # =======================================================
                # Joint Stream Strategy: Joint-to-Joint Alignment
                # Target: Ground Truth Joint Coordinates (V=22)
                # Pred:   Model Output Coordinates (V=22)
                # =======================================================
                target_data = gt_data
                pred_data = predicted_val

                # 维度对齐
                if pred_data.shape[-1] != target_data.shape[-1] or pred_data.shape[2] != target_data.shape[2]:
                    pred_data = F.interpolate(pred_data, size=(target_data.size(2), target_data.size(3)),
                                              mode='bilinear', align_corners=False)

                loss_reg = nn.MSELoss()(pred_data, target_data)

                # Joint 流的 Bone Cos Loss (需先由关节算出向量)
                loss_bone_cos = torch.tensor(0.0).to(self.dev)

                # 获取连接定义
                edges = []
                if hasattr(real_model.graph, 'source_edges'):
                    edges = real_model.graph.source_edges
                elif hasattr(real_model.graph, 'edge'):
                    edges = [e for e in real_model.graph.edge if e[0] != e[1]]  # 排除自环

                if edges:
                    try:
                        src_idx = [e[0] for e in edges if e[0] < pred_data.shape[-1] and e[1] < pred_data.shape[-1]]
                        dst_idx = [e[1] for e in edges if e[0] < pred_data.shape[-1] and e[1] < pred_data.shape[-1]]

                        if len(src_idx) > 0:
                            pred_vec = pred_data[..., src_idx] - pred_data[..., dst_idx]
                            gt_vec = target_data[..., src_idx] - target_data[..., dst_idx]

                            cos_sim = F.cosine_similarity(pred_vec, gt_vec, dim=1)
                            loss_bone_cos = 1.0 - cos_sim.mean()
                    except Exception:
                        pass

            # -----------------------------------------------------------
            # 3. Hypergraph / Physical Regularization Loss
            # -----------------------------------------------------------
            loss_entropy = torch.tensor(0.0).to(self.dev)
            loss_ortho = torch.tensor(0.0).to(self.dev)
            loss_physical = torch.tensor(0.0).to(self.dev)

            # 优先调用顶层的 get_hypergraph_loss
            if hasattr(real_model, 'get_hypergraph_loss'):
                l_ent, l_ortho, l_phy = real_model.get_hypergraph_loss()
                loss_entropy = l_ent
                loss_ortho = l_ortho
                loss_physical = l_phy
            else:
                # 遍历子模块搜集 Loss (备用方案)
                for m in real_model.modules():
                    if hasattr(m, 'get_loss') and callable(m.get_loss):
                        res = m.get_loss()
                        if isinstance(res, tuple):
                            if len(res) == 2:
                                l_e, l_o = res
                                loss_entropy += l_e
                                loss_ortho += l_o
                            elif len(res) == 3:
                                l_e, l_o, l_p = res
                                loss_entropy += l_e
                                loss_ortho += l_o
                                loss_physical += l_p

            # -----------------------------------------------------------
            # 4. Total Loss Aggregation
            # -----------------------------------------------------------
            # Dynamic Weights from Args (Hardcoded defaults as fallback)
            # 建议将 lambda_reg 和 lambda_bone 添加到 argparse 或 yaml 中配置
            lambda_reg = getattr(self.arg, 'lambda_reg', 0.1)
            lambda_bone = getattr(self.arg, 'lambda_bone', 0.05)

            loss = loss_ce + \
                   (self.arg.lambda_entropy * loss_entropy) + \
                   (self.arg.lambda_ortho * loss_ortho) + \
                   (self.arg.lambda_physical * loss_physical) + \
                   (lambda_reg * loss_reg) + \
                   (lambda_bone * loss_bone_cos)

            # Backward & Optimize
            self.optimizer.zero_grad()
            loss.backward()

            if self.arg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_clip_norm)

            self.optimizer.step()

            # -----------------------------------------------------------
            # 5. Logging
            # -----------------------------------------------------------
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['loss_ce'] = loss_ce.data.item()
            self.iter_info['loss_ent'] = loss_entropy.data.item()
            self.iter_info['loss_orth'] = loss_ortho.data.item()
            self.iter_info['loss_phy'] = loss_physical.data.item()
            self.iter_info['loss_reg'] = loss_reg.data.item()
            self.iter_info['loss_bone'] = loss_bone_cos.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)

            for k in loss_dict.keys():
                loss_dict[k].append(self.iter_info[k])

            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        # Epoch info logging
        for k in loss_dict.keys():
            self.epoch_info[f'mean_{k.replace("loss_", "")}'] = np.mean(loss_dict[k])
            self.train_writer.add_scalar(k, self.epoch_info[f'mean_{k.replace("loss_", "")}'], epoch)

        self.show_epoch_info()

    def test(self, epoch):
        self.model.eval()

        # [SAFETY GUARD]
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

        # [MODIFIED] Unpack safely (handle 3 or 4 items)
        for batch_data in loader:
            data = batch_data[0].float().to(self.dev, non_blocking=True)
            label = batch_data[1].long().to(self.dev, non_blocking=True)
            # Ignore index (batch_data[2]) and gt_joints (batch_data[3]) in test

            if self.arg.stream == 'bone':
                # [FIX] 测试阶段同样跳过手动骨骼计算
                if self.arg.model_args.get('in_channels', 3) == 8:
                    pass
                elif self.arg.test_feeder_args.get('bone', False):
                    pass
                else:
                    try:
                        from net.utils.graph import Graph
                        layout = self.arg.model_args['graph_args'].get('layout', 'ntu-rgb+d')
                        graph = Graph(layout)
                        bone = torch.zeros_like(data)
                        for v1, v2 in graph.Bones:
                            bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]
                        data = bone
                    except Exception:
                        pass

            with torch.no_grad():
                output = self.model(data)

            # Model in eval mode returns only logits (based on modification)
            # Safety check just in case
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

        # [Optimization] Regularization Parameters
        parser.add_argument('--lambda_entropy', type=float, default=0.001,
                            help='Weight for Entropy loss (Soft Sparsity)')
        parser.add_argument('--lambda_ortho', type=float, default=0.1, help='Weight for Orthogonality loss')

        # [NEW] Physical Constraint Weight
        parser.add_argument('--lambda_physical', type=float, default=0.1,
                            help='Weight for Physical Orthogonality loss (HGT-Bone Fusion)')

        # Kept for backward compatibility
        parser.add_argument('--lambda_sparsity', type=float, default=0.0, help='Deprecated alias for entropy weight')

        parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Gradient clipping norm')

        parser.add_argument('--mining_epoch', type=int, default=1e6,
                            help='the starting epoch of nearest neighbor mining')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in nearest neighbor mining')

        parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
        parser.add_argument('--warm_up_epoch', type=int, default=0, help='warm up epochs')

        return parser