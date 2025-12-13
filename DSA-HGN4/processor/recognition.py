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
    Supports Physically Guided DSA-HGN.
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

        # [SAFETY GUARD]
        is_multi_stream_model = 'DualBranch' in self.arg.model or 'MultiStream' in self.arg.model
        if is_multi_stream_model:
            if self.arg.stream != 'joint' or self.arg.train_feeder_args.get('bone', False):
                self.io.print_log(
                    f'[WARNING] Detected {self.arg.model}. Forcing stream="joint" to prevent Double-Diff.')
                self.arg.stream = 'joint'
                self.arg.train_feeder_args['bone'] = False
                self.arg.train_feeder_args['vel'] = False
                if 'bone' in self.arg.test_feeder_args: self.arg.test_feeder_args['bone'] = False
                if 'vel' in self.arg.test_feeder_args: self.arg.test_feeder_args['vel'] = False

        loader = self.data_loader['train']
        loss_value = []
        entropy_loss_value = []
        ortho_loss_value = []
        phy_loss_value = []
        reg_loss_value = []  # [NEW]
        bone_loss_value = []  # [NEW]

        # [MODIFIED] Unpack 4 values from loader
        for data, label, index, gt_joints_raw in loader:
            self.global_step += 1
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            gt_joints_raw = gt_joints_raw.float().to(self.dev, non_blocking=True)

            if self.arg.stream == 'bone':
                # [FIX] 如果是 8 通道输入 (HGT模式)，说明 Feeder 已经处理好了特征，
                # 跳过手动骨骼计算
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
                            bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]
                        data = bone
                    except Exception:
                        pass

            # [MODIFIED] Model returns (logits, predicted_joints) in training mode
            output, predicted_joints = self.model(data)

            # -----------------------------------------------------------
            # 1. Classification Loss
            # -----------------------------------------------------------
            loss_ce = self.loss(output, label)

            # -----------------------------------------------------------
            # 2. Regression Loss (Scheme B)
            # -----------------------------------------------------------
            # Prepare Ground Truth: (N, 3, T, V, M) -> (N*M, 3, T, V)
            N, C_gt, T, V, M = gt_joints_raw.shape
            gt_joints = gt_joints_raw.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C_gt, T, V)

            # Handle shape mismatch (e.g., if downsampling occurred incorrectly)
            if predicted_joints.shape != gt_joints.shape:
                predicted_joints = F.interpolate(predicted_joints, size=(gt_joints.size(2), gt_joints.size(3)))

            loss_reg = nn.MSELoss()(predicted_joints, gt_joints)

            # -----------------------------------------------------------
            # 3. Bone Direction Loss (Scheme B)
            # -----------------------------------------------------------
            loss_bone_cos = torch.tensor(0.0).to(self.dev)
            real_model = self.model.module if hasattr(self.model, 'module') else self.model

            # Try to get graph edges from model
            if hasattr(real_model, 'graph') and hasattr(real_model.graph, 'edge'):
                edges = [e for e in real_model.graph.edge if e[0] != e[1]]
                if edges:
                    src_idx = [e[0] for e in edges]
                    dst_idx = [e[1] for e in edges]

                    pred_vec = predicted_joints[..., src_idx] - predicted_joints[..., dst_idx]
                    gt_vec = gt_joints[..., src_idx] - gt_joints[..., dst_idx]

                    # Cosine Similarity along channel dim (dim=1)
                    cos_sim = F.cosine_similarity(pred_vec, gt_vec, dim=1)
                    loss_bone_cos = 1.0 - cos_sim.mean()

            # -----------------------------------------------------------
            # 4. Hypergraph / Physical Regularization Loss
            # -----------------------------------------------------------
            loss_entropy = torch.tensor(0.0).to(self.dev)
            loss_ortho = torch.tensor(0.0).to(self.dev)
            loss_physical = torch.tensor(0.0).to(self.dev)

            if hasattr(real_model, 'get_hypergraph_loss'):
                l_ent, l_ortho, l_phy = real_model.get_hypergraph_loss()
                loss_entropy = l_ent
                loss_ortho = l_ortho
                loss_physical = l_phy
            else:
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
            # Total Loss Aggregation
            # -----------------------------------------------------------
            # Weights for new losses
            lambda_reg = 0.1
            lambda_bone = 0.05

            loss = loss_ce + \
                   (self.arg.lambda_entropy * loss_entropy) + \
                   (self.arg.lambda_ortho * loss_ortho) + \
                   (self.arg.lambda_physical * loss_physical) + \
                   (lambda_reg * loss_reg) + \
                   (lambda_bone * loss_bone_cos)

            self.optimizer.zero_grad()
            loss.backward()

            if self.arg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_clip_norm)

            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['loss_ce'] = loss_ce.data.item()
            self.iter_info['loss_ent'] = loss_entropy.data.item()
            self.iter_info['loss_orth'] = loss_ortho.data.item()
            self.iter_info['loss_phy'] = loss_physical.data.item()
            self.iter_info['loss_reg'] = loss_reg.data.item()
            self.iter_info['loss_bone'] = loss_bone_cos.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)

            loss_value.append(self.iter_info['loss'])
            entropy_loss_value.append(self.iter_info['loss_ent'])
            ortho_loss_value.append(self.iter_info['loss_orth'])
            phy_loss_value.append(self.iter_info['loss_phy'])
            reg_loss_value.append(self.iter_info['loss_reg'])
            bone_loss_value.append(self.iter_info['loss_bone'])

            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.epoch_info['mean_ent'] = np.mean(entropy_loss_value)
        self.epoch_info['mean_orth'] = np.mean(ortho_loss_value)
        self.epoch_info['mean_phy'] = np.mean(phy_loss_value)
        self.epoch_info['mean_reg'] = np.mean(reg_loss_value)
        self.epoch_info['mean_bone'] = np.mean(bone_loss_value)

        self.show_epoch_info()
        self.train_writer.add_scalar('loss', self.epoch_info['mean_loss'], epoch)
        self.train_writer.add_scalar('loss_entropy', self.epoch_info['mean_ent'], epoch)
        self.train_writer.add_scalar('loss_ortho', self.epoch_info['mean_orth'], epoch)
        self.train_writer.add_scalar('loss_physical', self.epoch_info['mean_phy'], epoch)
        self.train_writer.add_scalar('loss_reg', self.epoch_info['mean_reg'], epoch)
        self.train_writer.add_scalar('loss_bone', self.epoch_info['mean_bone'], epoch)

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

