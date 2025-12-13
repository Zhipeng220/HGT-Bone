import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_modules import unit_tcn, unit_gcn, weights_init, bn_init, MultiScale_TemporalConv, import_class
# [MODIFIED] 引入新的物理引导超图类
from .hypergraph_modules import unit_hypergcn, PhysicallyGuidedDSAHypergraph


# =============================================================================
# Core Modules: Physics Attention & Regression Head
# =============================================================================

class PhysicsAttention(nn.Module):
    """
    [FIXED] Physics-Guided Attention Module (Scheme B)
    使用了基于 Embedding 的非线性距离偏置，替代了原本的线性缩放。
    """

    def __init__(self, in_channels, num_point, hop_grid, num_heads=8):
        super(PhysicsAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = in_channels // num_heads
        self.scale = self.d_model ** -0.5

        self.qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)

        # --- 处理拓扑距离矩阵 (Hop Grid) ---
        hop_grid = np.array(hop_grid)
        # 找出最大有效跳数 (排除 inf)
        valid_hops = hop_grid[~np.isinf(hop_grid)]
        # 如果全是 inf (例如初始化错误), 默认为 3
        max_hop = int(np.max(valid_hops)) if len(valid_hops) > 0 else 3

        # 将 inf 标记为 max_hop + 1, 作为 "remote" 的索引
        hop_grid[np.isinf(hop_grid)] = max_hop + 1

        # 注册为 buffer (随模型保存但不更新梯度)
        self.register_buffer('hop_grid', torch.from_numpy(hop_grid).long())

        # --- [FIX] 使用 Embedding 实现非线性映射 ---
        # 词表大小 = 0...max_hop (共 max_hop+1 个) + 1 (remote)
        self.bias_embedding = nn.Embedding(max_hop + 2, 1)

        # --- [FIX] 初始化策略：距离越近，Attention Bias 越大 ---
        with torch.no_grad():
            # 默认给一个较大的负值 (类似于 Mask)，抑制远距离连接
            self.bias_embedding.weight.data.fill_(-10.0)

            # Hop 0 (Self): 强增强 (1.0)
            self.bias_embedding.weight.data[0] = 1.0

            # Hop 1 (Direct Neighbor): 中等增强 (0.5)
            if max_hop >= 1:
                self.bias_embedding.weight.data[1] = 0.5

            # Hop 2 (Second Neighbor): 弱增强 (0.1)
            if max_hop >= 2:
                self.bias_embedding.weight.data[2] = 0.1

            # Remote: 保持 -10.0 或更低

    def forward(self, x):
        N, C, T, V = x.shape

        # 1. 生成 Q, K, V
        # (N, 3C, T, V) -> (N, 3, H, D, T, V)
        qkv = self.qkv(x).view(N, 3, self.num_heads, self.d_model, T, V)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # 2. 调整维度以计算 Attention: (N, T, H, V, D)
        # 这里的目的是让 Attention 发生在 V x V 维度上，同时保持 T 维度独立或广播
        q = q.permute(0, 3, 1, 4, 2).contiguous().view(-1, self.num_heads, V, self.d_model)
        k = k.permute(0, 3, 1, 4, 2).contiguous().view(-1, self.num_heads, V, self.d_model)
        v = v.permute(0, 3, 1, 4, 2).contiguous().view(-1, self.num_heads, V, self.d_model)

        # 3. 计算 Raw Attention Scores: (NT, H, V, V)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 4. [FIX] 注入物理结构偏置
        # hop_grid: (V, V) -> Embedding -> (V, V, 1) -> Permute -> (1, 1, V, V)
        # 这样可以广播到所有 Batch, Time 和 Heads
        bias = self.bias_embedding(self.hop_grid).permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias

        # 5. Softmax & Aggregate
        attn = attn.softmax(dim=-1)
        x_att = attn @ v

        # 6. 恢复原始维度 (N, C, T, V)
        x_att = x_att.view(N, T, self.num_heads, V, self.d_model)
        # (N, T, H, V, D) -> (N, H, D, T, V) -> (N, C, T, V)
        x_att = x_att.permute(0, 2, 4, 1, 3).contiguous().view(N, C, T, V)

        out = self.proj(x_att) + x
        return out


class JointRegressionHead(nn.Module):
    """
    [NEW] Joint Space Regression Head (Scheme B Auxiliary Task)
    将高维特征映射回 3D 关节坐标，用于正则化。
    """

    def __init__(self, in_channels, out_channels=3):
        super(JointRegressionHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, target_T):
        """
        Args:
            x: 特征图 (N, C, T_down, V)
            target_T: 原始输入的时间长度 T
        Returns:
            out: 回归的关节坐标 (N, 3, T_orig, V)
        """
        out = self.conv(x)
        # 如果特征层经过了时间下采样，需要插值回原始 T 长度以便计算 Loss
        if out.size(2) != target_T:
            out = F.interpolate(out, size=(target_T, out.size(3)), mode='bilinear', align_corners=False)
        return out


# =============================================================================
# Basic Units
# =============================================================================

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5,
                 dilations=[1, 2], num_hyperedges=16, num_point=21, **kwargs):
        super(TCN_GCN_unit, self).__init__()

        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)

        # [CRITICAL] 显式传递 num_point 给 unit_hypergcn
        self.hypergcn1 = unit_hypergcn(in_channels, out_channels,
                                       num_hyperedges=num_hyperedges,
                                       num_point=num_point,
                                       **kwargs)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=False)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        z_gcn = self.gcn1(x)
        z_hyp = self.hypergcn1(x)
        alpha = self.gate(x)

        z_fused = alpha * z_gcn + (1 - alpha) * z_hyp

        y = self.relu(self.tcn1(z_fused) + self.residual(x))
        return y


# =============================================================================
# Main Model Class
# =============================================================================

class Model(nn.Module):
    """
    Backbone Model for DSA-HGN-B (Physically Guided Fusion Version)
    Integrated Scheme B: PhysicsAttention + Joint Regression Head + Hop-based Bias
    """

    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, num_hyperedges=16,
                 base_channels=64, num_stages=10, inflate_stages=[5, 8], down_stages=[5, 8],
                 pretrained=None, data_bn_type='VC', ch_ratio=2, **kwargs):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError("Graph definition is required.")
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = base_channels

        # --- TCN-GCN Layers (Backbone) ---
        # 必须显式传递 num_point，因为 Hypergraph 模块需要它
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive,
                               num_hyperedges=num_hyperedges, num_point=self.num_point, **kwargs)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, num_hyperedges=num_hyperedges,
                               num_point=self.num_point, **kwargs)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, num_hyperedges=num_hyperedges,
                               num_point=self.num_point, **kwargs)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, num_hyperedges=num_hyperedges,
                               num_point=self.num_point, **kwargs)
        self.l5 = TCN_GCN_unit(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive,
                               num_hyperedges=num_hyperedges, num_point=self.num_point, **kwargs)
        self.l6 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive, num_hyperedges=num_hyperedges,
                               num_point=self.num_point, **kwargs)
        self.l7 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive, num_hyperedges=num_hyperedges,
                               num_point=self.num_point, **kwargs)
        self.l8 = TCN_GCN_unit(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive,
                               num_hyperedges=num_hyperedges, num_point=self.num_point, **kwargs)

        # --- [FIX] Physics Attention Module (Scheme B Step 1) ---
        # 尝试从 Graph 对象获取计算好的 hop_dis
        if hasattr(self.graph, 'hop_dis'):
            hop_matrix = self.graph.hop_dis
        else:
            print("[WARNING] Graph does not have 'hop_dis'. PhysicsAttention will use zeros (Degraded Mode).")
            hop_matrix = np.zeros((num_point, num_point))

        self.phys_att = PhysicsAttention(base_channel * 4, self.num_point, hop_matrix)

        self.l9 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive, num_hyperedges=num_hyperedges,
                               num_point=self.num_point, **kwargs)
        self.l10 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive, num_hyperedges=num_hyperedges,
                                num_point=self.num_point, **kwargs)

        # --- [NEW] Joint Regression Head (Scheme B Step 2) ---
        # 将 l10 的输出 (base_channel * 4) 映射回 3 维坐标
        self.regression_head = JointRegressionHead(base_channel * 4, out_channels=3)

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x, drop=False, return_features=False):
        # 1. 预处理 (NaN Check & Reshape)
        x = torch.nan_to_num(x, nan=0.0)

        # 兼容 (N, T, VC) 格式
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)

        N, C, T, V, M = x.size()

        # 2. BatchNorm & Permute
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # 记录原始时间长度 T, 用于回归头插值
        T_orig = T

        # 3. Backbone Forward
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)

        # 4. [FIX] Apply Physics Attention
        x = self.phys_att(x)

        x = self.l9(x)
        z = self.l10(x)

        # 5. [NEW] Joint Space Regression
        # 输入 z: (N*M, C_out, T_down, V) -> 输出: (N*M, 3, T_orig, V)
        predicted_joints = self.regression_head(z, T_orig)

        # 6. Global Pooling & Classification
        c_new = z.size(1)
        x_gap = z.view(N, M, c_new, -1)
        x_gap = x_gap.mean(3).mean(1)

        x_out = self.drop_out(x_gap)

        if return_features:
            return x_out, z

        logits = self.fc(x_out)

        # [MODIFIED] 训练模式下同时返回 logits 和 回归预测值
        if self.training:
            return logits, predicted_joints
        else:
            return logits

    def get_hypergraph_loss(self):
        """
        收集所有层中超图模块的正则化 Loss
        """
        total_entropy = 0
        total_proto_ortho = 0
        total_phy_ortho = 0
        count = 0

        layers = [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7, self.l8, self.l9, self.l10]

        for layer in layers:
            # 检查层中是否包含 hypergcn1 且包含 dhg 模块
            if hasattr(layer, 'hypergcn1') and hasattr(layer.hypergcn1, 'dhg'):
                module = layer.hypergcn1.dhg
                # 调用 PhysicallyGuidedDSAHypergraph.get_loss()
                if hasattr(module, 'get_loss'):
                    res = module.get_loss()
                    # 兼容不同返回值的模块
                    if isinstance(res, tuple):
                        if len(res) == 3:
                            ent, proto, phy = res
                            total_entropy += ent
                            total_proto_ortho += proto
                            total_phy_ortho += phy
                        elif len(res) == 2:
                            ent, proto = res
                            total_entropy += ent
                            total_proto_ortho += proto
                    count += 1

        if count > 0:
            return total_entropy / count, total_proto_ortho / count, total_phy_ortho / count
        else:
            device = self.data_bn.weight.device
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    def get_hypergraph_l1_loss(self):
        return torch.tensor(0.0, device=self.data_bn.weight.device)


# =============================================================================
# Helper Blocks for Multi-Stream (Retained)
# =============================================================================

class ChannelDifferentialBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.diff_conv = nn.Sequential(
            nn.Conv2d(in_channels - 1, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_diff = x[:, 1:, :, :] - x[:, :-1, :, :]
        out = self.diff_conv(x_diff)
        return out


class DualBranchDSA_HGN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 **kwargs):
        super().__init__()

        self.st_branch = Model(num_class=num_class, num_point=num_point, num_person=num_person,
                               graph=graph, graph_args=graph_args, in_channels=in_channels, **kwargs)

        self.diff_prep = ChannelDifferentialBlock(in_channels)
        self.diff_branch = Model(num_class=num_class, num_point=num_point, num_person=num_person,
                                 graph=graph, graph_args=graph_args, in_channels=in_channels, **kwargs)

        base_channel = kwargs.get('base_channels', 64)
        feature_dim = base_channel * 4

        self.fusion_fc = nn.Linear(feature_dim * 2, num_class)

    def forward(self, x, drop=False, return_features=False):
        x_st = x

        N, C, T, V, M = x.shape
        x_reshaped = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        x_diff = self.diff_prep(x_reshaped)
        x_diff = x_diff.view(N, M, C, T, V).permute(0, 2, 3, 4, 1).contiguous()

        feat_st, z_st = self.st_branch(x_st, return_features=True)
        feat_diff, z_diff = self.diff_branch(x_diff, return_features=True)

        feat_fused = torch.cat([feat_st, feat_diff], dim=1)

        if return_features:
            return feat_fused, z_st

        out = self.fusion_fc(feat_fused)
        return out


class HypergraphAttentionFusion(nn.Module):
    def __init__(self, in_channels, num_streams=4):
        super().__init__()
        self.num_streams = num_streams

        self.attn_conv = nn.Sequential(
            nn.Linear(in_channels * num_streams, in_channels * num_streams // 2),
            nn.ReLU(),
            nn.Linear(in_channels * num_streams // 2, num_streams),
            nn.Softmax(dim=1)
        )

    def forward(self, features_list):
        features_stack = torch.stack(features_list, dim=1)
        features_cat = torch.cat(features_list, dim=1)

        attn_weights = self.attn_conv(features_cat)

        attn_weights = attn_weights.unsqueeze(-1)
        fused_feature = (features_stack * attn_weights).sum(dim=1)

        return fused_feature, attn_weights


class MultiStreamDSA_HGN(nn.Module):
    def __init__(self, model_args, num_class=14, streams=['joint', 'bone', 'joint_motion', 'bone_motion']):
        super().__init__()
        self.streams = streams
        self.num_streams = len(streams)

        self.backbones = nn.ModuleList([
            DualBranchDSA_HGN(num_class=num_class, **model_args)
            for _ in range(self.num_streams)
        ])

        base_channel = model_args.get('base_channels', 64)
        feature_dim = base_channel * 4 * 2

        self.hafm = HypergraphAttentionFusion(feature_dim, num_streams=self.num_streams)
        self.fc = nn.Linear(feature_dim, num_class)

        self.bone_pairs = []
        if 'graph' in model_args:
            Graph = import_class(model_args['graph'])
            graph_args = model_args.get('graph_args', {})
            graph = Graph(**graph_args)
            if hasattr(graph, 'inward'):
                self.bone_pairs = graph.inward
            else:
                print("Warning: Graph does not have 'inward' attribute. Bone stream will be zero.")

    def forward(self, x_joint):
        inputs = []
        inputs.append(x_joint)

        x_bone = None
        if self.num_streams > 1:
            x_bone = torch.zeros_like(x_joint)
            if self.bone_pairs:
                for v1, v2 in self.bone_pairs:
                    x_bone[:, :, :, v1, :] = x_joint[:, :, :, v1, :] - x_joint[:, :, :, v2, :]
            inputs.append(x_bone)

        if self.num_streams > 2:
            x_jm = torch.zeros_like(x_joint)
            x_jm[:, :, :-1, :, :] = x_joint[:, :, 1:, :, :] - x_joint[:, :, :-1, :, :]
            inputs.append(x_jm)

        if self.num_streams > 3:
            if x_bone is None:
                x_bone = torch.zeros_like(x_joint)
                if self.bone_pairs:
                    for v1, v2 in self.bone_pairs:
                        x_bone[:, :, :, v1, :] = x_joint[:, :, :, v1, :] - x_joint[:, :, :, v2, :]

            x_bm = torch.zeros_like(x_bone)
            x_bm[:, :, :-1, :, :] = x_bone[:, :, 1:, :, :] - x_bone[:, :, :-1, :, :]
            inputs.append(x_bm)

        features = []
        for i, backbone in enumerate(self.backbones):
            if i < len(inputs):
                feat, _ = backbone(inputs[i], return_features=True)
                features.append(feat)

        fused_feat, attn = self.hafm(features)
        out = self.fc(fused_feat)

        return out