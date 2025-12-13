import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_modules import unit_tcn, unit_gcn, weights_init, bn_init, MultiScale_TemporalConv, import_class
# [MODIFIED] 引入新的物理引导超图类，并保留 unit_hypergcn
from .hypergraph_modules import unit_hypergcn, PhysicallyGuidedDSAHypergraph


# =============================================================================
# Assembly Modules
# =============================================================================

# [NEW] Physics Attention Module (方案B: 引入 PhysicsAttention)
class PhysicsAttention(nn.Module):
    def __init__(self, in_channels, num_point, hop_grid, num_heads=8):
        super(PhysicsAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = in_channels // num_heads
        self.scale = self.d_model ** -0.5

        self.qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)

        # 处理 hop_grid: 将 inf 替换为最大跳数 + 1，以便计算
        hop_grid = np.array(hop_grid)
        if np.isinf(hop_grid).any():
            max_val = np.max(hop_grid[~np.isinf(hop_grid)])
            hop_grid[np.isinf(hop_grid)] = max_val + 1

        self.register_buffer('hop_grid', torch.from_numpy(hop_grid).float())

        # 可学习的物理偏置权重
        self.hop_bias_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        N, C, T, V = x.shape

        # 生成 Q, K, V
        # self.qkv(x) -> (N, 3*C, T, V)
        # view -> (N, 3, H, D, T, V)
        qkv = self.qkv(x).view(N, 3, self.num_heads, self.d_model, T, V)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # q shape: (N, H, D, T, V)
        # Dimensions: 0:N, 1:H, 2:D, 3:T, 4:V
        # Target for Spatial Attention: (N*T, H, V, D)

        # [FIXED] Correct Permutation:
        # Move T(3) to index 1, V(4) to index 3, D(2) to index 4
        # (N, H, D, T, V) -> (N, T, H, V, D)
        # Indices: (0, 3, 1, 4, 2)

        q = q.permute(0, 3, 1, 4, 2).contiguous().view(-1, self.num_heads, V, self.d_model)
        k = k.permute(0, 3, 1, 4, 2).contiguous().view(-1, self.num_heads, V, self.d_model)
        v = v.permute(0, 3, 1, 4, 2).contiguous().view(-1, self.num_heads, V, self.d_model)

        # Attention Score: (NT, H, V, V)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # [Physics Guidance] 添加基于跳数的偏置
        bias = -self.hop_bias_weight * self.hop_grid.unsqueeze(0).unsqueeze(0)
        attn = attn + bias

        attn = attn.softmax(dim=-1)

        # 聚合值: (NT, H, V, D)
        x_att = attn @ v

        # 恢复维度: (N, C, T, V)
        # Inverse: (NT, H, V, D) -> (N, T, H, V, D) -> (N, H, D, T, V) -> (N, C, T, V)
        x_att = x_att.view(N, T, self.num_heads, V, self.d_model)

        # Permute (N, T, H, V, D) -> (N, H, D, T, V)
        # Src: 0:N, 1:T, 2:H, 3:V, 4:D
        # Dst: N(0), H(2), D(4), T(1), V(3)
        x_att = x_att.permute(0, 2, 4, 1, 3).contiguous().view(N, C, T, V)

        out = self.proj(x_att) + x
        return out


# [NEW] Joint Regression Head (方案B: 关节空间回归头)
class JointRegressionHead(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super(JointRegressionHead, self).__init__()
        # 简单的 1x1 卷积将特征映射到 3D 坐标空间
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, target_T):
        """
        Args:
            x: 特征图 (N, C, T_down, V)
            target_T: 原始时间长度
        Returns:
            out: 回归的关节坐标 (N, 3, T_orig, V)
        """
        out = self.conv(x)
        # 如果时间维度被下采样了，插值回原始尺寸
        if out.size(2) != target_T:
            out = F.interpolate(out, size=(target_T, out.size(3)), mode='bilinear', align_corners=False)
        return out


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5,
                 dilations=[1, 2], num_hyperedges=16, num_point=21, **kwargs):
        # [CRITICAL] 显式接收 num_point=21，确保默认值与 HGT-Bone 一致
        super(TCN_GCN_unit, self).__init__()

        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)

        # [MODIFIED] 实例化 unit_hypergcn 时显式传递 num_point
        # 确保 kwargs 中即便没有 num_point，也能通过显式参数传下去
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


class Model(nn.Module):
    """
    Backbone Model for DSA-HGN (Physically Guided Fusion Version)
    Modified with Scheme B: PhysicsAttention + Joint Regression Head
    """

    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, num_hyperedges=16,
                 base_channels=64, num_stages=10, inflate_stages=[5, 8], down_stages=[5, 8],
                 pretrained=None, data_bn_type='VC', ch_ratio=2, **kwargs):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = base_channels

        # Layer definition
        # [CRITICAL FIX] 在初始化每一层时，必须显式传递 num_point=self.num_point

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

        # [NEW] Physics Attention Module (Scheme B Step 1)
        # 插入在较深层（如 Stage 8 之后），此时特征通道数为 base_channel * 4
        # 需要传入 hop_dis (从 graph 对象中获取)
        if hasattr(self.graph, 'hop_dis'):
            hop_matrix = self.graph.hop_dis
        else:
            # Fallback if hop_dis missing
            print("Warning: Graph does not have 'hop_dis'. Using zeros.")
            hop_matrix = np.zeros((num_point, num_point))

        self.phys_att = PhysicsAttention(base_channel * 4, self.num_point, hop_matrix)

        self.l9 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive, num_hyperedges=num_hyperedges,
                               num_point=self.num_point, **kwargs)
        self.l10 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive, num_hyperedges=num_hyperedges,
                                num_point=self.num_point, **kwargs)

        # [NEW] Joint Regression Head (Scheme B Step 2)
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
        # 处理 NaN
        x = torch.nan_to_num(x, nan=0.0)

        # 兼容性处理：如果输入是 (N, T, VC) 格式，自动调整为 (N, C, T, V, M)
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)

        N, C, T, V, M = x.size()

        # 数据预处理与归一化 (Batch Norm)
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # 记录原始时间长度，用于回归头插值
        T_orig = T

        # 通过 TCN-GCN 单元
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)

        # [NEW] 应用 Physics Attention (Scheme B)
        # 增强非物理连接捕获能力
        x = self.phys_att(x)

        x = self.l9(x)
        z = self.l10(x)

        # [NEW] 计算关节回归 (Scheme B Step 3)
        # 输入 z: (N*M, C_out, T_down, V) -> 输出: (N*M, 3, T_orig, V)
        predicted_joints = self.regression_head(z, T_orig)

        # 全局平均池化 (Global Average Pooling)
        c_new = z.size(1)
        x_gap = z.view(N, M, c_new, -1)
        x_gap = x_gap.mean(3).mean(1)

        # Dropout
        features_before_drop = x_gap
        x_out = self.drop_out(x_gap)

        # 根据参数返回不同内容
        if return_features:
            return x_out, z  # 返回特征向量和特征图

        logits = self.fc(x_out)

        # [MODIFIED] 训练模式下返回分类结果和回归结果
        if self.training:
            return logits, predicted_joints
        else:
            return logits

    def get_hypergraph_loss(self):
        """
        [NEW] 收集所有层的超图损失
        Returns:
            avg_entropy, avg_proto_ortho, avg_phy_ortho
        """
        total_entropy = 0
        total_proto_ortho = 0
        total_phy_ortho = 0
        count = 0

        # 遍历所有定义的层
        layers = [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7, self.l8, self.l9, self.l10]

        for layer in layers:
            # 检查层中是否包含 hypergcn1 且包含 dhg 模块
            # 注意：unit_hypergcn 内部我们将模块命名为 dhg
            if hasattr(layer, 'hypergcn1') and hasattr(layer.hypergcn1, 'dhg'):
                module = layer.hypergcn1.dhg
                # 调用 PhysicallyGuidedDSAHypergraph.get_loss()
                if hasattr(module, 'get_loss'):
                    ent, proto, phy = module.get_loss()
                    total_entropy += ent
                    total_proto_ortho += proto
                    total_phy_ortho += phy
                    count += 1

        if count > 0:
            return total_entropy / count, total_proto_ortho / count, total_phy_ortho / count
        else:
            device = self.data_bn.weight.device
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    def get_hypergraph_l1_loss(self):
        # 保持接口兼容性，返回 0
        return torch.tensor(0.0, device=self.data_bn.weight.device)


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

        # Model.forward returns (feat, z) when return_features=True
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