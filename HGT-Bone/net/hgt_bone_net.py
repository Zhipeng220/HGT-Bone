import math
import torch
import torch.nn as nn
from .basic_modules import unit_tcn, unit_gcn, bn_init, MultiScale_TemporalConv, import_class
from .bone_hypergraph import BoneHypergraphModule


class HGTBlock(nn.Module):
    """
    HGT-Bone 的基础构建单元 (HGT Block)

    组成:
    1. Line Graph GCN: 处理骨骼间的物理连接 (显式拓扑)
    2. Bone Hypergraph: 处理手指内和手指间的协同 (高阶拓扑)
    3. Adaptive Fusion Gate: 动态融合上述两种特征
    4. MS-TCN: 多尺度时序卷积
    """

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5,
                 dilations=[1, 2]):
        super(HGTBlock, self).__init__()

        # 分支 1: 线图卷积 (基于 A_line)
        self.gcn = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)

        # 分支 2: 物理引导超图卷积 (基于 Fingers Priors)
        # HGT-Bone 专用模块，利用 21 骨骼节点的解剖学分组
        self.hyper = BoneHypergraphModule(in_channels, out_channels)

        # 融合门控 (Gate)
        # 学习一个权重 alpha (0~1)，决定每帧更依赖 GCN 还是 Hypergraph
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

        # 时序建模 (Multi-Scale TCN)
        self.tcn = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           dilations=dilations, residual=False)

        self.relu = nn.ReLU(inplace=True)

        # 残差连接
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        # x: (N, C, T, V=21)

        # 1. 双流特征提取
        x_gcn = self.gcn(x)  # 物理连接特征
        x_hyp = self.hyper(x)  # 协同特征 (含物理约束)

        # 2. 动态门控融合
        alpha = self.gate(x)
        x_fused = alpha * x_gcn + (1 - alpha) * x_hyp

        # 3. 时序卷积 + 残差
        x_tcn = self.tcn(x_fused)
        x_out = self.relu(x_tcn + self.residual(x))

        return x_out


class HGTBoneNet(nn.Module):
    """
    HGT-Bone: High-Order Geometric-Topology Bone Network

    Paper 1 (独立骨骼流) 的核心骨干网络。
    特点:
    - 输入: 8通道几何特征 (Vector, Len, Angle, Axis)
    - 节点: 21 (Line Graph Nodes)
    - 约束: 物理正交性损失 (Physical Orthogonality Loss)
    """

    def __init__(self, num_class=14, num_point=21, num_person=1, graph=None, graph_args=dict(),
                 in_channels=8, drop_out=0, adaptive=True, base_channels=64, **kwargs):
        super(HGTBoneNet, self).__init__()

        if graph is None:
            raise ValueError("Graph is required for HGTBoneNet")
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.num_class = num_class
        self.num_point = num_point  # 必须是 21

        # 输入 BN 层 (针对 8D 特征)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # 网络层定义 (10层架构，类似 ResNet/ST-GCN)
        self.layers = nn.ModuleList()

        # L1: Input -> 64
        self.layers.append(HGTBlock(in_channels, base_channels, A, residual=False, adaptive=adaptive))
        # L2-L4: 64 -> 64
        self.layers.append(HGTBlock(base_channels, base_channels, A, adaptive=adaptive))
        self.layers.append(HGTBlock(base_channels, base_channels, A, adaptive=adaptive))
        self.layers.append(HGTBlock(base_channels, base_channels, A, adaptive=adaptive))

        # L5: 64 -> 128 (Stride=2, 下采样)
        self.layers.append(HGTBlock(base_channels, base_channels * 2, A, stride=2, adaptive=adaptive))
        # L6-L7: 128 -> 128
        self.layers.append(HGTBlock(base_channels * 2, base_channels * 2, A, adaptive=adaptive))
        self.layers.append(HGTBlock(base_channels * 2, base_channels * 2, A, adaptive=adaptive))

        # L8: 128 -> 256 (Stride=2, 下采样)
        self.layers.append(HGTBlock(base_channels * 2, base_channels * 4, A, stride=2, adaptive=adaptive))
        # L9-L10: 256 -> 256
        self.layers.append(HGTBlock(base_channels * 4, base_channels * 4, A, adaptive=adaptive))
        self.layers.append(HGTBlock(base_channels * 4, base_channels * 4, A, adaptive=adaptive))

        # 分类头
        self.fc = nn.Linear(base_channels * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x, return_features=False):
        # x: (N, C=8, T, V=21, M)

        # 1. 预处理
        x = torch.nan_to_num(x, nan=0.0)
        N, C, T, V, M = x.size()

        # BN处理: (N, M, V, C, T) -> (N, M*V*C, T) -> BN -> restore
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # 2. 逐层前向传播
        for layer in self.layers:
            x = layer(x)

        # 3. 全局池化
        # x: (N*M, 256, T_down, V_down)
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)  # Average over Time and Nodes, then Person

        # 4. Dropout & Classification
        feature = self.drop_out(x)
        out = self.fc(feature)

        if return_features:
            return out, feature
        else:
            return out

    def get_orthogonality_loss(self):
        """
        获取物理约束损失 (用于 Processor 中的反向传播)
        累加所有层中 BoneHypergraphModule 的损失
        """
        total_loss = 0.0
        count = 0
        for layer in self.layers:
            if hasattr(layer, 'hyper') and hasattr(layer.hyper, 'get_physical_constraint_loss'):
                total_loss += layer.hyper.get_physical_constraint_loss()
                count += 1

        return total_loss / (count + 1e-6)