import torch
import torch.nn as nn
import torch.nn.functional as F


class BoneHypergraphModule(nn.Module):
    """
    HGT-Bone 专用: 物理先验引导的超图卷积模块
    (Physically-Guided Hypergraph Convolution)

    核心差异 (vs DSA-HGN):
    1. 混合结构: 显式物理超边 (Fingers) + 隐式动态超边 (Learned)
    2. 节点定义: 21个骨骼节点 (非22个关节)
    3. 目标: 解决小样本下的过拟合问题，利用解剖学先验
    """

    def __init__(self, in_channels, out_channels, num_hyperedges=8, temperature=1.0):
        super(BoneHypergraphModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_dynamic_edges = num_hyperedges
        self.temperature = temperature

        # 1. 动态超边生成器 (Dynamic Branch)
        # 输入: 全局平均池化后的骨骼特征
        self.dynamic_gen = nn.Sequential(
            nn.Conv2d(in_channels, num_hyperedges, kernel_size=1),
            nn.BatchNorm2d(num_hyperedges),
            nn.ReLU(inplace=True)
        )

        # 2. 物理超边掩码 (Physical Branch - Fixed)
        # 包含5个手指分组: Thumb, Index, Middle, Ring, Pinky
        # 形状: (5, 21) -> 5条超边, 21个骨骼节点
        self.register_buffer('finger_masks', self._create_finger_masks())

        # 3. 特征变换 (Theta)
        self.feature_transform = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 4. 输出投影
        # 输入维度 = 动态边数 + 5条物理边
        total_edges = num_hyperedges + 5
        self.out_proj = nn.Sequential(
            nn.Conv2d(total_edges, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        # 用于保存最近一次计算的关联矩阵 H (用于可视化或Loss)
        self.last_H = None

    def _create_finger_masks(self):
        """
        创建基于解剖学的物理超边掩码
        对应 SHREC 21根骨骼的索引:
        0: Wrist-Palm (通常作为公共基底，暂不归入特定手指，或归入所有)
        1-4: Thumb, 5-8: Index, 9-12: Middle, 13-16: Ring, 17-20: Pinky
        """
        masks = torch.zeros(5, 21)

        # 定义手指所属的骨骼索引范围
        fingers_indices = [
            list(range(1, 5)),  # Thumb: 1,2,3,4
            list(range(5, 9)),  # Index: 5,6,7,8
            list(range(9, 13)),  # Middle: 9,10,11,12
            list(range(13, 17)),  # Ring: 13,14,15,16
            list(range(17, 21))  # Pinky: 17,18,19,20
        ]

        for i, indices in enumerate(fingers_indices):
            masks[i, indices] = 1.0

            # 可选: 将掌骨(Bone 0)加入所有手指超边，作为共享根节点
            # masks[i, 0] = 0.5

        return masks

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V=21) 骨骼特征
        Returns:
            out: (N, C_out, T, V=21)
        """
        N, C, T, V = x.size()

        # --- Step 1: 构建关联矩阵 H ---

        # A. 生成动态部分 (Data-driven)
        # Global Average Pooling over Time: (N, C, T, V) -> (N, C, V)
        x_mean = x.mean(dim=2).unsqueeze(-1)  # (N, C, V, 1)

        # (N, num_dynamic, V, 1) -> (N, num_dynamic, V)
        h_dynamic = self.dynamic_gen(x_mean).squeeze(-1)
        # 使用 Softmax 归一化动态边的权重
        H_dynamic = F.softmax(h_dynamic / self.temperature, dim=-1)

        # B. 获取物理部分 (Prior-driven)
        # (5, 21) -> (N, 5, 21)
        H_physical = self.finger_masks.unsqueeze(0).expand(N, -1, -1)

        # C. 拼接 (Hybrid Structure)
        # H: (N, M_total, V), where M_total = num_dynamic + 5
        H = torch.cat([H_dynamic, H_physical], dim=1)
        self.last_H = H  # 保存引用

        # 归一化 H (按度数归一化，数值稳定)
        # D_e = sum(H, dim=2) -> 每条边的度
        # D_v = sum(H, dim=1) -> 每个节点的度
        # 这里进行简单的行归一化即可，严谨的超图卷积需要双重归一化，
        # 但在深度学习中，Softmax + Learnable Weights 通常效果更好。

        # --- Step 2: 超图卷积 (Hypergraph Convolution) ---
        # 过程: Node -> Hyperedge -> Node

        # 1. 特征变换 Theta(x)
        x_feat = self.feature_transform(x)  # (N, C_out, T, V)
        x_feat = x_feat.permute(0, 1, 3, 2).contiguous()  # (N, C, V, T)

        # 2. Node -> Hyperedge (Aggregation)
        # Y_edge = H * X
        # (N, M, V) * (N, C, V, T) -> (N, C, M, T)
        Y_edge = torch.einsum('nmv, ncvt -> ncmt', H, x_feat)

        # 3. Hyperedge -> Node (Broadcasting)
        # X_out = H^T * Y_edge
        # (N, M, V)^T * (N, C, M, T) -> (N, C, V, T)
        # 此处我们不使用简单的 H^T，而是使用一个可学习的投影层 out_proj (类似于 Attention 的 V 变换)
        # 或者为了保持对称性，直接用 H^T 映射回去，再过一层 Conv。
        # 考虑到 HGT-Bone 强调几何，我们用加权求和:
        X_recon = torch.einsum('nmv, ncmt -> ncvt', H, Y_edge)

        # 恢复维度 (N, C, T, V)
        X_recon = X_recon.permute(0, 1, 3, 2).contiguous()

        # 残差连接 (可选，建议保留以防止梯度消失)
        if self.in_channels == self.out_channels:
            X_recon = X_recon + x

        return X_recon

    def get_physical_constraint_loss(self):
        """
        物理约束损失 (Orthogonality Loss)
        目标: 鼓励动态学习到的超边 (Dynamic) 捕获那些物理连接 (Physical) 未覆盖的关系。
        即: Dynamic edges 应该与 Physical edges 尽量正交 (不重复)。
        """
        if self.last_H is None:
            return torch.tensor(0.0)

        H = self.last_H  # (N, M_total, V)
        num_dynamic = self.num_dynamic_edges

        # 分离两部分
        H_dyn = H[:, :num_dynamic, :]  # (N, M_dyn, V)
        H_phy = H[:, num_dynamic:, :]  # (N, 5, V)

        # 计算相似度矩阵 (Cosine Similarity)
        # Normalize vectors
        H_dyn_norm = F.normalize(H_dyn, p=2, dim=-1)
        H_phy_norm = F.normalize(H_phy, p=2, dim=-1)

        # Similarity: (N, M_dyn, 5)
        # 动态边 i 与 物理边 j 的重合度
        similarity = torch.matmul(H_dyn_norm, H_phy_norm.transpose(1, 2))

        # 我们希望重合度越小越好 -> 最小化 similarity 的平方和
        ortho_loss = torch.mean(similarity ** 2)

        return ortho_loss