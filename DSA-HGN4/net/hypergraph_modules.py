import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .basic_modules import bn_init


class DifferentiableSparseHypergraph(nn.Module):
    """
    [Original] Entropy-Regularized Softmax Hypergraph Generator
    (Kept for backward compatibility)
    """

    def __init__(self, in_channels, num_hyperedges, ratio=8, use_virtual_conn=True, **kwargs):
        super(DifferentiableSparseHypergraph, self).__init__()
        self.num_hyperedges = num_hyperedges
        self.in_channels = in_channels
        self.use_virtual_conn = use_virtual_conn

        inter_channels = max(1, in_channels // ratio)
        self.inter_channels = inter_channels

        self.query = nn.Conv2d(in_channels, inter_channels, 1)

        prototypes = torch.randn(inter_channels, num_hyperedges)
        if inter_channels >= num_hyperedges:
            q, _ = torch.linalg.qr(prototypes)
            self.key_prototypes = nn.Parameter(q.contiguous())
        else:
            q, _ = torch.linalg.qr(prototypes.T)
            self.key_prototypes = nn.Parameter(q.T.contiguous())

        self.last_h = None

    def forward(self, x):
        if not self.use_virtual_conn:
            N, C, T, V = x.shape
            return torch.zeros(N, V, self.num_hyperedges, device=x.device)

        N, C, T, V = x.shape
        q_node = self.query(x)
        q_node_pooled = q_node.mean(2)
        q_node_pooled = F.normalize(q_node_pooled, p=2, dim=1)
        q_node_pooled = q_node_pooled.permute(0, 2, 1)

        k = self.key_prototypes
        scale = self.inter_channels ** -0.5
        H_raw = torch.matmul(q_node_pooled, k) * scale
        H_final = torch.softmax(H_raw, dim=-1)
        self.last_h = H_final

        return H_final

    def get_loss(self):
        if not self.use_virtual_conn or self.last_h is None:
            dev = self.key_prototypes.device
            return torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev)

        H = self.last_h
        entropy = -torch.sum(H * torch.log(H + 1e-6), dim=-1).mean()

        k = self.key_prototypes
        k_norm = F.normalize(k, p=2, dim=0)
        gram = torch.matmul(k_norm.T, k_norm)
        identity = torch.eye(gram.shape[0], device=gram.device)
        loss_ortho = torch.mean((gram * (1 - identity)) ** 2)

        return entropy, loss_ortho


class PhysicallyGuidedDSAHypergraph(nn.Module):
    """
    [Fusion Module] DSA-HGN Dynamic Generation + HGT-Bone Physical Priors.
    ✅ 修复版v4: 正确的温度退火 + 目标熵 + 软物理约束 + 训练稳定性改进

    温度退火说明:
    - 配置文件中的 temperature 参数是最终温度（训练结束时的目标温度）
    - 初始温度自动设为 max(1.0, temperature * 5)
    - 每个epoch温度按 decay=0.95 衰减，逐渐接近最终温度
    """

    def __init__(self, in_channels, num_dynamic_edges=8, ratio=8, use_virtual_conn=True,
                 num_point=21, temperature=0.1, **kwargs):
        super(PhysicallyGuidedDSAHypergraph, self).__init__()

        self.num_dynamic_edges = num_dynamic_edges
        self.in_channels = in_channels
        self.use_virtual_conn = use_virtual_conn
        self.num_point = num_point

        # ✅ [修复1] 温度退火机制 - 正确设置初始和最终温度
        # 用户指定的temperature是最终温度（训练结束时）
        # 初始温度应该更高，逐渐退火到最终温度
        self.min_temperature = temperature  # 最终温度（用户指定）
        self.initial_temperature = max(1.0, temperature * 5)  # 初始温度：最终的5倍，至少1.0
        self.temperature_decay = 0.95  # 每个epoch衰减系数

        # 注册buffer，初始值设为initial_temperature（高温启动）
        self.register_buffer('temperature', torch.tensor(float(self.initial_temperature)))

        # ✅ [修复2] 目标熵 - 期望的稀疏程度
        # 对于 num_dynamic_edges 个超边，理想稀疏分布的熵约为 log(2~3)
        target_entropy = math.log(max(2, num_dynamic_edges // 4))
        self.register_buffer('target_entropy', torch.tensor(target_entropy))

        # ✅ [修复3] 增加inter_channels下限，提升表达能力
        inter_channels = max(16, in_channels // ratio)  # 从8提升到16
        self.inter_channels = inter_channels

        # 1. Feature Projection (Query) + LayerNorm for stability
        self.query = nn.Conv2d(in_channels, inter_channels, 1)
        self.query_norm = nn.LayerNorm(inter_channels)  # ✅ 新增稳定性

        # 2. Prototypes (Keys) for Dynamic Edges - 正交初始化
        prototypes = torch.randn(inter_channels, num_dynamic_edges)
        if inter_channels >= num_dynamic_edges:
            q, _ = torch.linalg.qr(prototypes)
            self.key_prototypes = nn.Parameter(q.contiguous())
        else:
            q, _ = torch.linalg.qr(prototypes.T)
            self.key_prototypes = nn.Parameter(q.T.contiguous())

        # --- [Part 2: Physical Branch (HGT-Bone Logic)] ---
        mask_tensor = self._create_finger_masks()
        self.register_buffer('finger_masks', mask_tensor.transpose(0, 1))
        self.num_physical_edges = mask_tensor.shape[0]

        # Cache for loss calculation
        self.last_h_dynamic = None
        self.current_epoch = 0  # 用于温度退火

        print(f"[INIT] PhysicallyGuidedDSAHypergraph v4. "
              f"Dynamic={num_dynamic_edges}, Physical={self.num_physical_edges}, "
              f"Temp={self.initial_temperature:.2f}->{self.min_temperature:.2f}, "
              f"TargetEntropy={target_entropy:.3f}, Inter_ch={inter_channels}")

    def _create_finger_masks(self):
        """创建完整的解剖学分组掩码"""
        num_physical_edges = 7
        masks = torch.zeros(num_physical_edges, self.num_point)

        if self.num_point == 21:  # Bone Stream
            masks[0, 0] = 1.0  # Root
            masks[1, 1:5] = 1.0  # Thumb
            masks[2, 5:9] = 1.0  # Index
            masks[3, 9:13] = 1.0  # Middle
            masks[4, 13:17] = 1.0  # Ring
            masks[5, 17:21] = 1.0  # Pinky
            # 指尖交互
            for idx in [4, 8, 12, 16, 20]:
                if idx < self.num_point:
                    masks[6, idx] = 1.0

        elif self.num_point == 22:  # Joint Stream
            masks[0, 0] = 1.0
            masks[0, 1] = 1.0
            masks[1, 2:6] = 1.0
            masks[2, 6:10] = 1.0
            masks[3, 10:14] = 1.0
            masks[4, 14:18] = 1.0
            masks[5, 18:22] = 1.0
            for idx in [5, 9, 13, 17, 21]:
                if idx < self.num_point:
                    masks[6, idx] = 1.0
        else:
            print(f"[WARNING] Unknown num_point={self.num_point}, using uniform masks")
            for i in range(num_physical_edges):
                masks[i, :] = 1.0 / self.num_point

        return masks

    def set_epoch(self, epoch):
        """
        ✅ [关键] 外部调用，用于温度退火
        温度公式: T = T_min + (T_init - T_min) * decay^epoch

        示例 (temperature=0.1):
        - Epoch 0:  T = 0.1 + (1.0 - 0.1) * 0.95^0  = 1.0
        - Epoch 10: T = 0.1 + (1.0 - 0.1) * 0.95^10 = 0.64
        - Epoch 30: T = 0.1 + (1.0 - 0.1) * 0.95^30 = 0.31
        - Epoch 60: T = 0.1 + (1.0 - 0.1) * 0.95^60 = 0.15
        - Epoch 100: T = 0.1 + (1.0 - 0.1) * 0.95^100 = 0.11
        """
        self.current_epoch = epoch
        current_temp = self.min_temperature + \
                       (self.initial_temperature - self.min_temperature) * \
                       (self.temperature_decay ** epoch)
        self.temperature.fill_(current_temp)

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V)
        Returns:
            H_final: (N, V, M_total)
        """
        N, C, T, V = x.shape

        if not self.use_virtual_conn:
            return self.finger_masks.unsqueeze(0).expand(N, -1, -1)

        # --- [1. Generate Dynamic Incidence Matrix H_dynamic] ---
        q_node = self.query(x)  # (N, C', T, V)
        q_node_pooled = q_node.mean(2)  # (N, C', V)

        # ✅ [修复5] LayerNorm稳定特征
        q_node_pooled = q_node_pooled.permute(0, 2, 1)  # (N, V, C')
        q_node_pooled = self.query_norm(q_node_pooled)
        q_node_pooled = F.normalize(q_node_pooled, p=2, dim=-1)

        # ✅ [修复6] 归一化keys
        k = F.normalize(self.key_prototypes, p=2, dim=0)

        # ✅ [修复7] 使用当前温度，而非固定值
        H_raw = torch.matmul(q_node_pooled, k)
        current_temp = self.temperature.item()
        # 添加数值稳定性
        H_raw_stable = H_raw / max(current_temp, 0.01)
        H_dynamic = torch.softmax(H_raw_stable, dim=-1)  # (N, V, M_dyn)

        self.last_h_dynamic = H_dynamic

        # --- [2. Retrieve Physical Incidence Matrix H_physical] ---
        H_physical = self.finger_masks.unsqueeze(0).expand(N, -1, -1)

        # --- [3. Hybrid Fusion] ---
        H_final = torch.cat([H_dynamic, H_physical], dim=-1)

        return H_final

    def get_loss(self):
        """
        Returns: entropy_loss, proto_ortho_loss, phy_alignment_loss
        ✅ 重新设计损失函数
        """
        if self.last_h_dynamic is None or not self.use_virtual_conn:
            device = self.key_prototypes.device
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero

        H = self.last_h_dynamic  # (N, V, M_dyn)

        # ✅ [修复8] 目标熵损失 - 让熵接近目标值，而非无限制优化
        # 这样既保持适度稀疏，又不会过度尖锐
        current_entropy = -torch.sum(H * torch.log(H + 1e-6), dim=-1).mean()
        loss_entropy = (current_entropy - self.target_entropy) ** 2  # L2距离

        # 2. Prototype Orthogonality Loss (保持不变，但降低权重)
        k = self.key_prototypes
        k_norm = F.normalize(k, p=2, dim=0)
        gram = torch.matmul(k_norm.T, k_norm)
        identity = torch.eye(gram.shape[0], device=gram.device)
        loss_proto_ortho = torch.mean((gram * (1 - identity)) ** 2)

        # ✅ [修复9] 软物理约束 - 鼓励动态边与物理边互补而非完全正交
        # 计算动态边的节点覆盖率
        H_dyn_coverage = H.mean(dim=0)  # (V, M_dyn) - 每个节点在各动态超边的平均权重
        H_phy_coverage = self.finger_masks  # (V, M_phy)

        # 软约束：动态边应该在物理边覆盖弱的地方有更多关注
        H_dyn_norm = F.normalize(H_dyn_coverage, p=2, dim=0)
        H_phy_norm = F.normalize(H_phy_coverage, p=2, dim=0)

        # 只惩罚过度相似的部分（相似度>0.5）
        similarity = torch.matmul(H_dyn_norm.T, H_phy_norm)  # (M_dyn, M_phy)
        loss_phy_soft = torch.mean(F.relu(similarity - 0.5) ** 2)  # 软阈值

        return loss_entropy, loss_proto_ortho, loss_phy_soft


class PhysicsAttention(nn.Module):
    """
    Physics-Guided Attention Block (PGAB).
    """

    def __init__(self, in_channels, out_channels, num_joints, hop_matrix, max_hop=3, dropout=0.0):
        super(PhysicsAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_joints = num_joints
        self.max_hop = max_hop

        self.q_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.k_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.v_conv = nn.Conv2d(in_channels, out_channels, 1)

        self.bias_embedding = nn.Embedding(max_hop + 2, 1)

        with torch.no_grad():
            self.bias_embedding.weight.data.fill_(0.0)
            self.bias_embedding.weight.data[0] = 1.0
            if max_hop >= 1:
                self.bias_embedding.weight.data[1] = 0.5
            if max_hop >= 2:
                self.bias_embedding.weight.data[2] = 0.1

        if not isinstance(hop_matrix, torch.Tensor):
            hop_matrix = torch.tensor(hop_matrix, dtype=torch.long)
        self.register_buffer('hop_matrix', hop_matrix)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, C, T, V = x.shape

        q = self.q_conv(x).permute(0, 2, 3, 1).contiguous().view(-1, V, self.out_channels)
        k = self.k_conv(x).permute(0, 2, 3, 1).contiguous().view(-1, V, self.out_channels)
        v = self.v_conv(x).permute(0, 2, 3, 1).contiguous().view(-1, V, self.out_channels)

        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.out_channels)

        physics_bias = self.bias_embedding(self.hop_matrix).squeeze(-1)
        attn_score = attn_score + physics_bias.unsqueeze(0)

        attn = self.softmax(attn_score)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        output = output.view(N, T, V, self.out_channels).permute(0, 3, 1, 2).contiguous()

        return output


class unit_hypergcn(nn.Module):
    def __init__(self, in_channels, out_channels, num_hyperedges=16, residual=True,
                 num_point=21, temperature=0.05, **kwargs):
        super(unit_hypergcn, self).__init__()

        # ✅ 传递temperature参数
        self.dhg = PhysicallyGuidedDSAHypergraph(
            in_channels,
            num_dynamic_edges=num_hyperedges,
            num_point=num_point,
            temperature=temperature,
            **kwargs
        )

        self.conv_v2e = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_e = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

        # ✅ [修复10] 添加dropout防止过拟合
        self.dropout = nn.Dropout(p=0.1)

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        # ✅ [修复11] BN初始化改为0.1，让超图分支有更大的学习空间
        bn_init(self.bn, 0.1)  # 从1e-5改为0.1

    def forward(self, x):
        N, C, T, V = x.shape

        H = self.dhg(x)  # (N, V, M)

        # ✅ [修复12] 修正归一化维度和添加数值稳定性
        # H: (N, V, M) - V是节点数，M是超边数

        # 节点到超边聚合：对每个超边，归一化其包含的节点权重
        H_sum_v = H.sum(dim=1, keepdim=True)  # (N, 1, M)
        H_norm_v2e = H / (H_sum_v + 1e-6)  # (N, V, M)

        x_v2e_feat = self.conv_v2e(x)  # (N, C, T, V)
        x_edge = torch.einsum('nctv,nvm->nctm', x_v2e_feat, H_norm_v2e)  # (N, C, T, M)

        x_e_feat = self.conv_e(x_edge)  # (N, C', T, M)

        # 超边到节点广播：对每个节点，归一化其关联的超边权重
        H_sum_e = H.sum(dim=2, keepdim=True)  # (N, V, 1)
        H_norm_e2v = H / (H_sum_e + 1e-6)  # (N, V, M)
        x_node = torch.einsum('nctm,nvm->nctv', x_e_feat, H_norm_e2v)  # (N, C', T, V)

        y = self.bn(x_node)
        y = self.dropout(y)  # ✅ 添加dropout
        y = y + self.down(x)
        y = self.relu(y)
        return y

    def set_epoch(self, epoch):
        """✅ 传递epoch给超图生成器用于温度退火"""
        if hasattr(self.dhg, 'set_epoch'):
            self.dhg.set_epoch(epoch)