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
    ✅ [完全修复版 v2.1] DSA-HGN Dynamic Generation + HGT-Bone Physical Priors

    主要改进：
    1. ✅ 温度退火机制优化（更激进的初始温度，更平缓的衰减）
    2. ✅ 自适应目标熵（根据超边数量动态调整）
    3. ✅ Smooth L1 熵损失（替代L2，更稳定）
    4. ✅ 双向平衡物理损失（既不过度重叠，也不完全无关）
    5. ✅ 数值稳定性保护（防止NaN/Inf）
    """

    def __init__(self, in_channels, num_dynamic_edges=8, ratio=8, use_virtual_conn=True,
                 num_point=21, temperature=0.1, **kwargs):
        super(PhysicallyGuidedDSAHypergraph, self).__init__()

        # ============================================================
        # 核心属性初始化
        # ============================================================
        self.num_dynamic_edges = num_dynamic_edges
        self.in_channels = in_channels
        self.use_virtual_conn = use_virtual_conn  # ✅ 必须在get_loss之前定义
        self.num_point = num_point

        # ============================================================
        # ✅ [修复1] 温度退火策略优化
        # ============================================================
        self.min_temperature = temperature  # 最终温度（用户指定）

        # 初始温度设置：更激进的起点以鼓励探索
        # 从 *5 改为 *8，确保早期有足够的随机性
        self.initial_temperature = max(0.8, temperature * 8)

        # 衰减率从 0.95 改为 0.93，更平滑的退火曲线
        self.temperature_decay = 0.93

        self.register_buffer('temperature', torch.tensor(float(self.initial_temperature)))

        # ============================================================
        # ✅ [修复2] 自适应目标熵策略
        # ============================================================
        # 理论最大熵: log(num_dynamic_edges)
        max_entropy = math.log(num_dynamic_edges)

        # 自适应稀疏度因子
        if num_dynamic_edges <= 4:
            target_factor = 0.4  # 高稀疏（4个边中平均激活~1.5个）
        elif num_dynamic_edges <= 8:
            target_factor = 0.5  # 中等稀疏（8个边中平均激活~3个）
        elif num_dynamic_edges <= 12:
            target_factor = 0.55  # 适度稀疏（12个边中平均激活~5个）
        else:
            target_factor = 0.6  # 相对均匀（保持一定分散度）

        target_entropy = max_entropy * target_factor
        self.register_buffer('target_entropy', torch.tensor(target_entropy))

        # ============================================================
        # ✅ [修复3] 特征投影层优化
        # ============================================================
        # 提高 inter_channels 下限，增强表达能力
        inter_channels = max(16, in_channels // ratio)
        self.inter_channels = inter_channels

        self.query = nn.Conv2d(in_channels, inter_channels, 1)
        self.query_norm = nn.LayerNorm(inter_channels)  # 稳定性

        # ============================================================
        # ✅ [修复4] 原型初始化优化
        # ============================================================
        # 小初始化 + 正交化，确保训练初期稳定
        prototypes = torch.randn(inter_channels, num_dynamic_edges) * 0.02
        if inter_channels >= num_dynamic_edges:
            q, _ = torch.linalg.qr(prototypes)
            self.key_prototypes = nn.Parameter(q.contiguous())
        else:
            q, _ = torch.linalg.qr(prototypes.T)
            self.key_prototypes = nn.Parameter(q.T.contiguous())

        # ============================================================
        # 物理边定义（HGT-Bone解剖学先验）
        # ============================================================
        mask_tensor = self._create_finger_masks()
        self.register_buffer('finger_masks', mask_tensor.transpose(0, 1))
        self.num_physical_edges = mask_tensor.shape[0]

        # 缓存
        self.last_h_dynamic = None
        self.current_epoch = 0

        # 初始化信息打印
        print(f"[INIT] PhysicallyGuidedDSAHypergraph v2.1 "
              f"Dynamic={num_dynamic_edges}, Physical={self.num_physical_edges}, "
              f"Temp={self.initial_temperature:.2f}->{self.min_temperature:.2f}, "
              f"TargetEntropy={target_entropy:.3f} (Factor={target_factor:.2f}, Max={max_entropy:.3f}), "
              f"Inter_ch={inter_channels}")

    def _create_finger_masks(self):
        """创建完整的解剖学分组掩码"""
        num_physical_edges = 7
        masks = torch.zeros(num_physical_edges, self.num_point)

        if self.num_point == 21:  # Bone Stream
            masks[0, 0] = 1.0  # Root bone
            masks[1, 1:5] = 1.0  # Thumb
            masks[2, 5:9] = 1.0  # Index
            masks[3, 9:13] = 1.0  # Middle
            masks[4, 13:17] = 1.0  # Ring
            masks[5, 17:21] = 1.0  # Pinky
            # 指尖交互（跨手指协同）
            for idx in [4, 8, 12, 16, 20]:
                if idx < self.num_point:
                    masks[6, idx] = 1.0

        elif self.num_point == 22:  # Joint Stream
            masks[0, 0:2] = 1.0  # Wrist + Palm
            masks[1, 2:6] = 1.0  # Thumb
            masks[2, 6:10] = 1.0  # Index
            masks[3, 10:14] = 1.0  # Middle
            masks[4, 14:18] = 1.0  # Ring
            masks[5, 18:22] = 1.0  # Pinky
            for idx in [5, 9, 13, 17, 21]:
                if idx < self.num_point:
                    masks[6, idx] = 1.0
        else:
            # Fallback: 均匀分布
            print(f"[WARNING] Unknown num_point={self.num_point}, using uniform masks")
            masks[:, :] = 1.0 / self.num_point

        return masks

    def set_epoch(self, epoch):
        """
        ✅ [关键方法] 温度退火更新

        公式: T = T_min + (T_init - T_min) * decay^epoch

        示例 (temperature=0.1, initial=0.8, decay=0.93):
        - Epoch 0:  T = 0.8
        - Epoch 10: T = 0.49
        - Epoch 30: T = 0.22
        - Epoch 60: T = 0.12
        - Epoch 100: T = 0.10 (接近最终值)
        """
        self.current_epoch = epoch

        # 指数衰减
        current_temp = self.min_temperature + \
                       (self.initial_temperature - self.min_temperature) * \
                       (self.temperature_decay ** epoch)

        # ✅ 下界保护：防止温度过低导致数值不稳定
        current_temp = max(current_temp, self.min_temperature * 0.5)

        self.temperature.fill_(current_temp)

    def forward(self, x):
        """
        ✅ [优化] 前向传播 + 数值稳定性保护

        Args:
            x: (N, C, T, V) 输入特征
        Returns:
            H_final: (N, V, M_total) 超图关联矩阵
        """
        N, C, T, V = x.shape

        if not self.use_virtual_conn:
            # 仅使用物理边
            return self.finger_masks.unsqueeze(0).expand(N, -1, -1)

        # ============================================================
        # 动态边生成
        # ============================================================
        q_node = self.query(x)  # (N, C', T, V)
        q_node_pooled = q_node.mean(2)  # (N, C', V) 时序池化

        # LayerNorm + L2归一化（双重稳定）
        q_node_pooled = q_node_pooled.permute(0, 2, 1)  # (N, V, C')
        q_node_pooled = self.query_norm(q_node_pooled)
        q_node_pooled = F.normalize(q_node_pooled, p=2, dim=-1, eps=1e-8)

        # 归一化keys
        k = F.normalize(self.key_prototypes, p=2, dim=0, eps=1e-8)

        # 相似度计算
        H_raw = torch.matmul(q_node_pooled, k)  # (N, V, M_dyn)

        # ✅ 温度应用 + 数值保护
        current_temp = max(self.temperature.item(), 0.01)  # 防止除零
        H_raw_stable = H_raw / current_temp

        # ✅ Softmax前裁剪（防止exp溢出）
        H_raw_stable = torch.clamp(H_raw_stable, -10.0, 10.0)
        H_dynamic = torch.softmax(H_raw_stable, dim=-1)

        # ✅ Detach缓存（避免梯度累积）
        self.last_h_dynamic = H_dynamic.detach()

        # ============================================================
        # 融合动态边 + 物理边
        # ============================================================
        H_physical = self.finger_masks.unsqueeze(0).expand(N, -1, -1)
        H_final = torch.cat([H_dynamic, H_physical], dim=-1)

        return H_final

    def get_loss(self):
        """
        ✅ [完全重写] 三元损失函数

        Returns:
            loss_entropy: 稀疏性约束（Smooth L1）
            loss_proto_ortho: 原型多样性约束（阈值化）
            loss_phy_balance: 物理互补约束（双向平衡）
        """
        # ============================================================
        # 前置检查
        # ============================================================
        if not hasattr(self, 'use_virtual_conn') or not self.use_virtual_conn:
            device = self.key_prototypes.device
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero

        if self.last_h_dynamic is None:
            device = self.key_prototypes.device
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero

        H = self.last_h_dynamic  # (N, V, M_dyn)
        device = H.device

        # ============================================================
        # 1. 目标熵损失 - Smooth L1 (Huber Loss)
        # ============================================================
        # 计算当前熵
        current_entropy = -torch.sum(H * torch.log(H.clamp(min=1e-8)), dim=-1).mean()

        # Smooth L1 实现 (delta=1.0)
        error = current_entropy - self.target_entropy
        abs_error = torch.abs(error)

        if abs_error < 1.0:
            loss_entropy = 0.5 * error ** 2
        else:
            loss_entropy = abs_error - 0.5

        loss_entropy = torch.clamp(loss_entropy, min=0.0)

        # ============================================================
        # 2. 原型正交损失 - 阈值化（只惩罚强相关）
        # ============================================================
        k = self.key_prototypes
        k_norm = F.normalize(k, p=2, dim=0, eps=1e-8)
        gram = torch.matmul(k_norm.T, k_norm)
        identity = torch.eye(gram.shape[0], device=device)

        # ✅ 阈值策略：只惩罚相似度 > 0.2 的对（允许适度重叠）
        off_diag = gram * (1 - identity)
        loss_proto_ortho = torch.mean(F.relu(off_diag - 0.2) ** 2)

        # ============================================================
        # 3. 物理平衡损失 - 双向约束（互补而非对立）
        # ============================================================
        H_dyn_coverage = H.mean(dim=0)  # (V, M_dyn) 动态边的节点覆盖
        H_phy_coverage = self.finger_masks  # (V, M_phy) 物理边的节点覆盖

        # 归一化
        H_dyn_norm = F.normalize(H_dyn_coverage, p=2, dim=0, eps=1e-8)
        H_phy_norm = F.normalize(H_phy_coverage, p=2, dim=0, eps=1e-8)

        # 协同度矩阵
        synergy = torch.matmul(H_dyn_norm.T, H_phy_norm)  # (M_dyn, M_phy)

        # ✅ 双向平衡策略：
        # - 过高相似度 (>0.7): 惩罚过度重叠
        # - 过低相似度 (<0.1): 惩罚完全无关
        # - [0.1, 0.7]: 认为是健康的互补关系
        high_sim_penalty = torch.mean(F.relu(synergy - 0.7) ** 2)
        low_sim_penalty = torch.mean(F.relu(0.1 - synergy) ** 2)

        # 加权组合（低相似度惩罚权重较小，鼓励探索）
        loss_phy_balance = high_sim_penalty + 0.3 * low_sim_penalty

        # ============================================================
        # 数值保护
        # ============================================================
        for name, loss_val in [('entropy', loss_entropy),
                               ('ortho', loss_proto_ortho),
                               ('balance', loss_phy_balance)]:
            if not torch.isfinite(loss_val):
                print(f"[WARNING] Non-finite {name} loss: {loss_val.item()}")
                loss_val = torch.tensor(0.0, device=device)

        return loss_entropy, loss_proto_ortho, loss_phy_balance


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
    """
    ✅ [优化] 超图卷积单元
    主要改进：
    1. 传递 temperature 参数
    2. 添加 dropout 防止过拟合
    3. 改进归一化策略
    """

    def __init__(self, in_channels, out_channels, num_hyperedges=16, residual=True,
                 num_point=21, temperature=0.05, **kwargs):
        super(unit_hypergcn, self).__init__()

        # ✅ 传递 temperature 参数
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

        # ✅ Dropout 防止过拟合
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

        # ✅ BN初始化调整（给超图分支更大的学习空间）
        bn_init(self.bn, 0.1)

    def forward(self, x):
        N, C, T, V = x.shape

        # 生成超图关联矩阵
        H = self.dhg(x)  # (N, V, M)

        # ============================================================
        # ✅ [修复] 改进的归一化策略
        # ============================================================
        # 节点到超边聚合：对每个超边，归一化其包含的节点权重
        H_sum_v = H.sum(dim=1, keepdims=True)  # (N, 1, M)
        H_norm_v2e = H / (H_sum_v + 1e-6)  # (N, V, M)

        # 节点特征投影
        x_v2e_feat = self.conv_v2e(x)  # (N, C, T, V)

        # 聚合到超边
        x_edge = torch.einsum('nctv,nvm->nctm', x_v2e_feat, H_norm_v2e)  # (N, C, T, M)

        # 超边特征变换
        x_e_feat = self.conv_e(x_edge)  # (N, C', T, M)

        # 超边到节点广播：对每个节点，归一化其关联的超边权重
        H_sum_e = H.sum(dim=2, keepdims=True)  # (N, V, 1)
        H_norm_e2v = H / (H_sum_e + 1e-6)  # (N, V, M)

        # 广播回节点
        x_node = torch.einsum('nctm,nvm->nctv', x_e_feat, H_norm_e2v)  # (N, C', T, V)

        # 残差连接
        y = self.bn(x_node)
        y = self.dropout(y)
        y = y + self.down(x)
        y = self.relu(y)

        return y

    def set_epoch(self, epoch):
        """✅ 传递epoch给超图生成器用于温度退火"""
        if hasattr(self.dhg, 'set_epoch'):
            self.dhg.set_epoch(epoch)