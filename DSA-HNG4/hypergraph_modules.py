"""
hypergraph_modules.py - Final Optimized Hypergraph Modules v2.2
============================================================
综合Claude+Grok二轮审核

关键调整:
1. bn_init从0.15调到0.2 (Grok建议)
2. diversity_penalty阈值从0.1调到0.15
3. 添加更详细的损失分解日志
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from .basic_modules import bn_init
except ImportError:
    def bn_init(bn, scale):
        nn.init.constant_(bn.weight, scale)
        nn.init.constant_(bn.bias, 0)


class DifferentiableSparseHypergraph(nn.Module):
    """原始超图生成器 (向后兼容)"""

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
    ✅ [最终版 v2.2] DSA-HGN 动态超图生成器
    
    综合所有反馈的最终优化版本
    """

    def __init__(self, in_channels, num_dynamic_edges=8, ratio=8, use_virtual_conn=True,
                 num_point=21, temperature=0.1, **kwargs):
        super(PhysicallyGuidedDSAHypergraph, self).__init__()

        self.num_dynamic_edges = num_dynamic_edges
        self.in_channels = in_channels
        self.use_virtual_conn = use_virtual_conn
        self.num_point = num_point

        # 温度退火
        self.min_temperature = max(temperature, 0.12)
        self.initial_temperature = 0.5
        self.temperature_decay = 0.98

        self.register_buffer('temperature', torch.tensor(float(self.initial_temperature)))

        # 目标熵
        max_entropy = math.log(num_dynamic_edges)
        
        if num_dynamic_edges <= 4:
            target_factor = 0.35
        elif num_dynamic_edges <= 8:
            target_factor = 0.40
        elif num_dynamic_edges <= 12:
            target_factor = 0.42
        elif num_dynamic_edges <= 16:
            target_factor = 0.45
        else:
            target_factor = 0.48
        
        target_entropy = max_entropy * target_factor
        self.register_buffer('target_entropy', torch.tensor(target_entropy))

        # 特征投影
        inter_channels = max(16, in_channels // ratio)
        self.inter_channels = inter_channels

        self.query = nn.Conv2d(in_channels, inter_channels, 1)
        self.query_norm = nn.LayerNorm(inter_channels)

        # 原型初始化
        prototypes = torch.randn(inter_channels, num_dynamic_edges) * 0.02
        if inter_channels >= num_dynamic_edges:
            q, _ = torch.linalg.qr(prototypes)
            self.key_prototypes = nn.Parameter(q.contiguous())
        else:
            q, _ = torch.linalg.qr(prototypes.T)
            self.key_prototypes = nn.Parameter(q.T.contiguous())

        # 物理边
        mask_tensor = self._create_finger_masks()
        self.register_buffer('finger_masks', mask_tensor)  # (7, 21)
        self.num_physical_edges = mask_tensor.shape[0]

        # 缓存
        self.last_h_dynamic = None
        self.current_epoch = 0
        
        # ✅ 损失分解缓存 (用于详细日志)
        self.loss_components = {}

        print(f"[INIT] PhysicallyGuidedDSAHypergraph v2.2")
        print(f"       Dynamic={num_dynamic_edges}, Physical={self.num_physical_edges}")
        print(f"       Temp: {self.initial_temperature:.2f} -> {self.min_temperature:.2f}")
        print(f"       TargetEntropy={target_entropy:.3f}")

    def _create_finger_masks(self):
        """创建解剖学分组掩码"""
        num_physical_edges = 7
        masks = torch.zeros(num_physical_edges, self.num_point)

        if self.num_point == 21:
            masks[0, 0] = 1.0
            masks[1, 1:5] = 1.0
            masks[2, 5:9] = 1.0
            masks[3, 9:13] = 1.0
            masks[4, 13:17] = 1.0
            masks[5, 17:21] = 1.0
            for idx in [4, 8, 12, 16, 20]:
                if idx < self.num_point:
                    masks[6, idx] = 1.0
        elif self.num_point == 22:
            masks[0, 0:2] = 1.0
            masks[1, 2:6] = 1.0
            masks[2, 6:10] = 1.0
            masks[3, 10:14] = 1.0
            masks[4, 14:18] = 1.0
            masks[5, 18:22] = 1.0
            for idx in [5, 9, 13, 17, 21]:
                if idx < self.num_point:
                    masks[6, idx] = 1.0
        else:
            masks[:, :] = 1.0 / self.num_point

        return masks

    def set_epoch(self, epoch):
        """温度退火更新"""
        self.current_epoch = epoch

        current_temp = self.min_temperature + \
                       (self.initial_temperature - self.min_temperature) * \
                       (self.temperature_decay ** epoch)

        current_temp = max(current_temp, self.min_temperature * 0.8)
        self.temperature.fill_(current_temp)

    def forward(self, x):
        """前向传播"""
        N, C, T, V = x.shape

        if not self.use_virtual_conn:
            return self.finger_masks.unsqueeze(0).expand(N, -1, -1)

        q_node = self.query(x)
        q_node_pooled = q_node.mean(2)
        q_node_pooled = q_node_pooled.permute(0, 2, 1)
        q_node_pooled = self.query_norm(q_node_pooled)
        q_node_pooled = F.normalize(q_node_pooled, p=2, dim=-1, eps=1e-8)

        k = F.normalize(self.key_prototypes, p=2, dim=0, eps=1e-8)
        H_raw = torch.matmul(q_node_pooled, k)

        current_temp = max(self.temperature.item(), 0.01)
        H_raw_stable = H_raw / current_temp
        H_raw_stable = torch.clamp(H_raw_stable, -10.0, 10.0)
        H_dynamic = torch.softmax(H_raw_stable, dim=-1)

        self.last_h_dynamic = H_dynamic


        H_physical = self.finger_masks.T.unsqueeze(0).expand(N, -1, -1)
        H_final = torch.cat([H_dynamic, H_physical], dim=-1)

        return H_final

    def get_loss(self):
        """
        ✅ [修复版本 v3] 三元损失函数
        修复了矩阵维度不匹配的问题
        """
        if not hasattr(self, 'use_virtual_conn') or not self.use_virtual_conn:
            device = self.key_prototypes.device
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero

        if self.last_h_dynamic is None:
            device = self.key_prototypes.device
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero

        H = self.last_h_dynamic
        device = H.device

        # 1. 熵损失 - 双向L2
        current_entropy = -torch.sum(H * torch.log(H.clamp(min=1e-8)), dim=-1).mean()
        entropy_diff = current_entropy - self.target_entropy
        loss_entropy = entropy_diff ** 2

        # 缓存当前熵用于日志
        self.loss_components['current_entropy'] = current_entropy.item()

        # 2. 原型正交损失
        k = self.key_prototypes
        k_norm = F.normalize(k, p=2, dim=0, eps=1e-8)
        gram = torch.matmul(k_norm.T, k_norm)
        identity = torch.eye(gram.shape[0], device=device)
        off_diag = gram * (1 - identity)

        # L2惩罚
        loss_proto_ortho = 0.5 * torch.mean(off_diag ** 2)

        # 多样性惩罚
        diversity_penalty = torch.mean(F.relu(off_diag - 0.15))
        loss_proto_ortho = loss_proto_ortho + 0.3 * diversity_penalty

        # 缓存
        self.loss_components['max_off_diag'] = off_diag.max().item()

        # ============================================================
        # 3. 物理平衡损失 - ✅ 修复版本 v3
        # ============================================================
        # H: (N, V, num_dynamic) where num_dynamic=16, V=21
        # finger_masks: (num_physical, V) where num_physical=7, V=21

        # 计算每个动态超边覆盖节点的平均权重
        H_dyn_mean = H.mean(dim=0)  # (V, num_dynamic) = (21, 16)

        # 归一化:每个超边的节点权重和为1
        H_dyn_sum = H_dyn_mean.sum(dim=0, keepdim=True) + 1e-8  # (1, 16)
        H_dyn_normalized = H_dyn_mean / H_dyn_sum  # (21, 16)

        # ✅ 关键修复:必须转置 finger_masks
        # finger_masks 原始形状: (7, 21)
        # 转置后: (21, 7)
        H_phy = self.finger_masks.T  # (21, 7) ← 这里是关键!

        # 归一化物理超边
        H_phy_sum = H_phy.sum(dim=0, keepdim=True) + 1e-8  # (1, 7)
        H_phy_normalized = H_phy / H_phy_sum  # (21, 7)

        # ✅ 现在维度匹配:
        # H_dyn_normalized.T: (16, 21)
        # H_phy_normalized:   (21, 7)
        # 结果: (16, 7)
        synergy = torch.matmul(H_dyn_normalized.T, H_phy_normalized)

        # 取绝对值
        synergy_abs = torch.abs(synergy)

        # 降低阈值,从0.6降到0.3
        loss_phy_balance = torch.mean(F.relu(synergy_abs - 0.3) ** 2)

        # 添加覆盖度惩罚
        max_synergy_per_dyn = synergy_abs.max(dim=1)[0]  # (16,)
        coverage_penalty = torch.mean(F.relu(0.1 - max_synergy_per_dyn))

        loss_phy_balance = loss_phy_balance + 0.5 * coverage_penalty

        # 缓存用于调试
        self.loss_components['synergy_mean'] = synergy_abs.mean().item()
        self.loss_components['synergy_max'] = synergy_abs.max().item()

        # 数值保护
        for name, loss_val in [('entropy', loss_entropy),
                               ('ortho', loss_proto_ortho),
                               ('balance', loss_phy_balance)]:
            if not torch.isfinite(loss_val):
                if name == 'entropy':
                    loss_entropy = torch.tensor(0.0, device=device)
                elif name == 'ortho':
                    loss_proto_ortho = torch.tensor(0.0, device=device)
                else:
                    loss_phy_balance = torch.tensor(0.0, device=device)

        return loss_entropy, loss_proto_ortho, loss_phy_balance



class PhysicsAttention(nn.Module):
    """Physics-Guided Attention Block"""

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
    ✅ [最终版 v2.2] 超图卷积单元
    """

    def __init__(self, in_channels, out_channels, num_hyperedges=16, residual=True,
                 num_point=21, temperature=0.1, **kwargs):
        super(unit_hypergcn, self).__init__()

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

        self.dropout = nn.Dropout(p=0.12)

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

        # ✅ Grok建议: bn_init从0.15调到0.2
        bn_init(self.bn, 0.2)

    def forward(self, x):
        N, C, T, V = x.shape

        H = self.dhg(x)

        H_sum_v = H.sum(dim=1, keepdims=True)
        H_norm_v2e = H / (H_sum_v + 1e-6)

        x_v2e_feat = self.conv_v2e(x)
        x_edge = torch.einsum('nctv,nvm->nctm', x_v2e_feat, H_norm_v2e)

        x_e_feat = self.conv_e(x_edge)

        H_sum_e = H.sum(dim=2, keepdims=True)
        H_norm_e2v = H / (H_sum_e + 1e-6)
        x_node = torch.einsum('nctm,nvm->nctv', x_e_feat, H_norm_e2v)

        y = self.bn(x_node)
        y = self.dropout(y)
        y = y + self.down(x)
        y = self.relu(y)

        return y

    def set_epoch(self, epoch):
        if hasattr(self.dhg, 'set_epoch'):
            self.dhg.set_epoch(epoch)
