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
    ✅ 修复版v2: 添加temperature参数 + 修复inter_channels
    """

    def __init__(self, in_channels, num_dynamic_edges=8, ratio=8, use_virtual_conn=True,
                 num_point=21, temperature=0.05, **kwargs):
        super(PhysicallyGuidedDSAHypergraph, self).__init__()

        self.num_dynamic_edges = num_dynamic_edges
        self.in_channels = in_channels
        self.use_virtual_conn = use_virtual_conn
        self.num_point = num_point

        # ✅ [修复1] 保存temperature参数
        self.temperature = temperature

        # --- [Part 1: Dynamic Branch (DSA-HGN Logic)] ---
        # ✅ [修复2] 确保inter_channels有足够的表达能力
        # 最小值从1提升到8，避免浅层表达能力过弱
        inter_channels = max(8, in_channels // ratio)
        self.inter_channels = inter_channels

        # 1. Feature Projection (Query)
        self.query = nn.Conv2d(in_channels, inter_channels, 1)

        # 2. Prototypes (Keys) for Dynamic Edges
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

        print(f"[INIT] PhysicallyGuidedDSAHypergraph. "
              f"Dynamic Edges={num_dynamic_edges}, Physical Edges={self.num_physical_edges}, "
              f"Temperature={temperature}, Inter_channels={inter_channels}")

    def _create_finger_masks(self):
        """
        创建完整的解剖学分组掩码
        """
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
        q_node = self.query(x)
        q_node_pooled = q_node.mean(2)
        q_node_pooled = F.normalize(q_node_pooled, p=2, dim=1)
        q_node_pooled = q_node_pooled.permute(0, 2, 1)  # (N, V, C')

        k = self.key_prototypes

        # ✅ [修复3] 使用temperature控制softmax尖锐度
        # temperature越小，分布越尖锐（更稀疏）
        H_raw = torch.matmul(q_node_pooled, k)
        H_dynamic = torch.softmax(H_raw / self.temperature, dim=-1)  # (N, V, M_dyn)

        self.last_h_dynamic = H_dynamic

        # --- [2. Retrieve Physical Incidence Matrix H_physical] ---
        H_physical = self.finger_masks.unsqueeze(0).expand(N, -1, -1)

        # --- [3. Hybrid Fusion] ---
        H_final = torch.cat([H_dynamic, H_physical], dim=-1)

        return H_final

    def get_loss(self):
        """
        Returns: entropy, proto_ortho, phy_ortho
        """
        if self.last_h_dynamic is None or not self.use_virtual_conn:
            device = self.key_prototypes.device
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero

        H = self.last_h_dynamic  # (N, V, M_dyn)

        # 1. Entropy Loss
        current_entropy = -torch.sum(H * torch.log(H + 1e-6), dim=-1).mean()
        loss_entropy = current_entropy

        # 2. Prototype Orthogonality Loss
        k = self.key_prototypes
        k_norm = F.normalize(k, p=2, dim=0)
        gram = torch.matmul(k_norm.T, k_norm)
        identity = torch.eye(gram.shape[0], device=gram.device)
        loss_proto_ortho = torch.mean((gram * (1 - identity)) ** 2)

        # 3. Physical Constraint Loss
        H_dyn_T = self.last_h_dynamic.permute(0, 2, 1)
        H_phy = self.finger_masks.unsqueeze(0)

        H_dyn_norm = F.normalize(H_dyn_T, p=2, dim=-1)
        H_phy_norm = F.normalize(H_phy.transpose(1, 2), p=2, dim=-1)

        similarity = torch.matmul(H_dyn_norm, H_phy_norm.transpose(1, 2))
        loss_phy_ortho = torch.mean(similarity ** 2)

        return loss_entropy, loss_proto_ortho, loss_phy_ortho


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

        bn_init(self.bn, 1e-5)

    def forward(self, x):
        N, C, T, V = x.shape

        H = self.dhg(x)

        H_norm_v = H / (H.sum(dim=1, keepdim=True) + 1e-5)

        x_v2e_feat = self.conv_v2e(x)
        x_edge = torch.einsum('nctv,nve->ncte', x_v2e_feat, H_norm_v)

        x_e_feat = self.conv_e(x_edge)

        H_norm_e = H / (H.sum(dim=2, keepdim=True) + 1e-5)
        x_node = torch.einsum('ncte,nev->nctv', x_e_feat, H_norm_e.transpose(1, 2))

        y = self.bn(x_node)
        y = y + self.down(x)
        y = self.relu(y)
        return y