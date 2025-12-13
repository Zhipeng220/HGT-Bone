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
    Generates a hybrid incidence matrix H_final = [H_dynamic, H_physical].
    """

    def __init__(self, in_channels, num_dynamic_edges=8, ratio=8, use_virtual_conn=True, num_point=21, **kwargs):
        super(PhysicallyGuidedDSAHypergraph, self).__init__()

        self.num_dynamic_edges = num_dynamic_edges
        self.in_channels = in_channels
        self.use_virtual_conn = use_virtual_conn
        self.num_point = num_point

        # --- [Part 1: Dynamic Branch (DSA-HGN Logic)] ---
        inter_channels = max(1, in_channels // ratio)
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
        # Generate anatomical masks (Thumb to Pinky)
        mask_tensor = self._create_finger_masks()

        # [Critical] Register as buffer to ensure device consistency
        # Shape becomes (V, 5)
        self.register_buffer('finger_masks', mask_tensor.transpose(0, 1))

        # Cache for loss calculation
        self.last_h_dynamic = None

        print(f"[INIT] PhysicallyGuidedDSAHypergraph. Dynamic Edges={num_dynamic_edges}, Physical Edges=5")

    def _create_finger_masks(self):
        """
        Creates masks based on HGT-Bone anatomical grouping.
        """
        masks = torch.zeros(5, self.num_point)

        # Fingers indices (1-based in standard description, 0 is wrist)
        # Assuming standard SHREC/NTU layout where:
        # Thumb: 1-4, Index: 5-8, Middle: 9-12, Ring: 13-16, Pinky: 17-20
        fingers_indices = [
            list(range(1, 5)),  # Thumb
            list(range(5, 9)),  # Index
            list(range(9, 13)),  # Middle
            list(range(13, 17)),  # Ring
            list(range(17, 21))  # Pinky
        ]

        for i, indices in enumerate(fingers_indices):
            # Filter indices to ensure they are within num_point bounds
            valid_indices = [idx for idx in indices if idx < self.num_point]
            if valid_indices:
                masks[i, valid_indices] = 1.0

        return masks

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V)
        Returns:
            H_final: (N, V, M_total) where M_total = num_dynamic_edges + 5
        """
        N, C, T, V = x.shape

        if not self.use_virtual_conn:
            # Fallback: Only return physical connections if virtual is disabled
            return self.finger_masks.unsqueeze(0).expand(N, -1, -1)

        # --- [1. Generate Dynamic Incidence Matrix H_dynamic] ---
        q_node = self.query(x)
        q_node_pooled = q_node.mean(2)
        q_node_pooled = F.normalize(q_node_pooled, p=2, dim=1)
        q_node_pooled = q_node_pooled.permute(0, 2, 1)  # (N, V, C')

        k = self.key_prototypes
        scale = self.inter_channels ** -0.5
        H_raw = torch.matmul(q_node_pooled, k) * scale

        H_dynamic = torch.softmax(H_raw, dim=-1)  # (N, V, M_dyn)

        self.last_h_dynamic = H_dynamic

        # --- [2. Retrieve Physical Incidence Matrix H_physical] ---
        H_physical = self.finger_masks.unsqueeze(0).expand(N, -1, -1)

        # --- [3. Hybrid Fusion] ---
        H_final = torch.cat([H_dynamic, H_physical], dim=-1)  # (N, V, M_dyn + 5)

        return H_final

    def get_loss(self):
        """
        Returns: entropy, proto_ortho, phy_ortho
        """
        if self.last_h_dynamic is None or not self.use_virtual_conn:
            device = self.key_prototypes.device
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero

        # 1. Entropy Loss
        H = self.last_h_dynamic
        entropy = -torch.sum(H * torch.log(H + 1e-6), dim=-1).mean()

        # 2. Prototype Orthogonality Loss
        k = self.key_prototypes
        k_norm = F.normalize(k, p=2, dim=0)
        gram = torch.matmul(k_norm.T, k_norm)
        identity = torch.eye(gram.shape[0], device=gram.device)
        loss_proto_ortho = torch.mean((gram * (1 - identity)) ** 2)

        # 3. Physical Constraint Loss
        # H_dynamic: (N, V, M_dyn) -> (N, M_dyn, V)
        H_dyn_flat = self.last_h_dynamic.permute(0, 2, 1)

        # H_physical: (V, 5) -> (N, V, 5) -> (N, 5, V)
        H_phy_flat = self.finger_masks.unsqueeze(0).expand(H.shape[0], -1, -1).permute(0, 2, 1)

        H_dyn_norm = F.normalize(H_dyn_flat, p=2, dim=-1)
        H_phy_norm = F.normalize(H_phy_flat, p=2, dim=-1)

        # Cosine Similarity: (N, M_dyn, V) @ (N, V, 5) -> (N, M_dyn, 5)
        similarity = torch.matmul(H_dyn_norm, H_phy_norm.transpose(1, 2))

        loss_phy_ortho = torch.mean(similarity ** 2)

        return entropy, loss_proto_ortho, loss_phy_ortho


class PhysicsAttention(nn.Module):
    """
    [New Module] Physics-Guided Attention Block (PGAB).
    Injects structural topology bias into self-attention.
    """
    def __init__(self, in_channels, out_channels, num_joints, hop_matrix, max_hop=3, dropout=0.0):
        super(PhysicsAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_joints = num_joints
        self.max_hop = max_hop

        # 1. Q, K, V Mappings
        # Using Conv2d 1x1 to act as Linear projection per node
        self.q_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.k_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.v_conv = nn.Conv2d(in_channels, out_channels, 1)

        # 2. Physics Bias Embedding
        # Embeddings for distances: 0, 1, 2, ..., max_hop, >max_hop
        # Total indices = max_hop + 2
        self.bias_embedding = nn.Embedding(max_hop + 2, 1)

        # Initialization Strategy (Distance-based decay)
        # 0-hop (Self) -> 1.0
        # 1-hop (Direct) -> 0.5
        # 2-hop (Neighbor) -> 0.1
        # >2-hop -> 0.0 or small negative
        with torch.no_grad():
            self.bias_embedding.weight.data.fill_(0.0)  # Initialize all to 0.0
            self.bias_embedding.weight.data[0] = 1.0    # Dist 0
            if max_hop >= 1:
                self.bias_embedding.weight.data[1] = 0.5  # Dist 1
            if max_hop >= 2:
                self.bias_embedding.weight.data[2] = 0.1  # Dist 2
            # Distances > 2 remain 0.0

        # 3. Hop Matrix (Buffer)
        # Ensure it's a LongTensor
        if not isinstance(hop_matrix, torch.Tensor):
            hop_matrix = torch.tensor(hop_matrix, dtype=torch.long)
        self.register_buffer('hop_matrix', hop_matrix)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V)
        Returns:
            out: (N, C_out, T, V)
        """
        N, C, T, V = x.shape

        # Projections
        # Q: (N, C_out, T, V) -> (N, T, V, C_out) -> (N*T, V, C_out)
        q = self.q_conv(x).permute(0, 2, 3, 1).contiguous().view(-1, V, self.out_channels)
        k = self.k_conv(x).permute(0, 2, 3, 1).contiguous().view(-1, V, self.out_channels)
        v = self.v_conv(x).permute(0, 2, 3, 1).contiguous().view(-1, V, self.out_channels)

        # Standard Self-Attention Score: (Q @ K^T) / sqrt(d)
        # (N*T, V, C_out) @ (N*T, C_out, V) -> (N*T, V, V)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.out_channels)

        # Inject Physical Bias
        # bias_embedding(hop_matrix) -> (V, V, 1) -> squeeze -> (V, V)
        # Add unsqueeze(0) for broadcasting over batch dimension (N*T)
        physics_bias = self.bias_embedding(self.hop_matrix).squeeze(-1)
        attn_score = attn_score + physics_bias.unsqueeze(0)

        # Softmax & Dropout
        attn = self.softmax(attn_score)
        attn = self.dropout(attn)

        # Value Aggregation
        # (N*T, V, V) @ (N*T, V, C_out) -> (N*T, V, C_out)
        output = torch.matmul(attn, v)

        # Reshape back to (N, C_out, T, V)
        # (N*T, V, C_out) -> (N, T, V, C_out) -> (N, C_out, T, V)
        output = output.view(N, T, V, self.out_channels).permute(0, 3, 1, 2).contiguous()

        return output


class unit_hypergcn(nn.Module):
    def __init__(self, in_channels, out_channels, num_hyperedges=16, residual=True, num_point=21, **kwargs):
        super(unit_hypergcn, self).__init__()

        # [MODIFIED] Use PhysicallyGuidedDSAHypergraph
        # Pass num_point explicitly to ensure masks are created correctly
        self.dhg = PhysicallyGuidedDSAHypergraph(in_channels, num_dynamic_edges=num_hyperedges, num_point=num_point,
                                                 **kwargs)

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

        # 1. Hybrid Incidence Matrix (Dynamic + Physical)
        H = self.dhg(x)

        # 2. Hypergraph Convolution
        # Add epsilon for numerical stability
        H_norm_v = H / (H.sum(dim=1, keepdim=True) + 1e-5)

        x_v2e_feat = self.conv_v2e(x)
        x_edge = torch.einsum('nctv,nve->ncte', x_v2e_feat, H_norm_v)

        x_e_feat = self.conv_e(x_edge)

        H_norm_e = H / (H.sum(dim=2, keepdim=True) + 1e-5)
        x_node = torch.einsum('ncte,nev->nctv', x_e_feat, H_norm_e.transpose(1, 2))

        # 3. Residual & Activation
        y = self.bn(x_node)
        y = y + self.down(x)
        y = self.relu(y)
        return y