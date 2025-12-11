import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperGraphGenerator(nn.Module):
    """
    Generates dynamic hypergraph structure (Incidence Matrix H) from node features.
    Updated for Phase 1 Optimization: Includes Temperature Scaling.
    """

    def __init__(self, in_channels, num_hyperedges, k_neighbors=None, use_virtual_conn=True, temperature=0.5):
        super(HyperGraphGenerator, self).__init__()

        self.num_hyperedges = num_hyperedges
        self.k_neighbors = k_neighbors
        self.use_virtual_conn = use_virtual_conn
        self.temperature = temperature  # [Phase 1] 温度系数

        # Feature transformation before structure generation
        self.f_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_hyperedges, kernel_size=1),
            nn.BatchNorm2d(num_hyperedges),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.size()

        # Global Average Pooling over Time to get static structure per sample
        # x_mean: (N, C, 1, V) -> (N, C, V)
        x_mean = x.mean(dim=2)

        # Transform features to hyperedge space
        # h_feat: (N, num_hyperedges, V)
        h_feat = self.f_conv(x_mean.unsqueeze(-1)).squeeze(-1)

        # Calculate similarity/attention to form Incidence Matrix H
        # [PHASE 1 CRITICAL FIX] Temperature Scaling
        # H: (N, num_hyperedges, V) -> Softmax over V (nodes in edge)
        H = F.softmax(h_feat / self.temperature, dim=-1)

        return H


class unit_hypergcn(nn.Module):
    """
    DSA-HGN Unit: Dynamic Structure Adaptation Hypergraph Convolution
    """

    def __init__(self, in_channels, out_channels, num_hyperedges=16, k_neighbors=4, use_virtual_conn=True, bias=True):
        super(unit_hypergcn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_hyperedges = num_hyperedges

        # Structure Generator with Temperature Scaling
        self.graph_gen = HyperGraphGenerator(
            in_channels,
            num_hyperedges,
            k_neighbors=k_neighbors,
            use_virtual_conn=use_virtual_conn,
            temperature=0.5  # [Phase 1] 默认温度 0.5
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Output projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(num_hyperedges, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

        # Storage for visualization
        self.last_h = None

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.size()

        # 1. Generate Dynamic Incidence Matrix H
        # H: (N, M, V) where M = num_hyperedges
        H = self.graph_gen(x)
        self.last_h = H  # Save for topology export

        # 2. Hypergraph Convolution
        # Formula: X' = D_e^{-1} * H^T * W * D_v^{-1} * H * X

        # Feature transformation
        # x_feat: (N, C_out, T, V)
        x_feat = self.conv(x)

        # [FIXED] Permute to (N, C, V, T) for aggregation
        x_feat_reshaped = x_feat.permute(0, 1, 3, 2).contiguous()

        # Step 1: Nodes to Hyperedges
        # H: (N, M, V)
        # X: (N, C, V, T)
        # Y_edge = H * X -> (N, C, M, T) (Contract V)
        Y_edge = torch.einsum('nmv, ncvt -> ncmt', H, x_feat_reshaped)

        # Step 2: Hyperedges back to Nodes
        # X_out = H^T * Y_edge
        # (N, M, V)^T -> (N, V, M)
        # X_recon = H^T * Y -> (N, C, V, T) (Contract M)
        X_recon = torch.einsum('nmv, ncmt -> ncvt', H, Y_edge)

        # [FIXED] Permute back to (N, C, T, V) to match GCN branch
        # Before: (N, C, V, T) -> Now: (N, C, T, V)
        X_recon = X_recon.permute(0, 1, 3, 2).contiguous()

        return X_recon

    def get_loss(self):
        """
        Calculates Entropy and Orthogonality loss for the learned structure H.
        """
        if self.last_h is None:
            return torch.tensor(0.0).to(self.graph_gen.f_conv[0].weight.device), torch.tensor(0.0).to(
                self.graph_gen.f_conv[0].weight.device)

        H = self.last_h  # (N, M, V)

        # 1. Entropy Loss: -sum(p * log(p))
        # We want LOW entropy (sparse connection).
        epsilon = 1e-8
        entropy = -torch.sum(H * torch.log(H + epsilon), dim=-1).mean()

        # 2. Orthogonality Loss: H * H^T should be Identity (edges are diverse)
        # HH_T: (N, M, M)
        HH_T = torch.matmul(H, H.transpose(1, 2))
        I = torch.eye(self.num_hyperedges).to(H.device).unsqueeze(0)
        ortho_loss = torch.norm(HH_T - I, p='fro') / H.size(0)  # Mean over batch

        return entropy, ortho_loss