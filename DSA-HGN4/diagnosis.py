# diagnosis.py
import torch
import numpy as np
from net.hypergraph_modules import PhysicallyGuidedDSAHypergraph

# 创建模块
module = PhysicallyGuidedDSAHypergraph(
    in_channels=8, 
    num_dynamic_edges=12, 
    num_point=21
)

# 模拟输入
x = torch.randn(2, 8, 180, 21)  # (N, C, T, V)
H = module(x)

print(f"超边矩阵形状: {H.shape}")  # 应该是 (2, 21, 17)
print(f"动态边数: {module.num_dynamic_edges}")
print(f"物理边数: {module.finger_masks.shape[1]}")

# 计算损失
entropy, proto_ortho, phy_ortho = module.get_loss()
print(f"\n当前损失值:")
print(f"  Entropy: {entropy:.4f}")
print(f"  Proto Ortho: {proto_ortho:.4f}")
print(f"  Physical Ortho: {phy_ortho:.4f}")  # ❌ 这个值应该很高

# 检查物理边是否真的起作用
H_dynamic = module.last_h_dynamic[0]  # (V, M_dyn)
H_physical = module.finger_masks      # (21, 5) 保持原始形状，行是节点
# 计算每个节点在“动态边”和“物理边”中的平均活跃度是否一致
overlap = (H_dynamic.abs().mean(1) * H_physical.abs().mean(1)).mean()

print(f"\n动态边与物理边的重叠度: {overlap:.4f}")
# ❌ 如果接近0,说明正交约束把它们推开了