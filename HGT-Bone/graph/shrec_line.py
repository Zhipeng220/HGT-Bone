import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

# SHREC Line Graph: 21 Nodes (Bones)
num_node = 21
self_link = [(i, i) for i in range(num_node)]

# 骨骼索引定义 (与 Feeder 中的 bone_pairs 对应):
# 0: Wrist->Palm
# Thumb: 1, 2, 3, 4
# Index: 5, 6, 7, 8
# Middle: 9, 10, 11, 12
# Ring: 13, 14, 15, 16
# Pinky: 17, 18, 19, 20

# 邻接定义：如果两根骨骼共享一个关节，则视为相连
# 这里的索引对 (i, j) 表示骨骼 i 和骨骼 j 相连
inward_ori_index = [
    # 1. 掌骨 (Bone 0) 连接所有手指的基骨
    (0, 1),  # Wrist-Palm <-> Palm-ThumbBase
    (0, 5),  # Wrist-Palm <-> Palm-IndexBase
    (0, 9),  # ... Middle
    (0, 13),  # ... Ring
    (0, 17),  # ... Pinky

    # 2. 拇指链 (Thumb)
    (1, 2), (2, 3), (3, 4),

    # 3. 食指链 (Index)
    (5, 6), (6, 7), (7, 8),

    # 4. 中指链 (Middle)
    (9, 10), (10, 11), (11, 12),

    # 5. 无名指链 (Ring)
    (13, 14), (14, 15), (15, 16),

    # 6. 小指链 (Pinky)
    (17, 18), (18, 19), (19, 20),

    # 7. 手指间的跨指连接 (可选，模拟手掌处的物理约束)
    # Palm-IndexBase (5) <-> Palm-MiddleBase (9) 等
    # 增加这些连接有助于信息在手指间传递 (Hyper-Bone 思想)
    (5, 9), (9, 13), (13, 17)
]

inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError(f"Labeling mode '{labeling_mode}' is not supported")
        return A