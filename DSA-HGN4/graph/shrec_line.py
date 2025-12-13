import sys
import numpy as np

# 将上级目录加入路径以便导入模块
sys.path.extend(['../'])
from graph import tools


def build_line_graph_from_joint_graph(joint_edges, num_joints):
    """
    根据关节图构建线图 (Line Graph)

    理论依据:
    《Bone流动作识别精度提升研究》 Section 3.1
    线图邻接矩阵 A_line = N^T * N - 2I
    其中 N 为关联矩阵 (Incidence Matrix), 维度 [joints, bones]

    Args:
        joint_edges: list of tuple, 定义了21根骨骼的关节连接关系 (j1, j2)
        num_joints: int, 原始关节数量 (SHREC 2017为22)

    Returns:
        line_edges: list of tuple, 线图中的边 (即骨骼间的连接关系)
    """
    num_bones = len(joint_edges)

    # 1. 构建关联矩阵 N (joints x bones)
    # N[i, j] = 1 表示关节 i 是骨骼 j 的端点
    N = np.zeros((num_joints, num_bones))
    for bone_idx, (j1, j2) in enumerate(joint_edges):
        N[j1, bone_idx] = 1
        N[j2, bone_idx] = 1

    # 2. 计算线图邻接矩阵 A_line
    # N.T @ N 计算每两根骨骼共享的关节数量 (0, 1, 或 2)
    # 减去 2I 是为了去除骨骼自身的两个端点导致的自环 (权重为2)
    A_line = np.dot(N.T, N) - 2 * np.eye(num_bones)

    # 3. 提取线图的边
    # 如果 A_line[i, j] > 0，说明骨骼 i 和骨骼 j 共享至少一个关节，即在线图中相连
    line_edges = []
    for i in range(num_bones):
        for j in range(i + 1, num_bones):  # 只遍历上三角，避免重复
            if A_line[i, j] > 0:
                line_edges.append((i, j))

    return line_edges


# =============================================================================
# SHREC 2017 骨骼定义与图构建
# =============================================================================

# 线图节点数 = 骨骼数 = 21
num_node = 21

# 自环定义 (GCN 需要)
self_link = [(i, i) for i in range(num_node)]

# SHREC 原始骨骼定义 (Bone Pairs)
# 索引对应关系:
# 0: Wrist-Palm
# 1-4: Thumb, 5-8: Index, 9-12: Middle, 13-16: Ring, 17-20: Pinky
bone_pairs_shrec = [
    (0, 1),  # Bone 0: Wrist -> Palm
    (1, 2), (2, 3), (3, 4), (4, 5),  # Bone 1-4: Thumb
    (1, 6), (6, 7), (7, 8), (8, 9),  # Bone 5-8: Index
    (1, 10), (10, 11), (11, 12), (12, 13),  # Bone 9-12: Middle
    (1, 14), (14, 15), (15, 16), (16, 17),  # Bone 13-16: Ring
    (1, 18), (18, 19), (19, 20), (20, 21)  # Bone 17-20: Pinky
]

# 自动生成线图的物理连接 (Inward connections)
inward_ori_index = build_line_graph_from_joint_graph(
    bone_pairs_shrec,
    num_joints=22
)

# 构建有向图边列表 (双向)
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
            # 使用 tools 生成标准空间配置邻接矩阵 (Spatial Configuration)
            # 这会将邻居分为三类: 根节点本身, 向心节点, 离心节点
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError(f"Labeling mode '{labeling_mode}' is not supported")
        return A