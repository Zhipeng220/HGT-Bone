import sys
import numpy as np

# 将上级目录加入路径以便导入模块
sys.path.extend(['../'])
from graph import tools


def build_line_graph_from_joint_graph(joint_edges, num_joints):
    """
    根据关节图构建线图 (Line Graph)
    """
    num_bones = len(joint_edges)

    # 1. 构建关联矩阵 N (joints x bones)
    N = np.zeros((num_joints, num_bones))
    for bone_idx, (j1, j2) in enumerate(joint_edges):
        N[j1, bone_idx] = 1
        N[j2, bone_idx] = 1

    # 2. 计算线图邻接矩阵 A_line
    A_line = np.dot(N.T, N) - 2 * np.eye(num_bones)

    # 3. 提取线图的边
    line_edges = []
    for i in range(num_bones):
        for j in range(i + 1, num_bones):  # 只遍历上三角，避免重复
            if A_line[i, j] > 0:
                line_edges.append((i, j))

    return line_edges


def get_hop_distance(num_node, edge, max_hop=3):
    """
    [FIXED] 计算图节点的跳数矩阵 (最短路径)，用于 PhysicsAttention
    """
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # 初始化为无穷大
    hop_dis = np.zeros((num_node, num_node)) + np.inf

    # 动态规划或矩阵幂次计算跳数
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)

    # 倒序填充，保留最小跳数
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d

    return hop_dis


# =============================================================================
# SHREC 2017 骨骼定义与图构建
# =============================================================================

# 线图节点数 = 骨骼数 = 21
num_node = 21

# 自环定义 (GCN 需要)
self_link = [(i, i) for i in range(num_node)]

# SHREC 原始骨骼定义 (Bone Pairs) - 22个关节
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

        # [FIXED] 1. 计算跳数矩阵，供 PhysicsAttention 使用
        # 对于骨骼流，self.neighbor 实际上定义了骨骼间的连接关系
        self.hop_dis = get_hop_distance(self.num_node, self.neighbor, max_hop=3)

        # [FIXED] 2. 暴露图的边信息，供 Recognition Loss 计算使用
        self.edge = self.neighbor

        # [FIXED] 3. 暴露原始关节连接，用于关节回归任务的 Bone Direction Loss
        # 因为回归头预测的是关节坐标，所以我们需要知道哪些关节连成了一根骨头
        self.source_edges = bone_pairs_shrec

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError(f"Labeling mode '{labeling_mode}' is not supported")
        return A