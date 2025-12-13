import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 22
self_link = [(i, i) for i in range(num_node)]

# SHREC'17 Track Layout (22 Joints)
# 0: Wrist
# 1: Palm
# 2-5: Thumb (Base, Joint1, Joint2, Tip)
# 6-9: Index
# 10-13: Middle
# 14-17: Ring
# 18-21: Pinky

# 定义邻接列表 (neighbor_link) - 排除自环
neighbor_link = [
    # Wrist to Palm
    (0, 1),
    # Palm to Fingers
    (1, 2), (1, 6), (1, 10), (1, 14), (1, 18),
    # Thumb: 1-2-3-4-5
    (2, 3), (3, 4), (4, 5),
    # Index: 1-6-7-8-9
    (6, 7), (7, 8), (8, 9),
    # Middle: 1-10-11-12-13
    (10, 11), (11, 12), (12, 13),
    # Ring: 1-14-15-16-17
    (14, 15), (15, 16), (16, 17),
    # Pinky: 1-18-19-20-21
    (18, 19), (19, 20), (20, 21)
]

inward = [(i, j) for (i, j) in neighbor_link]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.neighbor_link = neighbor_link  # 保存基础拓扑结构
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError(f"Labeling mode '{labeling_mode}' is not supported")
        return A

    def get_hop_matrix(self, max_hop=3):
        """
        计算节点的跳数距离矩阵 (Hop Distance Matrix)。

        Args:
            max_hop (int): 最大跳数截断值。超过此距离的值将被设为 max_hop + 1。

        Returns:
            np.ndarray: 形状为 (num_node, num_node) 的整数矩阵。
        """
        # 初始化距离矩阵为无穷大
        adj = np.full((self.num_node, self.num_node), np.inf)

        # 自身距离为0
        np.fill_diagonal(adj, 0)

        # 直接相邻节点距离为1
        for i, j in self.neighbor:
            adj[i, j] = 1
            adj[j, i] = 1  # 确保无向性

        # 使用 Floyd-Warshall 算法计算所有节点对之间的最短路径
        # 由于节点数较少 (22)，O(N^3) 是完全可接受的
        for k in range(self.num_node):
            for i in range(self.num_node):
                for j in range(self.num_node):
                    adj[i, j] = min(adj[i, j], adj[i, k] + adj[k, j])

        # 截断距离：超过 max_hop 的距离设为 max_hop + 1
        hop_matrix = np.where(adj > max_hop, max_hop + 1, adj)

        return hop_matrix.astype(int)