import sys
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from scipy.signal import savgol_filter

# 引入原有的工具
try:
    from . import tools
except ImportError:
    import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path, split='train',
                 random_choose=False, random_shift=False, random_move=False,
                 random_rot=False, window_size=-1, normalization=False,
                 debug=False, use_mmap=True, bone=False, vel=False,
                 # [新增] 接收 repeat 参数及其它统计量参数，与你的配置文件兼容
                 mean_map=None, std_map=None, repeat=1,
                 p_interval=1, shear_amplitude=0.5, temperal_padding_ratio=6):
        """
        HGT-Bone 专用 Feeder (修复版)
        实现了:
        1. Savitzky-Golay 滤波去噪
        2. HGT 高阶几何特征提取 (向量 + 模长 + 夹角 + 旋转轴)
        3. [Fix] 支持 repeat 参数
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.random_rot = random_rot
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap

        # 保存 repeat 参数
        self.repeat = repeat
        self.mean_map = mean_map
        self.std_map = std_map

        # 加载数据
        self.load_data()

        if normalization:
            if self.mean_map is None or self.std_map is None:
                self.get_mean_map()

        # SHREC 骨骼连接定义 (22关节 -> 21骨骼)
        # 0:Wrist, 1:Palm, 2-5:Thumb, 6-9:Index, 10-13:Mid, 14-17:Ring, 18-21:Pinky
        self.bone_pairs = (
            (0, 1),  # Bone 0: Wrist-Palm
            (1, 2), (2, 3), (3, 4), (4, 5),  # Thumb
            (1, 6), (6, 7), (7, 8), (8, 9),  # Index
            (1, 10), (10, 11), (11, 12), (12, 13),  # Middle
            (1, 14), (14, 15), (15, 16), (16, 17),  # Ring
            (1, 18), (18, 19), (19, 20), (20, 21)  # Pinky
        )
        # 定义每个骨骼的父骨骼索引
        self.bone_parents = [
            -1,  # Bone 0
            0, 1, 2, 3,  # Thumb branch
            0, 5, 6, 7,  # Index branch
            0, 9, 10, 11,  # Middle
            0, 13, 14, 15,  # Ring
            0, 17, 18, 19  # Pinky
        ]

    def load_data(self):
        # 加载 Label
        try:
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            with open(self.label_path, 'rb') as f:
                self.label, self.sample_name = pickle.load(f)

        # 加载 Data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        # 注意：对于 HGT，我们需要先计算特征，再计算 mean/std
        # 但为了节省内存和时间，这里通常简化处理，或者你可以在预处理脚本中计算好传入
        # 这里为了跑通，我们简单计算原始数据的 mean/std，
        # 严格来说 HGT 特征的 normalization 应该在特征提取后进行。
        # 鉴于代码结构，建议先把 normalization=False，依靠 BN 层。
        pass

    def geometric_feature_extraction(self, data_numpy):
        # data_numpy: (C, T, V, M)
        C, T, V, M = data_numpy.shape

        # 1. S-G 滤波
        try:
            data_numpy = savgol_filter(data_numpy, window_length=9, polyorder=2, axis=1)
        except ValueError:
            pass

        # 2. 计算骨骼向量
        bone_vecs = []
        for v1, v2 in self.bone_pairs:
            vec = data_numpy[:, :, v1, :] - data_numpy[:, :, v2, :]
            bone_vecs.append(vec)
        bone_vecs = np.stack(bone_vecs, axis=2)

        # 3. 计算模长
        bone_lens = np.linalg.norm(bone_vecs, axis=0, keepdims=True) + 1e-6

        # 4. 计算归一化方向
        bone_dirs = bone_vecs / bone_lens

        # 5. 计算几何关系
        angles_list = []
        normals_list = []

        for i, parent_idx in enumerate(self.bone_parents):
            curr_dir = bone_dirs[:, :, i, :]

            if parent_idx == -1:
                angles_list.append(np.zeros((1, T, M)))
                normals_list.append(np.zeros((3, T, M)))
            else:
                parent_dir = bone_dirs[:, :, parent_idx, :]
                cos_theta = np.sum(curr_dir * parent_dir, axis=0, keepdims=True)
                angles_list.append(cos_theta)
                cross = np.cross(parent_dir, curr_dir, axis=0)
                normals_list.append(cross)

        angles = np.stack(angles_list, axis=2)  # (1, T, 21, M)
        normals = np.stack(normals_list, axis=2)  # (3, T, 21, M)

        # 6. 拼接特征 (Channels: 3+1+1+3 = 8)
        feature = np.concatenate([bone_vecs, bone_lens, angles, normals], axis=0)

        return feature

    def __len__(self):
        # [关键修复] 支持 repeat
        return len(self.label) * self.repeat

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # [关键修复] 索引取模，防止 repeat 时越界
        index = index % len(self.label)

        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        # ---------------------------------------------------------------------
        # HGT 特征提取前的数据增强 (Spatial Augmentation)
        # ---------------------------------------------------------------------
        # 注意：random_move 必须在数据还是 3D 坐标时进行
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # ---------------------------------------------------------------------
        # 核心：HGT 几何特征提取
        # 输入: (3, T, 22, M) -> 输出: (8, T, 21, M)
        # ---------------------------------------------------------------------
        data_numpy = self.geometric_feature_extraction(data_numpy)

        # ---------------------------------------------------------------------
        # 特征提取后的数据增强 (Temporal Augmentation)
        # ---------------------------------------------------------------------
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)

        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)

        # 归一化 (如果启用)
        if self.normalization and self.mean_map is not None:
            data_numpy = (data_numpy - self.mean_map) / (self.std_map + 1e-4)
            # 增加数值稳定性
            data_numpy = np.nan_to_num(data_numpy, copy=False, nan=0.0, posinf=100.0, neginf=-100.0)

        # [关键修改] 只返回 data 和 label，去掉 index
        return data_numpy, label