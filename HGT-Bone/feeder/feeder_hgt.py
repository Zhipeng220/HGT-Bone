import sys
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from scipy.signal import savgol_filter

# 引入工具模块
try:
    from . import tools
except ImportError:
    import tools


class HGTFeatureExtractor:
    """
    HGT-Bone 核心特征提取器

    流程:
    1. S-G 滤波去噪 (Denoising)
    2. 提取 8 维高阶几何特征:
       - Channels 0-2: 骨骼向量 (Bone Vectors, 3D)
       - Channel 3:    骨骼模长 (Bone Lengths, 1D)
       - Channel 4:    弯曲角度 (Flexion Angles, 1D)
       - Channels 5-7: 旋转轴/法向量 (Rotation Axes, 3D)
    """

    def __init__(self):
        # SHREC 2017 骨骼连接定义 (22关节 -> 21骨骼)
        # 索引: 0:Wrist, 1:Palm, 2-5:Thumb, ...
        self.bone_pairs = (
            (0, 1),  # Bone 0: Wrist-Palm
            (1, 2), (2, 3), (3, 4), (4, 5),  # Thumb
            (1, 6), (6, 7), (7, 8), (8, 9),  # Index
            (1, 10), (10, 11), (11, 12), (12, 13),  # Middle
            (1, 14), (14, 15), (15, 16), (16, 17),  # Ring
            (1, 18), (18, 19), (19, 20), (20, 21)  # Pinky
        )
        self.num_bones = len(self.bone_pairs)  # 21
        self.bone_parents = self._build_bone_tree()

    def _build_bone_tree(self):
        """构建骨骼树，找到每根骨骼的父骨骼索引"""
        parents = [-1] * self.num_bones
        for i, (j1, j2) in enumerate(self.bone_pairs):
            if i == 0: continue  # 根骨骼无父
            # 寻找终点是当前起点的骨骼
            for parent_idx, (p1, p2) in enumerate(self.bone_pairs[:i]):
                if p2 == j1:
                    parents[i] = parent_idx
                    break
        return parents

    def _denoise_sg(self, data, window=9, order=2):
        """Savitzky-Golay 滤波"""
        # data: (C, T, V, M)
        C, T, V, M = data.shape
        filtered = np.zeros_like(data)
        for m in range(M):
            for v in range(V):
                for c in range(C):
                    try:
                        filtered[c, :, v, m] = savgol_filter(
                            data[c, :, v, m],
                            window_length=min(window, T),
                            polyorder=order
                        )
                    except ValueError:
                        filtered[c, :, v, m] = data[c, :, v, m]
        return filtered

    def extract(self, joint_coords):
        """
        Input: (3, T, 22, M) 关节坐标
        Output: (8, T, 21, M) HGT特征
        """
        # 1. 滤波
        joints = self._denoise_sg(joint_coords)

        # 2. 计算骨骼向量
        bone_vecs_list = []
        for v1, v2 in self.bone_pairs:
            vec = joints[:, :, v1, :] - joints[:, :, v2, :]
            bone_vecs_list.append(vec)
        bone_vecs = np.stack(bone_vecs_list, axis=2)  # (3, T, 21, M)

        # 3. 计算模长 & 归一化方向
        bone_lens = np.linalg.norm(bone_vecs, axis=0, keepdims=True) + 1e-6
        bone_dirs = bone_vecs / bone_lens

        # 4. 计算几何关系 (角度 & 旋转轴)
        angles_list = []
        rot_axes_list = []

        for i in range(self.num_bones):
            parent_idx = self.bone_parents[i]
            curr_dir = bone_dirs[:, :, i, :]

            if parent_idx == -1:
                angles_list.append(np.zeros_like(curr_dir[:1]))
                rot_axes_list.append(np.zeros_like(curr_dir))
            else:
                parent_dir = bone_dirs[:, :, parent_idx, :]

                # 弯曲角 (Dot Product)
                cos_theta = np.sum(curr_dir * parent_dir, axis=0, keepdims=True)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angles_list.append(cos_theta)

                # 旋转轴 (Cross Product) - 代表骨骼弯曲所在的平面
                axis = np.cross(parent_dir, curr_dir, axis=0)
                axis_norm = np.linalg.norm(axis, axis=0, keepdims=True) + 1e-6
                axis = axis / axis_norm
                rot_axes_list.append(axis)

        angles = np.stack(angles_list, axis=2)  # (1, T, 21, M)
        rot_axes = np.stack(rot_axes_list, axis=2)  # (3, T, 21, M)

        # 5. 拼接特征
        features = np.concatenate([bone_vecs, bone_lens, angles, rot_axes], axis=0)
        return features.astype(np.float32)


class Feeder(Dataset):
    """
    HGT-Bone 数据加载器
    Pipeline: Load -> View Align -> Spatial Aug -> HGT Extract -> Temporal Aug -> Norm
    """

    def __init__(self, data_path, label_path, split='train',
                 random_choose=False, random_shift=False, random_move=False,
                 random_rot=False, window_size=-1, normalization=False,
                 debug=False, use_mmap=True, bone=False, vel=False,
                 mean_map=None, std_map=None, repeat=1):

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = False  # [Fix] 强制关闭 random_shift 以防止报错
        self.random_move = random_move
        self.random_rot = random_rot
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.repeat = repeat

        self.load_data()
        self.extractor = HGTFeatureExtractor()

        if normalization:
            self.mean_map = mean_map
            self.std_map = std_map

    def load_data(self):
        try:
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            with open(self.label_path, 'rb') as f:
                self.label, self.sample_name = pickle.load(f)

        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def view_invariant_transform(self, data_numpy):
        """
        视角对齐: 消除手势识别中的视角差异
        以每一帧的腕关节(Wrist)为原点，建立局部坐标系
        """
        C, T, V, M = data_numpy.shape
        data_aligned = np.zeros_like(data_numpy)

        for m in range(M):
            for t in range(T):
                frame_data = data_numpy[:, t, :, m]  # (3, V)

                # 1. 平移: Wrist (Index 0) -> Origin
                wrist = frame_data[:, 0]
                frame_data = frame_data - wrist[:, None]

                # 2. 旋转对齐
                # v1: Wrist -> Palm (Index 1) 作为 Y轴
                v1 = frame_data[:, 1]
                norm_v1 = np.linalg.norm(v1) + 1e-6
                v1 = v1 / norm_v1

                # v2: Wrist -> IndexBase (Index 6) 辅助确定平面
                v2 = frame_data[:, 6]
                v2 = v2 / (np.linalg.norm(v2) + 1e-6)

                # Z轴: 掌心平面法线
                z_axis = np.cross(v1, v2)
                z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-6)

                # X轴: 正交化
                x_axis = np.cross(v1, z_axis)
                x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)

                # 旋转矩阵
                R = np.stack([x_axis, v1, z_axis])  # (3, 3)

                data_aligned[:, t, :, m] = np.dot(R, frame_data)

        return data_aligned

    def __len__(self):
        return len(self.label) * self.repeat

    def __iter__(self):
        return self

    def __getitem__(self, index):
        index = index % len(self.label)
        data_numpy = np.array(self.data[index])  # (3, T, 22, M)
        label = self.label[index]

        # 1. 视角对齐 (Pre-processing)
        data_numpy = self.view_invariant_transform(data_numpy)

        # 2. 空间增强 (Spatial Augmentation) - 在3D坐标上做
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        # 3. HGT 特征提取
        # Input: (3, T, 22, M) -> Output: (8, T, 21, M)
        data_numpy = self.extractor.extract(data_numpy)

        # 4. 时间增强 (Temporal Augmentation)
        # random_choose 会返回固定长度(window_size)的数据，通常是稠密的
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)

        # [Fix] 移除 random_shift
        # random_shift 需要数据中有零填充区域才能工作，而 random_choose 返回的数据通常是满的
        # 且 HGT 特征提取后，数据不再是简单的坐标，移位操作需谨慎
        # if self.random_shift:
        #     data_numpy = tools.random_shift(data_numpy)

        # 5. 归一化
        if self.normalization and self.mean_map is not None:
            data_numpy = (data_numpy - self.mean_map) / (self.std_map + 1e-4)

        # 数值清理 (防止 S-G 滤波或除法产生 NaN)
        data_numpy = np.nan_to_num(data_numpy, copy=False, nan=0.0, posinf=100.0, neginf=-100.0)

        return data_numpy, label