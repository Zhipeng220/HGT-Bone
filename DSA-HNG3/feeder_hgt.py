import sys
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from scipy.signal import savgol_filter

try:
    from . import tools
except ImportError:
    import tools


class HGTFeatureExtractor:
    """HGT-Bone 核心特征提取器"""

    def __init__(self, use_sg=True, sg_window=9, sg_order=2,
                 use_angles=True, use_rotations=True):
        self.use_sg = use_sg
        self.sg_window = sg_window
        self.sg_order = sg_order
        self.use_angles = use_angles
        self.use_rotations = use_rotations

        # SHREC 2017 骨骼连接定义 (22关节 -> 21骨骼)
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
        """构建骨骼树"""
        parents = [-1] * self.num_bones
        for i, (j1, j2) in enumerate(self.bone_pairs):
            if i == 0: continue
            for parent_idx, (p1, p2) in enumerate(self.bone_pairs[:i]):
                if p2 == j1:
                    parents[i] = parent_idx
                    break
        return parents

    def _denoise_sg(self, data):
        """Savitzky-Golay 滤波"""
        C, T, V, M = data.shape
        filtered = np.zeros_like(data)

        window = self.sg_window
        order = self.sg_order

        effective_window = min(window, T)
        if effective_window % 2 == 0:
            effective_window -= 1

        if effective_window < 3:
            return data

        for m in range(M):
            for v in range(V):
                for c in range(C):
                    try:
                        filtered[c, :, v, m] = savgol_filter(
                            data[c, :, v, m],
                            window_length=effective_window,
                            polyorder=order
                        )
                    except ValueError:
                        filtered[c, :, v, m] = data[c, :, v, m]
        return filtered

    def extract(self, joint_coords):
        """
        Input: (3, T, 22, M) 关节坐标
        Output: (D, T, 21, M) HGT特征
        """
        if self.use_sg:
            joints = self._denoise_sg(joint_coords)
        else:
            joints = joint_coords

        # 基础特征: 骨骼向量
        bone_vecs_list = []
        for v1, v2 in self.bone_pairs:
            vec = joints[:, :, v1, :] - joints[:, :, v2, :]
            bone_vecs_list.append(vec)
        bone_vecs = np.stack(bone_vecs_list, axis=2)  # (3, T, 21, M)

        # 基础特征: 模长 & 归一化方向
        bone_lens = np.linalg.norm(bone_vecs, axis=0, keepdims=True) + 1e-6
        bone_dirs = bone_vecs / bone_lens

        feature_list = [bone_vecs, bone_lens]

        # 可选特征: 几何关系
        if self.use_angles or self.use_rotations:
            angles_list = []
            rot_axes_list = []

            for i in range(self.num_bones):
                parent_idx = self.bone_parents[i]
                curr_dir = bone_dirs[:, :, i, :]

                if parent_idx == -1:
                    if self.use_angles:
                        angles_list.append(np.zeros_like(curr_dir[:1]))
                    if self.use_rotations:
                        rot_axes_list.append(np.zeros_like(curr_dir))
                else:
                    parent_dir = bone_dirs[:, :, parent_idx, :]

                    if self.use_angles:
                        cos_theta = np.sum(curr_dir * parent_dir, axis=0, keepdims=True)
                        cos_theta = np.clip(cos_theta, -1.0, 1.0)
                        angles_list.append(cos_theta)

                    if self.use_rotations:
                        axis = np.cross(parent_dir, curr_dir, axis=0)
                        axis_norm = np.linalg.norm(axis, axis=0, keepdims=True) + 1e-6
                        axis = axis / axis_norm
                        rot_axes_list.append(axis)

            if self.use_angles:
                angles = np.stack(angles_list, axis=2)
                feature_list.append(angles)

            if self.use_rotations:
                rot_axes = np.stack(rot_axes_list, axis=2)
                feature_list.append(rot_axes)

        features = np.concatenate(feature_list, axis=0)
        return features.astype(np.float32)


class Feeder(Dataset):
    """HGT-Bone 数据加载器 - 完整修复版"""

    def __init__(self, data_path, label_path, split='train',
                 random_choose=False, random_shift=False, random_move=False,
                 random_rot=False, window_size=-1, normalization=False,
                 debug=False, use_mmap=True, bone=False, vel=False,
                 mean_map=None, std_map=None, repeat=1,
                 use_sg=True, sg_window=9, sg_order=2,
                 use_angles=True, use_rotations=True):

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift  # ✅ 保留参数
        self.random_move = random_move
        self.random_rot = random_rot
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.repeat = repeat

        self.load_data()

        self.extractor = HGTFeatureExtractor(
            use_sg=use_sg,
            sg_window=sg_window,
            sg_order=sg_order,
            use_angles=use_angles,
            use_rotations=use_rotations
        )

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
        """视角对齐"""
        C, T, V, M = data_numpy.shape
        data_aligned = np.zeros_like(data_numpy)

        for m in range(M):
            for t in range(T):
                frame_data = data_numpy[:, t, :, m]

                wrist = frame_data[:, 0]
                frame_data = frame_data - wrist[:, None]

                v1 = frame_data[:, 1]
                norm_v1 = np.linalg.norm(v1) + 1e-6
                v1 = v1 / norm_v1

                v2 = frame_data[:, 6]
                v2 = v2 / (np.linalg.norm(v2) + 1e-6)

                z_axis = np.cross(v1, v2)
                z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-6)

                x_axis = np.cross(v1, z_axis)
                x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)

                R = np.stack([x_axis, v1, z_axis])
                data_aligned[:, t, :, m] = np.dot(R, frame_data)

        return data_aligned

    def __len__(self):
        return len(self.label) * self.repeat

    def __iter__(self):
        return self

    def _apply_temporal_shift(self, data, max_shift_ratio=0.2):
        """
        ✅ [新增] 时序平移增强
        随机在时间轴上平移数据，增强模型对动作起始时间的鲁棒性

        Args:
            data: (C, T, V, M) 格式的数据
            max_shift_ratio: 最大平移比例 (相对于总帧数)
        Returns:
            shifted_data: 平移后的数据
        """
        C, T, V, M = data.shape
        max_shift = int(T * max_shift_ratio)

        if max_shift == 0:
            return data

        # 随机选择平移量（正负都可以）
        shift = np.random.randint(-max_shift, max_shift + 1)

        if shift == 0:
            return data

        shifted_data = np.zeros_like(data)

        if shift > 0:
            # 向后平移：前面补零或重复第一帧
            shifted_data[:, shift:, :, :] = data[:, :-shift, :, :]
            shifted_data[:, :shift, :, :] = data[:, 0:1, :, :]  # 重复第一帧
        else:
            # 向前平移：后面补零或重复最后一帧
            shift = abs(shift)
            shifted_data[:, :-shift, :, :] = data[:, shift:, :, :]
            shifted_data[:, -shift:, :, :] = data[:, -1:, :, :]  # 重复最后一帧

        return shifted_data

    def __getitem__(self, index):
        index = index % len(self.label)
        data_numpy = np.array(self.data[index])  # (3, T, 22, M)
        label = self.label[index]

        # 1. 视角对齐
        data_numpy = self.view_invariant_transform(data_numpy)

        # 2. 空间增强
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        # ✅ [修复] 2.5. 时序平移增强 - 在特征提取之前应用
        if self.random_shift:
            data_numpy = self._apply_temporal_shift(data_numpy, max_shift_ratio=0.2)

        # 保留用于回归的 Joint Data
        joint_data_for_regression = data_numpy.copy()

        # 3. HGT 特征提取 (Joint 22 -> Bone 21)
        data_features = self.extractor.extract(data_numpy)

        # 4. 时序采样
        if self.window_size > 0:
            C, T, V, M = data_features.shape
            if self.random_choose and T > self.window_size:
                # ✅ 随机起点采样（不是均匀采样）
                start = np.random.randint(0, T - self.window_size + 1)
                data_features = data_features[:, start:start + self.window_size, :, :]
                joint_data_for_regression = joint_data_for_regression[:, start:start + self.window_size, :, :]
            elif T != self.window_size:
                # 确定性采样(测试时使用)
                ids = np.linspace(0, T - 1, self.window_size).astype(int)
                data_features = data_features[:, ids, :, :]
                joint_data_for_regression = joint_data_for_regression[:, ids, :, :]

        # 5. 归一化
        if self.normalization and self.mean_map is not None:
            data_features = (data_features - self.mean_map) / (self.std_map + 1e-4)

        # 数值清理
        data_features = np.nan_to_num(data_features, copy=False, nan=0.0,
                                      posinf=100.0, neginf=-100.0)
        joint_data_for_regression = np.nan_to_num(joint_data_for_regression,
                                                  copy=False, nan=0.0)

        # ✅ 将 Joint(22) 转换为 Bone(21) 用于回归目标
        bone_data_for_regression = self._convert_joint_to_bone(joint_data_for_regression)

        return data_features, label, index, bone_data_for_regression

    def _convert_joint_to_bone(self, joint_data):
        """
        将关节坐标(3, T, 22, M)转换为骨骼向量(3, T, 21, M)
        用于Bone流的回归目标对齐
        """
        C, T, V, M = joint_data.shape
        bone_vecs_list = []

        for v1, v2 in self.extractor.bone_pairs:
            vec = joint_data[:, :, v1, :] - joint_data[:, :, v2, :]
            bone_vecs_list.append(vec)

        bone_data = np.stack(bone_vecs_list, axis=2)  # (3, T, 21, M)
        return bone_data