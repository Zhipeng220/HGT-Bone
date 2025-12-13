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
    HGT-Bone 核心特征提取器 (可配置版)

    支持通过参数控制：
    1. S-G 滤波去噪 (开关 + 参数)
    2. 高阶几何特征组合 (基础向量+模长 必选, 角度/旋转轴 可选)
    """

    def __init__(self, use_sg=True, sg_window=9, sg_order=2,
                 use_angles=True, use_rotations=True):

        # 保存配置参数
        self.use_sg = use_sg
        self.sg_window = sg_window
        self.sg_order = sg_order
        self.use_angles = use_angles
        self.use_rotations = use_rotations

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

    def _denoise_sg(self, data):
        """Savitzky-Golay 滤波 (使用配置参数)"""
        # data: (C, T, V, M)
        C, T, V, M = data.shape
        filtered = np.zeros_like(data)

        # 获取配置参数
        window = self.sg_window
        order = self.sg_order

        # 确保窗口大小不超过时间长度且为奇数
        effective_window = min(window, T)
        if effective_window % 2 == 0:
            effective_window -= 1

        # 如果时间太短无法滤波，直接返回原数据
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
        Output: (D, T, 21, M) HGT特征, D根据配置动态变化
        """
        # 1. 滤波 (根据开关决定)
        if self.use_sg:
            joints = self._denoise_sg(joint_coords)
        else:
            joints = joint_coords

        # 2. 基础特征: 骨骼向量 (Bone Vectors)
        bone_vecs_list = []
        for v1, v2 in self.bone_pairs:
            vec = joints[:, :, v1, :] - joints[:, :, v2, :]
            bone_vecs_list.append(vec)
        bone_vecs = np.stack(bone_vecs_list, axis=2)  # (3, T, 21, M)

        # 3. 基础特征: 模长 (Bone Lengths) & 归一化方向
        bone_lens = np.linalg.norm(bone_vecs, axis=0, keepdims=True) + 1e-6
        bone_dirs = bone_vecs / bone_lens

        # 初始化特征列表 (默认包含向量和模长)
        feature_list = [bone_vecs, bone_lens]

        # 4. 可选特征: 几何关系 (角度 & 旋转轴)
        if self.use_angles or self.use_rotations:
            angles_list = []
            rot_axes_list = []

            for i in range(self.num_bones):
                parent_idx = self.bone_parents[i]
                curr_dir = bone_dirs[:, :, i, :]

                if parent_idx == -1:
                    # 根骨骼无父节点，填充零
                    if self.use_angles:
                        angles_list.append(np.zeros_like(curr_dir[:1]))
                    if self.use_rotations:
                        rot_axes_list.append(np.zeros_like(curr_dir))
                else:
                    parent_dir = bone_dirs[:, :, parent_idx, :]

                    if self.use_angles:
                        # 弯曲角 (Dot Product)
                        cos_theta = np.sum(curr_dir * parent_dir, axis=0, keepdims=True)
                        cos_theta = np.clip(cos_theta, -1.0, 1.0)
                        angles_list.append(cos_theta)

                    if self.use_rotations:
                        # 旋转轴 (Cross Product) - 代表骨骼弯曲所在的平面
                        axis = np.cross(parent_dir, curr_dir, axis=0)
                        axis_norm = np.linalg.norm(axis, axis=0, keepdims=True) + 1e-6
                        axis = axis / axis_norm
                        rot_axes_list.append(axis)

            # 堆叠并添加到特征列表
            if self.use_angles:
                angles = np.stack(angles_list, axis=2)  # (1, T, 21, M)
                feature_list.append(angles)

            if self.use_rotations:
                rot_axes = np.stack(rot_axes_list, axis=2)  # (3, T, 21, M)
                feature_list.append(rot_axes)

        # 5. 拼接所有特征
        features = np.concatenate(feature_list, axis=0)
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
                 mean_map=None, std_map=None, repeat=1,
                 # [新增接口参数] 用于消融实验
                 use_sg=True, sg_window=9, sg_order=2,
                 use_angles=True, use_rotations=True):

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

        # [修改] 实例化特征提取器时传递配置参数
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

    def _resize_temporal(self, data, window_size):
        """
        同步调整时间维度 (Resizing/Padding)
        用于确保特征数据和回归目标数据的时间维度严格对齐
        """
        C, T, V, M = data.shape
        if T == window_size:
            return data

        if T < window_size:
            # Padding (Loop padding)
            pad_len = window_size - T
            # 简单重复填充
            tile_num = (pad_len // T) + 1
            padded = np.tile(data, (1, tile_num, 1, 1))
            return np.concatenate([data, padded[:, :pad_len, :, :]], axis=1)
        else:
            # Sampling (Downsample)
            # 使用均匀间隔采样，确保确定性
            ids = np.linspace(0, T - 1, window_size).astype(int)
            return data[:, ids, :, :]

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

        # [NEW] 保留用于回归的 Joint Data (Ground Truth)
        # 此时已经应用了 View Invariant 和 Spatial Augmentation，但未进行 HGT 提取和时间缩放
        # 这是为了确保回归目标与输入在空间上是一致的
        joint_data_for_regression = data_numpy.copy()

        # 3. HGT 特征提取 (包含S-G滤波和动态特征选择)
        # Input: (3, T, 22, M) -> Output: (D, T, 21, M)
        data_features = self.extractor.extract(data_numpy)

        # 4. 时间增强 (Temporal Augmentation)
        # [MODIFIED] 必须同步对 feature 和 joint_data 进行相同的时间缩放/裁剪
        # 这里使用自定义的 _resize_temporal 替代 tools.random_choose 以保证同步
        if self.window_size > 0:
            data_features = self._resize_temporal(data_features, self.window_size)
            joint_data_for_regression = self._resize_temporal(joint_data_for_regression, self.window_size)

        # 5. 归一化 (仅针对模型输入 Features)
        if self.normalization and self.mean_map is not None:
            data_features = (data_features - self.mean_map) / (self.std_map + 1e-4)

        # 数值清理 (防止 S-G 滤波或除法产生 NaN)
        data_features = np.nan_to_num(data_features, copy=False, nan=0.0, posinf=100.0, neginf=-100.0)

        # 回归目标通常不需要去中心化归一化 (或者在 loss 计算时处理)，保持原始物理尺度更直观
        joint_data_for_regression = np.nan_to_num(joint_data_for_regression, copy=False, nan=0.0)

        # [MODIFIED] 返回 (Input, Label, Index, RegressionTarget)
        # 注意: processor 必须能够解包这 4 个返回值
        return data_features, label, index, joint_data_for_regression