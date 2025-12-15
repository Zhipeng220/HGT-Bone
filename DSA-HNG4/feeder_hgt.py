"""
feeder_hgt.py - Final Enhanced HGT-Bone Feeder v2.3 (Patch 3)
============================================================
综合Claude+Grok二轮审核 + Patch 3 深度优化

新增功能:
1. Mixup数据增强
2. 自动归一化计算 (Patch 3: 增加样本量)
3. 更完善的边界检查
4. 改进的tools fallback
5. 渐进式数据增强 (Progressive Augmentation)
============================================================
"""

import sys
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

try:
    from . import tools
except ImportError:
    try:
        import tools
    except ImportError:
        # ✅ 改进的Fallback tools
        class tools:
            @staticmethod
            def random_move(data_numpy):
                """随机平移 - 更完整实现"""
                C, T, V, M = data_numpy.shape
                data_shifted = data_numpy.copy()
                for m in range(M):
                    # 随机平移量，考虑数据范围
                    data_range = np.abs(data_numpy[:, :, :, m]).max() + 1e-6
                    offset = np.random.uniform(-0.1, 0.1, size=(3, 1, 1)) * data_range
                    data_shifted[:, :, :, m] = data_numpy[:, :, :, m] + offset
                return data_shifted

            @staticmethod
            def random_rot(data_numpy):
                """随机旋转 - 三轴旋转"""
                C, T, V, M = data_numpy.shape

                # 随机旋转角度 (-30°到30°)
                angles = np.random.uniform(-np.pi/6, np.pi/6, size=3)

                # 构建旋转矩阵 (ZYX顺序)
                cx, cy, cz = np.cos(angles)
                sx, sy, sz = np.sin(angles)

                Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
                Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
                Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

                R = Rz @ Ry @ Rx

                data_rot = np.zeros_like(data_numpy)
                for m in range(M):
                    for t in range(T):
                        data_rot[:, t, :, m] = R @ data_numpy[:, t, :, m]
                return data_rot


class HGTFeatureExtractorV2:
    """
    Enhanced HGT-Bone 特征提取器 v2.2
    输出维度: 10 (原8 + 速度模长1 + 加速度模长1)
    """

    def __init__(self, use_sg=True, sg_window=9, sg_order=2,
                 use_angles=True, use_rotations=True, use_velocity=True):
        self.use_sg = use_sg
        self.sg_window = sg_window
        self.sg_order = sg_order
        self.use_angles = use_angles
        self.use_rotations = use_rotations
        self.use_velocity = use_velocity

        # SHREC 2017 骨骼连接定义
        self.bone_pairs = (
            (0, 1),
            (1, 2), (2, 3), (3, 4), (4, 5),
            (1, 6), (6, 7), (7, 8), (8, 9),
            (1, 10), (10, 11), (11, 12), (12, 13),
            (1, 14), (14, 15), (15, 16), (16, 17),
            (1, 18), (18, 19), (19, 20), (20, 21)
        )
        self.num_bones = len(self.bone_pairs)
        self.bone_parents = self._build_bone_tree()

    def _build_bone_tree(self):
        """构建骨骼树"""
        parents = [-1] * self.num_bones
        for i, (j1, j2) in enumerate(self.bone_pairs):
            if i == 0:
                continue
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

        effective_order = min(order, effective_window - 1)

        for m in range(M):
            for v in range(V):
                for c in range(C):
                    try:
                        filtered[c, :, v, m] = savgol_filter(
                            data[c, :, v, m],
                            window_length=effective_window,
                            polyorder=effective_order
                        )
                    except ValueError:
                        filtered[c, :, v, m] = data[c, :, v, m]
        return filtered

    def extract(self, joint_coords):
        """
        提取HGT特征

        Input: (3, T, 22, M)
        Output: (D, T, 21, M), D=10
        """
        if self.use_sg:
            joints = self._denoise_sg(joint_coords)
        else:
            joints = joint_coords

        C, T, V_joint, M = joints.shape

        # 骨骼向量
        bone_vecs_list = []
        for v1, v2 in self.bone_pairs:
            if v1 < V_joint and v2 < V_joint:
                vec = joints[:, :, v1, :] - joints[:, :, v2, :]
            else:
                vec = np.zeros((C, T, M))
            bone_vecs_list.append(vec)
        bone_vecs = np.stack(bone_vecs_list, axis=2)

        # 模长 & 方向
        bone_lens = np.linalg.norm(bone_vecs, axis=0, keepdims=True) + 1e-6
        bone_dirs = bone_vecs / bone_lens

        feature_list = [bone_vecs, bone_lens]

        # 角度和旋转轴
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

        # 运动特征
        if self.use_velocity:
            bone_velocity = np.zeros_like(bone_vecs)
            if T > 1:
                bone_velocity[:, 1:, :, :] = bone_vecs[:, 1:, :, :] - bone_vecs[:, :-1, :, :]

            bone_accel = np.zeros_like(bone_vecs)
            if T > 2:
                bone_accel[:, 2:, :, :] = bone_velocity[:, 2:, :, :] - bone_velocity[:, 1:-1, :, :]

            vel_norm = np.linalg.norm(bone_velocity, axis=0, keepdims=True)
            acc_norm = np.linalg.norm(bone_accel, axis=0, keepdims=True)

            feature_list.append(vel_norm)
            feature_list.append(acc_norm)

        features = np.concatenate(feature_list, axis=0)
        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

        return features.astype(np.float32)


class Feeder(Dataset):
    """
    Enhanced HGT-Bone 数据加载器 v2.3

    新增:
    - Mixup支持
    - 自动归一化
    - 改进的增强
    - 渐进式增强 (Progressive Augmentation)
    """

    def __init__(self, data_path, label_path, split='train',
                 random_choose=False, random_shift=False, random_move=False,
                 random_rot=False, window_size=-1, normalization=False,
                 debug=False, use_mmap=True, bone=False, vel=False,
                 mean_map=None, std_map=None, repeat=1,
                 use_sg=True, sg_window=9, sg_order=2,
                 use_angles=True, use_rotations=True, use_velocity=True,
                 noise_std=0.0, joint_drop_prob=0.0, time_warp_prob=0.0,
                 use_mixup=False, mixup_alpha=0.2,
                 # ✅ Patch 3: 新增渐进式增强参数
                 progressive_augment=False, warmup_epochs=5):

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
        self.repeat = repeat

        # 增强参数（保存原始强度）
        self.noise_std_full = noise_std
        self.joint_drop_prob_full = joint_drop_prob
        self.time_warp_prob_full = time_warp_prob

        # ✅ Patch 3: 渐进式增强控制
        self.progressive_augment = progressive_augment and (split == 'train')
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0  # 由外部设置

        # 当前实际使用的增强强度（会动态调整）
        self.noise_std = noise_std
        self.joint_drop_prob = joint_drop_prob
        self.time_warp_prob = time_warp_prob

        # Mixup参数
        self.use_mixup = use_mixup and (split == 'train')
        self.mixup_alpha = mixup_alpha

        self.load_data()
        self.extractor = HGTFeatureExtractorV2(
            use_sg=use_sg, sg_window=sg_window, sg_order=sg_order,
            use_angles=use_angles, use_rotations=use_rotations, use_velocity=use_velocity
        )

        # 归一化统计量
        self.mean_map = mean_map
        self.std_map = std_map

        if normalization and mean_map is None and split == 'train':
            self._compute_normalization_stats()

    def load_data(self):
        """加载数据"""
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

    def _compute_normalization_stats(self):
        """✅ Patch 3: 改进版，使用更多样本计算归一化统计量"""
        print("[INFO] Computing normalization statistics...")

        # ✅ Patch 3: 使用更多样本（从500增加到2000）
        sample_size = min(2000, len(self.label))

        # ✅ Patch 3: 如果数据量充足，至少使用20%的数据
        if len(self.label) > 5000:
            sample_size = min(len(self.label) // 5, 3000)

        indices = np.random.choice(len(self.label), sample_size, replace=False)

        all_features = []
        for idx in indices:
            data_numpy = np.array(self.data[idx])
            data_numpy = self.view_invariant_transform(data_numpy)
            features = self.extractor.extract(data_numpy)
            all_features.append(features)

        all_features = np.stack(all_features, axis=0)  # (N, C, T, V, M)

        # 计算每个通道的统计量
        self.mean_map = np.mean(all_features, axis=(0, 2, 4), keepdims=True)
        self.std_map = np.std(all_features, axis=(0, 2, 4), keepdims=True) + 1e-4

        # 调整形状以匹配特征
        self.mean_map = self.mean_map.squeeze(0).squeeze(-1)  # (C, 1, V)
        self.std_map = self.std_map.squeeze(0).squeeze(-1)

        print(f"[INFO] Normalization computed from {sample_size} samples")
        print(f"[INFO] Mean range: [{self.mean_map.min():.4f}, {self.mean_map.max():.4f}]")
        print(f"[INFO] Std range: [{self.std_map.min():.4f}, {self.std_map.max():.4f}]")

    def set_epoch(self, epoch):
        """
        ✅ Patch 3: 设置当前epoch，动态调整增强强度

        渐进式增强策略：
        - Epoch 0-warmup_epochs: 使用 50% 强度
        - Epoch warmup_epochs+: 线性增加到 100% 强度
        """
        self.current_epoch = epoch

        if not self.progressive_augment or self.split != 'train':
            return

        if epoch < self.warmup_epochs:
            # Warmup阶段：50%强度
            scale = 0.5
            # 避免日志过于频繁，可以考虑只在epoch开始时打印，这里为了明确逻辑保留逻辑
        elif epoch < self.warmup_epochs + 10:
            # 渐进增强阶段：50% -> 100% (10个epoch)
            progress = (epoch - self.warmup_epochs) / 10.0
            scale = 0.5 + 0.5 * progress
        else:
            # 完全增强阶段：100%强度
            scale = 1.0

        # 动态调整增强参数
        self.noise_std = self.noise_std_full * scale
        self.joint_drop_prob = self.joint_drop_prob_full * scale
        self.time_warp_prob = self.time_warp_prob_full * scale

        if epoch % 5 == 0 and epoch <= self.warmup_epochs + 11:
            print(f"[Augment] Epoch {epoch}: Augmentation strength at {scale*100:.0f}%")

    def view_invariant_transform(self, data_numpy):
        """视角不变性变换"""
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

                v2 = frame_data[:, 6] if V > 6 else frame_data[:, 2]
                v2 = v2 / (np.linalg.norm(v2) + 1e-6)

                z_axis = np.cross(v1, v2)
                z_norm = np.linalg.norm(z_axis)
                if z_norm < 1e-6:
                    z_axis = np.array([0, 0, 1])
                else:
                    z_axis = z_axis / z_norm

                x_axis = np.cross(v1, z_axis)
                x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)

                R = np.stack([x_axis, v1, z_axis])
                data_aligned[:, t, :, m] = np.dot(R, frame_data)

        return data_aligned

    def __len__(self):
        return len(self.label) * self.repeat

    # =========================================================
    # 数据增强方法
    # =========================================================

    def _add_gaussian_noise(self, data, std=0.01):
        if std <= 0:
            return data
        noise = np.random.randn(*data.shape).astype(np.float32) * std
        return data + noise

    def _joint_dropout(self, data, drop_prob=0.1):
        if drop_prob <= 0:
            return data

        C, T, V, M = data.shape
        data_dropped = data.copy()
        mask = np.random.random((V,)) > drop_prob

        # 保护关键关节
        protected_joints = [0, 1, 2, 6, 10, 14, 18]
        for j in protected_joints:
            if j < V:
                mask[j] = True

        for v in range(V):
            if not mask[v]:
                data_dropped[:, :, v, :] = 0

        return data_dropped

    def _time_warp(self, data, sigma=0.15, knot=4):
        C, T, V, M = data.shape

        if T < 10:
            return data

        orig_steps = np.arange(T)
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
        random_warps = np.clip(random_warps, 0.5, 1.5)
        random_warps[0] = 1.0
        random_warps[-1] = 1.0

        warp_steps = np.linspace(0, T - 1, num=knot + 2)
        cum_warps = np.cumsum(random_warps)
        cum_warps = cum_warps / cum_warps[-1] * (T - 1)

        try:
            time_warp = interp1d(warp_steps, cum_warps, kind='linear')
            warped_steps = time_warp(orig_steps)
            warped_steps = np.clip(warped_steps, 0, T - 1)
        except Exception:
            return data

        data_warped = np.zeros_like(data)
        for c in range(C):
            for v in range(V):
                for m in range(M):
                    try:
                        interp_func = interp1d(
                            orig_steps, data[c, :, v, m],
                            kind='linear', fill_value='extrapolate'
                        )
                        data_warped[c, :, v, m] = interp_func(warped_steps)
                    except Exception:
                        data_warped[c, :, v, m] = data[c, :, v, m]

        return data_warped

    def _apply_temporal_shift(self, data, max_shift_ratio=0.2):
        C, T, V, M = data.shape
        max_shift = int(T * max_shift_ratio)

        if max_shift == 0:
            return data

        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return data

        shifted_data = np.zeros_like(data)

        if shift > 0:
            shifted_data[:, shift:, :, :] = data[:, :-shift, :, :]
            shifted_data[:, :shift, :, :] = data[:, 0:1, :, :]
        else:
            shift = abs(shift)
            shifted_data[:, :-shift, :, :] = data[:, shift:, :, :]
            shifted_data[:, -shift:, :, :] = data[:, -1:, :, :]

        return shifted_data

    def _get_mixup_sample(self):
        """✅ 获取Mixup配对样本"""
        mix_idx = np.random.randint(0, len(self.label))
        mix_data = np.array(self.data[mix_idx])
        mix_label = self.label[mix_idx]
        return mix_data, mix_label

    def __getitem__(self, index):
        """获取单个样本"""
        index = index % len(self.label)
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # 1. 视角对齐
        data_numpy = self.view_invariant_transform(data_numpy)

        # 2. 数据增强 (仅训练时)
        if self.split == 'train':
            # 高斯噪声 (使用动态调整后的 self.noise_std)
            if self.noise_std > 0:
                data_numpy = self._add_gaussian_noise(data_numpy, std=self.noise_std)

            # 关节丢弃 (使用动态调整后的 self.joint_drop_prob)
            if self.joint_drop_prob > 0 and np.random.random() < 0.3:
                data_numpy = self._joint_dropout(data_numpy, drop_prob=self.joint_drop_prob)

            # 时序扭曲 (使用动态调整后的 self.time_warp_prob)
            if self.time_warp_prob > 0 and np.random.random() < self.time_warp_prob:
                data_numpy = self._time_warp(data_numpy, sigma=0.15)

            # 空间增强
            if self.random_move:
                data_numpy = tools.random_move(data_numpy)
            if self.random_rot:
                data_numpy = tools.random_rot(data_numpy)

            # 时序平移
            if self.random_shift:
                data_numpy = self._apply_temporal_shift(data_numpy, max_shift_ratio=0.2)

        # 保留回归数据
        joint_data_for_regression = data_numpy.copy()

        # 3. HGT特征提取
        data_features = self.extractor.extract(data_numpy)

        # 4. 时序采样
        if self.window_size > 0:
            C, T, V, M = data_features.shape
            if self.random_choose and T > self.window_size and self.split == 'train':
                start = np.random.randint(0, T - self.window_size + 1)
                data_features = data_features[:, start:start + self.window_size, :, :]
                joint_data_for_regression = joint_data_for_regression[:, start:start + self.window_size, :, :]
            elif T != self.window_size:
                ids = np.linspace(0, T - 1, self.window_size).astype(int)
                data_features = data_features[:, ids, :, :]
                joint_data_for_regression = joint_data_for_regression[:, ids, :, :]

        # 5. 归一化 (Patch 3: 修复维度广播)
        if self.normalization and self.mean_map is not None:
            # data_features: (C, T, V, M)
            # mean_map/std_map: (C, 1, V) -> need (C, 1, V, 1) for M broadcasting
            if len(self.mean_map.shape) == 3:  # (C, 1, V)
                mean = self.mean_map[:, :, :, np.newaxis]  # -> (C, 1, V, 1)
                std = self.std_map[:, :, :, np.newaxis]
            else:
                mean = self.mean_map
                std = self.std_map

            data_features = (data_features - mean) / (std + 1e-4)

        # 6. 数值清理
        data_features = np.nan_to_num(data_features, copy=False, nan=0.0,
                                      posinf=100.0, neginf=-100.0)
        joint_data_for_regression = np.nan_to_num(joint_data_for_regression,
                                                  copy=False, nan=0.0)

        # 7. 转换为Bone向量
        bone_data_for_regression = self._convert_joint_to_bone(joint_data_for_regression)

        # ✅ 8. Mixup (返回额外信息供训练时使用)
        if self.use_mixup and self.split == 'train':
            # 生成mixup系数
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            return data_features, label, index, bone_data_for_regression, lam

        return data_features, label, index, bone_data_for_regression

    def _convert_joint_to_bone(self, joint_data):
        """将关节坐标转换为骨骼向量"""
        C, T, V, M = joint_data.shape
        bone_vecs_list = []

        for v1, v2 in self.extractor.bone_pairs:
            if v1 < V and v2 < V:
                vec = joint_data[:, :, v1, :] - joint_data[:, :, v2, :]
            else:
                vec = np.zeros((C, T, M))
            bone_vecs_list.append(vec)

        bone_data = np.stack(bone_vecs_list, axis=2)
        return bone_data

    def collate_fn_mixup(self, batch):
        """✅ Mixup collate函数"""
        if len(batch[0]) == 5:  # 有mixup
            data, label, index, bone, lam = zip(*batch)

            data = torch.from_numpy(np.stack(data))
            label = torch.tensor(label)
            bone = torch.from_numpy(np.stack(bone))
            lam = torch.tensor(lam)

            # 打乱顺序用于mixup
            batch_size = data.size(0)
            perm = torch.randperm(batch_size)

            # 混合数据
            lam = lam.view(-1, 1, 1, 1, 1).float()
            mixed_data = lam * data + (1 - lam) * data[perm]

            return mixed_data, label, label[perm], lam.squeeze(), index, bone
        else:
            data, label, index, bone = zip(*batch)
            return (torch.from_numpy(np.stack(data)),
                    torch.tensor(label),
                    index,
                    torch.from_numpy(np.stack(bone)))