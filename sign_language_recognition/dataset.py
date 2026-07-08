import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d


TARGET_T = 60

# Normalization scale: all nodes now in unified coordinate frame (~±500mm range)
POS_SCALE = 100.0  # divide all positions by this -> std ~0.5-1.0


def extract_26_nodes(row):
    """Extract 26 joint positions from a 131-column row.

    Returns shape [26, 3].
    """
    nodes = np.zeros((26, 3), dtype=np.float32)
    nodes[0] = row[3:6]  # palm

    for finger_idx in range(5):
        finger_start = 6 + finger_idx * 25
        base_node = 1 + finger_idx * 5
        nodes[base_node + 0] = row[finger_start + 1: finger_start + 4]    # bone0 prev
        nodes[base_node + 1] = row[finger_start + 7: finger_start + 10]   # bone1 prev
        nodes[base_node + 2] = row[finger_start + 13: finger_start + 16]  # bone2 prev
        nodes[base_node + 3] = row[finger_start + 19: finger_start + 22]  # bone3 prev
        nodes[base_node + 4] = row[finger_start + 22: finger_start + 25]  # bone3 next (fingertip)

    return nodes


def resample_to_fixed_length(data, target_len=TARGET_T):
    """Resample temporal data to fixed length using linear interpolation.

    Args:
        data: [T_orig, ...] array
        target_len: target number of frames

    Returns:
        resampled: [target_len, ...] array
    """
    T_orig = data.shape[0]
    if T_orig == target_len:
        return data
    if T_orig == 1:
        return np.repeat(data, target_len, axis=0)

    orig_shape = data.shape[1:]
    data_flat = data.reshape(T_orig, -1)

    x_orig = np.linspace(0, 1, T_orig)
    x_new = np.linspace(0, 1, target_len)

    f = interp1d(x_orig, data_flat, axis=0, kind='linear')
    resampled = f(x_new).reshape(target_len, *orig_shape)

    return resampled.astype(np.float32)


def compute_velocity_uniform(positions, T):
    """Compute velocity on uniformly-spaced resampled frames.

    Uses central difference with dt=1 (frame-based, scale-normalized later).
    Args:
        positions: [T, 26, 3] already resampled to uniform spacing
        T: number of frames (used for dt normalization)

    Returns:
        velocities: [T, 26, 3]
    """
    vel = np.zeros_like(positions)
    if T < 2:
        return vel

    # Central difference (dt=2 frames for interior)
    vel[1:-1] = (positions[2:] - positions[:-2]) / 2.0
    # Boundary: forward/backward difference
    vel[0] = positions[1] - positions[0]
    vel[-1] = positions[-1] - positions[-2]

    return vel


def augment_rotation_y(positions, max_angle=15.0):
    """Random rotation around Y-axis."""
    angle = np.random.uniform(-max_angle, max_angle) * np.pi / 180.0
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, 0, sin_a],
                  [0, 1, 0],
                  [-sin_a, 0, cos_a]], dtype=np.float32)
    return positions @ R.T


def augment_scale(positions, low=0.9, high=1.1):
    """Random uniform scaling."""
    scale = np.random.uniform(low, high)
    return positions * scale


def augment_noise(positions, sigma=0.5):
    """Add Gaussian noise (after normalization, so sigma is in normalized units)."""
    noise = np.random.randn(*positions.shape).astype(np.float32) * sigma
    return positions + noise


def augment_time_warp(positions, sigma=0.1):
    """Random temporal warping by perturbing sample points."""
    T = positions.shape[0]
    x_orig = np.linspace(0, 1, T)
    # Perturb interior points
    warp = np.random.randn(T) * sigma
    warp[0] = 0
    warp[-1] = 0
    x_warped = x_orig + warp / T
    x_warped = np.sort(x_warped)
    # Ensure monotonic and bounded
    x_warped = np.clip(x_warped, 0, 1)

    orig_shape = positions.shape[1:]
    flat = positions.reshape(T, -1)
    f = interp1d(x_warped, flat, axis=0, kind='linear', fill_value='extrapolate')
    warped = f(x_orig).reshape(T, *orig_shape)
    return warped.astype(np.float32)


def load_session(csv_path, skip_first_n=0):
    """Load a single absolute-coordinate session CSV and return dual-hand positions.

    Rows before the first frame that contains a right hand are discarded.
    All remaining nodes from both hands are translated into a shared coordinate
    frame whose origin is the first detected right-palm position.

    Returns:
        right_pos: [T_right, 26, 3] or None
        left_pos: [T_left, 26, 3] or None
        right_ts: [T_right] timestamps in seconds
        left_ts: [T_left] timestamps in seconds
    """
    data = np.loadtxt(csv_path, delimiter=',', dtype=np.float64)
    data = np.atleast_2d(data)

    if skip_first_n > 0:
        data = data[skip_first_n:]

    if len(data) == 0:
        return None, None, None, None

    # Find the first frame where the right hand exists and use that palm as the
    # global origin for all later right/left coordinates.
    right_rows = data[data[:, 2] == 1]
    if len(right_rows) == 0:
        return None, None, None, None

    anchor_frame_id = right_rows[0, 0]
    anchor_palm = right_rows[0, 3:6].astype(np.float32)

    # Keep the entire first anchored frame (including a possible left-hand row
    # that appears before the right-hand row in file order) and everything after.
    data = data[data[:, 0] >= anchor_frame_id]

    # Split by hand type: 1=right, 0=left
    right_mask = data[:, 2] == 1
    left_mask = data[:, 2] == 0

    right_data = data[right_mask]
    left_data = data[left_mask]

    def process_hand(hand_data):
        if len(hand_data) == 0:
            return None, None
        T = hand_data.shape[0]
        ts = hand_data[:, 1] / 1e6
        positions = np.zeros((T, 26, 3), dtype=np.float32)
        for t in range(T):
            positions[t] = extract_26_nodes(hand_data[t])
        positions -= anchor_palm[None, None, :]

        return positions, ts

    right_pos, right_ts = process_hand(right_data)
    left_pos, left_ts = process_hand(left_data)

    return right_pos, left_pos, right_ts, left_ts


# Chinese to pinyin label mapping
LABEL_PINYIN = {
    '大家好': 'dajiahao',
    '你好': 'nihao',
    '忙': 'mang',
    '伤心': 'shangxin',
    '害怕': 'haipa',
    '开心': 'kaixin',
    '感动': 'gandong',
    '现在': 'xianzai',
    '未知': 'unknown',
    '去': 'qu',
    '吃': 'chi',
    '名字': 'mingzi',
    '工作': 'gongzuo',
    '来': 'lai',
    '看': 'kan',
}


def _preprocess_one_hand(positions, target_t):
    """Deterministic preprocessing: normalize + resample. Cacheable.

    Args:
        positions: [T_orig, 26, 3] or None
        target_t: target frame count

    Returns:
        [target_t, 26, 3] normalized+resampled positions, or None
    """
    if positions is None:
        return None
    positions = positions / POS_SCALE
    positions = resample_to_fixed_length(positions, target_t)
    return positions


def _finalize_one_hand(positions, target_t, augment):
    """Per-call processing: augment + velocity. Cheap, runs every __getitem__.

    Args:
        positions: [target_t, 26, 3] preprocessed positions, or None
        target_t: target frame count
        augment: whether to apply augmentation

    Returns:
        features: [6, target_t, 26]
    """
    if positions is None:
        return np.zeros((6, target_t, 26), dtype=np.float32)

    if augment:
        positions = augment_rotation_y(positions)
        positions = augment_scale(positions)
        positions = augment_time_warp(positions)
        positions = augment_noise(positions, sigma=0.02)

    velocity = compute_velocity_uniform(positions, target_t)
    velocity *= 8.0

    features = np.concatenate([positions, velocity], axis=-1)  # [T, 26, 6]
    features = features.transpose(2, 0, 1)  # [6, T, 26]
    return features


class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, split='train', train_ratio=0.8,
                 augment=True, skip_first_n=0, seed=42):
        self.augment = augment and (split == 'train')
        self.target_t = TARGET_T

        self.label_names = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        self.num_classes = len(self.label_names)
        self.label_to_idx = {name: i for i, name in enumerate(self.label_names)}
        # Pinyin labels for display/export
        self.pinyin_labels = [LABEL_PINYIN.get(n, n) for n in self.label_names]

        all_samples = []
        for label_name in self.label_names:
            label_dir = os.path.join(data_dir, label_name)
            label_idx = self.label_to_idx[label_name]
            sessions = sorted([
                s for s in os.listdir(label_dir)
                if os.path.isdir(os.path.join(label_dir, s))
            ])
            for session in sessions:
                csv_path = os.path.join(label_dir, session, 'tracking.csv')
                if os.path.exists(csv_path):
                    all_samples.append((csv_path, label_idx))

        # Train/val split by session index
        np.random.seed(seed)
        indices = np.random.permutation(len(all_samples))
        split_idx = int(len(all_samples) * train_ratio)

        if split == 'train':
            selected = indices[:split_idx]
        else:
            selected = indices[split_idx:]

        selected_samples = [all_samples[i] for i in selected]
        self.skip_first_n = skip_first_n

        # Precompute deterministic part (CSV parse + node extract + resample)
        # and cache in memory. This removes the heavy per-epoch text parsing.
        self.samples = []
        self.cache = []  # list of (right_pos [T,26,3] or None, left_pos, label)
        for csv_path, label in selected_samples:
            right_pos, left_pos, _, _ = load_session(csv_path, self.skip_first_n)
            if right_pos is None:
                continue
            right_c = _preprocess_one_hand(right_pos, self.target_t)
            left_c = _preprocess_one_hand(left_pos, self.target_t)
            self.samples.append((csv_path, label))
            self.cache.append((right_c, left_c, label))

        # For val/no-augment, the final tensor is fully deterministic -> cache it
        self.final_cache = None
        if not self.augment:
            self.final_cache = []
            for right_c, left_c, label in self.cache:
                rf = _finalize_one_hand(right_c, self.target_t, False)
                lf = _finalize_one_hand(left_c, self.target_t, False)
                feat = np.stack([rf, lf], axis=-1)  # [6, T, 26, 2]
                self.final_cache.append((torch.from_numpy(feat), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Val/no-augment: return precomputed tensor directly
        if self.final_cache is not None:
            return self.final_cache[idx]

        right_c, left_c, label = self.cache[idx]
        right_feat = _finalize_one_hand(right_c, self.target_t, self.augment)
        left_feat = _finalize_one_hand(left_c, self.target_t, self.augment)
        features = np.stack([right_feat, left_feat], axis=-1)  # [6, T, 26, 2]
        return torch.from_numpy(features), label
