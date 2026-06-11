"""
PIMBCN 数据集加载类（2026-06-10 损失函数 V2 配套版）

优化点：
1. 预解析：__init__ 中预解析所有 ast.literal_eval 字符串为 numpy 数组。
2. 零开销索引：__getitem__ 改为纯 numpy 索引 + torch.from_numpy，消除 pandas 开销。
3. 频谱对齐：预截断/补齐频谱到 2501 点后，截取 20~5000Hz 范围（1246点, 间隔4Hz）。
4. 分层抽样 (Stratified Sampling)：依据 mode 和 type 的物理组合进行严格的等比例数据划分。
5. OASPL 约束自洽：增强时基于 20~5000Hz 频谱自算 OASPL 目标值。

与 0609 版相同，仅更新文件头说明。
"""
import torch
import pandas as pd
import os
import ast
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class PIMBCNDataset(Dataset):
    def __init__(self, directory_path, input_cols=[4, 5, 6],
                 oaspl_col=11, octave_col=12, spectrum_col=13,
                 type_col=3, mode_col=2, val_split=0.2,
                 is_validation=False, norm_params=None, random_seed=42,
                 augment=True):
        self.directory_path = directory_path
        self.input_cols = input_cols
        self.oaspl_col = oaspl_col
        self.octave_col = octave_col
        self.spectrum_col = spectrum_col
        self.type_col = type_col
        self.mode_col = mode_col
        self.val_split = val_split
        self.is_validation = is_validation
        self.norm_params = norm_params
        self.random_seed = random_seed
        self.augment = augment

        self.type_encoder = LabelEncoder()
        self.mode_encoder = LabelEncoder()

        self.data = None
        self.column_names = None
        self.train_indices = None
        self.val_indices = None
        self.indices = None

        self.input_mean = None
        self.input_std = None

        self.inputs_array = None
        self.types_array = None
        self.modes_array = None
        self.oaspl_array = None
        self.octave_array = None
        self.spectrum_array = None

        self.freq_axis = np.linspace(20, 5000, 1246)

        np.random.seed(random_seed)

        if not self.is_validation:
            self._load_and_process_data()
            self._preparse_all_columns()
            self._split_data()
            self._compute_normalization_params()
        else:
            self.input_mean = self.norm_params['input_mean']
            self.input_std = self.norm_params['input_std']

    def _load_and_process_data(self):
        csv_files = [f for f in os.listdir(self.directory_path) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"目录 {self.directory_path} 中未找到CSV文件")

        dfs = [pd.read_csv(os.path.join(self.directory_path, f)) for f in csv_files]
        self.data = pd.concat(dfs, ignore_index=True)
        self.column_names = self.data.columns.tolist()
        self._validate_columns()

        type_col_name = self.data.columns[self.type_col]
        mode_col_name = self.data.columns[self.mode_col]
        self.data[type_col_name] = self.type_encoder.fit_transform(
            self.data[type_col_name])
        self.data[mode_col_name] = self.mode_encoder.fit_transform(
            self.data[mode_col_name])
        print(f"成功加载 {len(self.data)} 条数据记录")

    def _preparse_all_columns(self):
        n = len(self.data)

        self.inputs_array = self.data.iloc[:, self.input_cols].values.astype(np.float32)
        self.types_array = self.data.iloc[:, self.type_col].values.astype(np.int64)
        self.modes_array = self.data.iloc[:, self.mode_col].values.astype(np.int64)
        self.oaspl_array = self.data.iloc[:, self.oaspl_col].values.astype(np.float32)

        octave_list = []
        raw_octave = self.data.iloc[:, self.octave_col]
        for val in raw_octave:
            parsed = ast.literal_eval(val) if isinstance(val, str) else val
            octave_list.append(parsed)
        self.octave_array = np.array(octave_list, dtype=np.float32)

        spectrum_list = []
        raw_spectrum = self.data.iloc[:, self.spectrum_col]
        for val in raw_spectrum:
            parsed = ast.literal_eval(val) if isinstance(val, str) else val
            if len(parsed) > 2501:
                parsed = parsed[:2501]
            elif len(parsed) < 2501:
                parsed = parsed + [0.0] * (2501 - len(parsed))
            parsed = parsed[5:1251]  # 20~5000Hz
            spectrum_list.append(parsed)
        self.spectrum_array = np.array(spectrum_list, dtype=np.float32)

        print(f"预解析完成: inputs {self.inputs_array.shape}, "
              f"spectrum {self.spectrum_array.shape} (20~5000Hz), octave {self.octave_array.shape}")

    def _validate_columns(self):
        max_col_index = len(self.column_names) - 1
        for col in self.input_cols + [self.oaspl_col, self.octave_col,
                                      self.spectrum_col, self.type_col, self.mode_col]:
            if col < 0 or col > max_col_index:
                raise ValueError(f"列索引 {col} 超出有效范围")

    def _split_data(self):
        train_idx_list = []
        val_idx_list = []

        combo_labels = self.modes_array * 1000 + self.types_array
        unique_combos = np.unique(combo_labels)

        combo_indices = []
        combo_sizes = []
        for combo in unique_combos:
            idx_for_combo = np.where(combo_labels == combo)[0]
            np.random.shuffle(idx_for_combo)
            combo_indices.append(idx_for_combo)
            combo_sizes.append(len(idx_for_combo))

        total_samples = sum(combo_sizes)
        target_val_samples = int(total_samples * self.val_split)

        val_counts = []
        for size in combo_sizes:
            count = max(1, int(np.round(size * self.val_split)))
            count = min(count, size // 2)
            val_counts.append(count)

        total_assigned = sum(val_counts)
        if total_assigned > target_val_samples:
            ratio = target_val_samples / total_assigned
            val_counts = [max(1, int(np.round(c * ratio))) for c in val_counts]

        for idx_for_combo, n_val in zip(combo_indices, val_counts):
            n_total = len(idx_for_combo)
            if n_total == 1:
                train_idx_list.extend(idx_for_combo)
            else:
                val_idx_list.extend(idx_for_combo[:n_val])
                train_idx_list.extend(idx_for_combo[n_val:])

        self.val_indices = np.array(val_idx_list, dtype=np.int64)
        self.train_indices = np.array(train_idx_list, dtype=np.int64)
        self.indices = self.train_indices

        actual_split_ratio = len(self.val_indices) / total_samples
        print(f"分层抽样完毕: 共检测到 {len(unique_combos)} 种物理工况组合。")
        print(f"数据分布 -> 训练集: {len(self.train_indices)} 条 | 验证集: {len(self.val_indices)} 条")
        print(f"实际验证集比例: {actual_split_ratio:.2%} (目标: {self.val_split:.2%})")

    def _compute_normalization_params(self):
        if self.train_indices is None:
            self._split_data()
        train_inputs = self.inputs_array[self.train_indices]
        self.input_mean = np.mean(train_inputs, axis=0).astype(np.float32)
        self.input_std = np.std(train_inputs, axis=0).astype(np.float32)
        self.input_std = np.where(self.input_std < 1e-8, 1e-8, self.input_std)

    def get_rpm_norm_params(self):
        return float(self.input_mean[2]), float(self.input_std[2])

    def _augment_inputs(self, inputs):
        inputs = inputs + np.random.randn(*inputs.shape) * 0.02
        return inputs

    def _augment_spectrum(self, spectrum):
        """对频谱数据进行增强 (20~5000Hz 范围内自洽)"""
        spectrum = spectrum.copy()

        scaled_spec = spectrum / 10.0
        max_val = np.max(scaled_spec)
        sum_exp = np.sum(np.power(10.0, scaled_spec - max_val))
        target_oaspl = 10.0 * (np.log10(sum_exp + 1e-10) + max_val)

        spectrum = spectrum + np.random.randn(len(spectrum)) * 0.8
        scale = 1.0 + np.random.uniform(-0.08, 0.08)
        spectrum = spectrum * scale

        shift = np.random.randint(-3, 4)
        if shift != 0:
            spectrum = np.roll(spectrum, shift)
            if shift > 0:
                spectrum[:shift] = spectrum[shift]
            else:
                spectrum[shift:] = spectrum[shift-1]

        if np.random.random() > 0.5:
            window_size = np.random.randint(3, 7)
            window = np.hanning(window_size)
            window = window / window.sum()
            spectrum = np.convolve(spectrum, window, mode='same')

        scaled_spec = spectrum / 10.0
        max_val = np.max(scaled_spec)
        sum_exp = np.sum(np.power(10.0, scaled_spec - max_val))
        new_oaspl = 10.0 * (np.log10(sum_exp + 1e-10) + max_val)
        oaspl_scale = target_oaspl / new_oaspl
        spectrum = spectrum * oaspl_scale

        return spectrum

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.data)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]

        inp = (self.inputs_array[data_idx] - self.input_mean) / self.input_std
        spectrum = self.spectrum_array[data_idx]

        if self.augment and not self.is_validation:
            inp = self._augment_inputs(inp)
            spectrum = self._augment_spectrum(spectrum)

        return (
            torch.from_numpy(inp),
            torch.tensor(self.types_array[data_idx], dtype=torch.int64),
            torch.tensor(self.modes_array[data_idx], dtype=torch.int64),
            torch.tensor(self.oaspl_array[data_idx], dtype=torch.float32).unsqueeze(0),
            torch.from_numpy(self.octave_array[data_idx]),
            torch.from_numpy(spectrum),
        )

    def get_validation_dataset(self):
        if self.is_validation:
            raise ValueError("当前实例已是验证集")
        val_dataset = PIMBCNDataset(
            directory_path=self.directory_path, input_cols=self.input_cols,
            oaspl_col=self.oaspl_col, octave_col=self.octave_col, spectrum_col=self.spectrum_col,
            type_col=self.type_col, mode_col=self.mode_col, val_split=self.val_split,
            is_validation=True, norm_params={'input_mean': self.input_mean, 'input_std': self.input_std},
            random_seed=self.random_seed,
            augment=False
        )
        # 直接共享训练集的 numpy 数组，避免重复加载 CSV
        val_dataset.type_encoder = self.type_encoder
        val_dataset.mode_encoder = self.mode_encoder
        val_dataset.data = self.data
        val_dataset.column_names = self.column_names
        val_dataset.inputs_array = self.inputs_array
        val_dataset.types_array = self.types_array
        val_dataset.modes_array = self.modes_array
        val_dataset.oaspl_array = self.oaspl_array
        val_dataset.octave_array = self.octave_array
        val_dataset.spectrum_array = self.spectrum_array
        val_dataset.val_indices = self.val_indices
        val_dataset.indices = self.val_indices
        return val_dataset
