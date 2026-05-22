"""
PIMBCN 数据集加载类（2026-05-21 优化版）

优化点：
- __init__ 中预解析所有 ast.literal_eval 字符串为 numpy 数组
- __getitem__ 改为纯 numpy 索引 + torch.from_numpy，消除 pandas iloc 和 ast 解析开销
- 预截断/补齐频谱到 2501 点
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
                 is_validation=False, norm_params=None, random_seed=42):
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

        self.type_encoder = LabelEncoder()
        self.mode_encoder = LabelEncoder()

        self.data = None
        self.column_names = None
        self.train_indices = None
        self.val_indices = None
        self.indices = None

        self.input_mean = None
        self.input_std = None

        # 预解析后存储的 numpy 数组
        self.inputs_array = None
        self.types_array = None
        self.modes_array = None
        self.oaspl_array = None
        self.octave_array = None
        self.spectrum_array = None

        np.random.seed(random_seed)
        self._load_and_process_data()
        self._preparse_all_columns()

        if not self.is_validation:
            self._split_data()
            self._compute_normalization_params()
        else:
            self.input_mean = self.norm_params['input_mean']
            self.input_std = self.norm_params['input_std']

    # ================= 数据加载 =================
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

    # ================= 核心优化：预解析所有列到 numpy =================
    def _preparse_all_columns(self):
        n = len(self.data)

        self.inputs_array = self.data.iloc[:, self.input_cols].values.astype(np.float32)
        self.types_array = self.data.iloc[:, self.type_col].values.astype(np.int64)
        self.modes_array = self.data.iloc[:, self.mode_col].values.astype(np.int64)
        self.oaspl_array = self.data.iloc[:, self.oaspl_col].values.astype(np.float32)

        # 预解析 octave 列
        octave_list = []
        raw_octave = self.data.iloc[:, self.octave_col]
        for val in raw_octave:
            parsed = ast.literal_eval(val) if isinstance(val, str) else val
            octave_list.append(parsed)
        self.octave_array = np.array(octave_list, dtype=np.float32)

        # 预解析 spectrum 列（含截断/补齐到 2501）
        spectrum_list = []
        raw_spectrum = self.data.iloc[:, self.spectrum_col]
        for val in raw_spectrum:
            parsed = ast.literal_eval(val) if isinstance(val, str) else val
            if len(parsed) > 2501:
                parsed = parsed[:2501]
            elif len(parsed) < 2501:
                parsed = parsed + [0.0] * (2501 - len(parsed))
            spectrum_list.append(parsed)
        self.spectrum_array = np.array(spectrum_list, dtype=np.float32)

        print(f"预解析完成: inputs {self.inputs_array.shape}, "
              f"spectrum {self.spectrum_array.shape}, octave {self.octave_array.shape}")

    def _validate_columns(self):
        max_col_index = len(self.column_names) - 1
        for col in self.input_cols + [self.oaspl_col, self.octave_col,
                                       self.spectrum_col, self.type_col, self.mode_col]:
            if col < 0 or col > max_col_index:
                raise ValueError(f"列索引 {col} 超出有效范围")

    # ================= 数据划分与归一化 =================
    def _split_data(self):
        n_samples = len(self.data)
        n_val = int(n_samples * self.val_split)
        self.val_indices = np.random.choice(n_samples, size=n_val, replace=False)
        self.train_indices = np.setdiff1d(np.arange(n_samples), self.val_indices)
        self.indices = self.train_indices

    def _compute_normalization_params(self):
        if self.train_indices is None:
            self._split_data()
        train_inputs = self.inputs_array[self.train_indices]
        self.input_mean = np.mean(train_inputs, axis=0).astype(np.float32)
        self.input_std = np.std(train_inputs, axis=0).astype(np.float32)
        self.input_std = np.where(self.input_std < 1e-8, 1e-8, self.input_std)

    def get_rpm_norm_params(self):
        return float(self.input_mean[2]), float(self.input_std[2])

    # ================= 核心优化：纯 numpy 索引，零 pandas 开销 =================
    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.data)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]

        inp = (self.inputs_array[data_idx] - self.input_mean) / self.input_std

        return (
            torch.from_numpy(inp),
            torch.tensor(self.types_array[data_idx], dtype=torch.int64),
            torch.tensor(self.modes_array[data_idx], dtype=torch.int64),
            torch.tensor(self.oaspl_array[data_idx], dtype=torch.float32).unsqueeze(0),
            torch.from_numpy(self.octave_array[data_idx]),
            torch.from_numpy(self.spectrum_array[data_idx]),
        )

    # ================= 验证集构建 =================
    def get_validation_dataset(self):
        if self.is_validation:
            raise ValueError("当前实例已是验证集")
        val_dataset = PIMBCNDataset(
            directory_path=self.directory_path, input_cols=self.input_cols,
            oaspl_col=self.oaspl_col, octave_col=self.octave_col, spectrum_col=self.spectrum_col,
            type_col=self.type_col, mode_col=self.mode_col, val_split=self.val_split,
            is_validation=True, norm_params={'input_mean': self.input_mean, 'input_std': self.input_std},
            random_seed=self.random_seed
        )
        val_dataset.type_encoder = self.type_encoder
        val_dataset.mode_encoder = self.mode_encoder
        val_dataset.data = self.data
        val_dataset.val_indices = self.val_indices
        val_dataset.indices = self.val_indices
        val_dataset.column_names = self.column_names
        return val_dataset
