import torch
import pandas as pd
import os
import ast
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class PIMBCNDataset(Dataset):
    """
    PI-MBCN 网络数据集加载类
    支持三个输出：总声压级 (OASPL), 28维三分之一倍频程, 2501维窄带声压级曲线
    """
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
        
        np.random.seed(random_seed)
        self.load_and_process_data()
        
        if not self.is_validation:
            self.split_data()
            self.compute_normalization_params()
        else:
            self.input_mean = self.norm_params['input_mean']
            self.input_std = self.norm_params['input_std']
    
    def load_and_process_data(self):
        try:
            csv_files = [f for f in os.listdir(self.directory_path) if f.endswith('.csv')]
            if not csv_files:
                raise ValueError(f"目录 {self.directory_path} 中未找到CSV文件")
            
            dfs = [pd.read_csv(os.path.join(self.directory_path, f)) for f in csv_files]
            self.data = pd.concat(dfs, ignore_index=True)
            self.column_names = self.data.columns.tolist()
            
            self._validate_columns()
            
            self.data.iloc[:, self.type_col] = self.type_encoder.fit_transform(self.data.iloc[:, self.type_col])
            self.data.iloc[:, self.mode_col] = self.mode_encoder.fit_transform(self.data.iloc[:, self.mode_col])
            print(f"成功加载 {len(self.data)} 条数据记录")
        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            raise
    
    def split_data(self):
        n_samples = len(self.data)
        n_val = int(n_samples * self.val_split)
        self.val_indices = np.random.choice(n_samples, size=n_val, replace=False)
        self.train_indices = np.setdiff1d(np.arange(n_samples), self.val_indices)
        self.indices = self.train_indices
    
    def compute_normalization_params(self):
        if self.train_indices is None:
            self.split_data()
        train_inputs = self.data.iloc[self.train_indices, self.input_cols].values
        self.input_mean = np.mean(train_inputs, axis=0)
        self.input_std = np.std(train_inputs, axis=0)
        self.input_std = np.where(self.input_std < 1e-8, 1e-8, self.input_std)
    
    def _validate_columns(self):
        max_col_index = len(self.column_names) - 1
        for col in self.input_cols + [self.oaspl_col, self.octave_col, self.spectrum_col, self.type_col, self.mode_col]:
            if col < 0 or col > max_col_index:
                raise ValueError(f"列索引 {col} 超出有效范围")
    
    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.data)
    
    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        input_sample = self.data.iloc[data_idx, self.input_cols].values.astype(float)
        
        # 应用标准归一化
        input_normalized = (input_sample - self.input_mean) / self.input_std
        
        type_sample = self.data.iloc[data_idx, self.type_col]
        mode_sample = self.data.iloc[data_idx, self.mode_col]
        oaspl_sample = self.data.iloc[data_idx, self.oaspl_col]
        
        octave_sample = self.data.iloc[data_idx, self.octave_col]
        octave_list = ast.literal_eval(octave_sample) if isinstance(octave_sample, str) else octave_sample
        
        spectrum_sample = self.data.iloc[data_idx, self.spectrum_col]
        spectrum_list = ast.literal_eval(spectrum_sample) if isinstance(spectrum_sample, str) else spectrum_sample
        
        # 补齐或截断
        if len(spectrum_list) > 2501:
            spectrum_list = spectrum_list[:2501]
        elif len(spectrum_list) < 2501:
            spectrum_list = spectrum_list + [0.0] * (2501 - len(spectrum_list))
            
        return (torch.tensor(input_normalized, dtype=torch.float32),
                torch.tensor(type_sample, dtype=torch.int64),
                torch.tensor(mode_sample, dtype=torch.int64),
                torch.tensor(oaspl_sample, dtype=torch.float32).unsqueeze(0),
                torch.tensor(octave_list, dtype=torch.float32),
                torch.tensor(spectrum_list, dtype=torch.float32))
    
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

    # ================= 新增：提取 RPM 反归一化参数的方法 =================
    def get_rpm_norm_params(self):
        """假设RPM是 input_cols 的第3个元素 (索引为2)"""
        return self.input_mean[2], self.input_std[2]