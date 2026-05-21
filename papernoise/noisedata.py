import torch
import pandas as pd
import os
import ast
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import math
# 总声压级预测数据集
class SPLdata(Dataset):
    def __init__(self, directory_path, input_cols=[4, 5, 6],
                 output_col=11, type_col=3, mode_col=2, val_split=0.2, 
                 is_validation=False, norm_params=None, random_seed=42):
        """
        非线性类型绑定数据集，支持训练/验证集划分和标准归一化
        
        参数:
            directory_path: CSV文件所在目录路径
            input_cols: 输入特征列索引列表
            output_col: 输出序列列索引
            type_col: 空调型号型列索引
            mode_col: 模式类型列索引
            val_split: 验证集比例 (仅在非验证集实例中有效)
            is_validation: 是否为验证集
            norm_params: 归一化参数字典 (mean, std)，验证集需从训练集传递
            random_seed: 随机种子，确保可重现性
        """
        self.directory_path = directory_path
        self.input_cols = input_cols
        self.output_col = output_col
        self.type_col = type_col
        self.mode_col = mode_col
        self.val_split = val_split
        self.is_validation = is_validation
        self.norm_params = norm_params
        self.random_seed = random_seed
        
        # 初始化编码器
        self.type_encoder = LabelEncoder()
        self.mode_encoder = LabelEncoder()
        
        # 初始化数据相关变量
        self.data = None
        self.column_names = None
        self.output_length = None
        self.train_indices = None
        self.val_indices = None
        self.indices = None  # 当前使用的索引（训练或验证）
        
        # 归一化参数
        self.input_mean = None
        self.input_std = None
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 加载并处理数据
        self.load_and_concat_csv()
        
        # 划分数据集
        if not self.is_validation:
            self.split_data()
            self.compute_normalization_params()
        else:
            # 验证集使用训练集传递的归一化参数
            self.input_mean = self.norm_params['input_mean']
            self.input_std = self.norm_params['input_std']
    
    def load_and_concat_csv(self):
        """加载并拼接目录下所有CSV文件，执行数据预处理"""
        try:
            # 获取目录下所有CSV文件
            csv_files = [f for f in os.listdir(self.directory_path) if f.endswith('.csv')]
            
            if not csv_files:
                raise ValueError(f"目录 {self.directory_path} 中未找到CSV文件")
            
            # 读取并拼接所有CSV文件
            dfs = []
            for file in csv_files:
                file_path = os.path.join(self.directory_path, file)
                df = pd.read_csv(file_path)
                dfs.append(df)
            
            # 拼接所有数据框
            self.data = pd.concat(dfs, ignore_index=True)
            
            # 获取列名并打印
            self.column_names = self.data.columns.tolist()
            print("列名:", self.column_names)
            
            # 验证列索引是否有效
            self._validate_columns()
            
            # 打印输出列列表的长度
            # sample_output = self.data.iloc[0, self.output_col]
            # sample_list = ast.literal_eval(sample_output) if isinstance(sample_output, str) else sample_output
            # self.output_length = len(sample_list)
            # print(f"输出列列表的长度: {self.output_length}")
            
            # 对空调型号和模式列进行标签编码
            self.data.iloc[:, self.type_col] = self.type_encoder.fit_transform(self.data.iloc[:, self.type_col])
            self.data.iloc[:, self.mode_col] = self.mode_encoder.fit_transform(self.data.iloc[:, self.mode_col])
            print(f"空调型号编码: {dict(zip(self.type_encoder.classes_, self.type_encoder.transform(self.type_encoder.classes_)))}")
            print(f"模式类型编码: {dict(zip(self.mode_encoder.classes_, self.mode_encoder.transform(self.mode_encoder.classes_)))}")
            
            print(f"成功加载 {len(self.data)} 条数据记录")
            
        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            raise
    
    def split_data(self):
        """将数据划分为训练集和验证集"""
        n_samples = len(self.data)
        n_val = int(n_samples * self.val_split)
        
        # 随机选择验证集索引
        self.val_indices = np.random.choice(n_samples, size=n_val, replace=False)
        self.train_indices = np.setdiff1d(np.arange(n_samples), self.val_indices)
        
        # 设置当前使用的索引（训练集）
        self.indices = self.train_indices
        
        print(f"数据集划分完成 - 训练集: {len(self.train_indices)} 样本, 验证集: {len(self.val_indices)} 样本")
    
    def compute_normalization_params(self):
        """使用训练集计算输入特征的均值和标准差（标准归一化）"""
        if self.train_indices is None:
            self.split_data()
            
        # 获取训练集输入特征
        train_inputs = self.data.iloc[self.train_indices, self.input_cols].values
        
        # 计算均值和标准差
        self.input_mean = np.mean(train_inputs, axis=0)
        self.input_std = np.std(train_inputs, axis=0)
        
        # 添加微小值避免除零错误
        self.input_std = np.where(self.input_std < 1e-8, 1e-8, self.input_std)
        
        print("输入特征归一化参数计算完成")
        print(f"输入特征均值: {self.input_mean}")
        print(f"输入特征标准差: {self.input_std}")
    
    def _validate_columns(self):
        """验证指定的列索引是否有效"""
        max_col_index = len(self.column_names) - 1
        
        # 检查输入列
        for col in self.input_cols:
            if col < 0 or col > max_col_index:
                raise ValueError(f"输入列索引 {col} 超出有效范围 (0~{max_col_index})")
        
        # 检查其他列
        for col in [self.output_col, self.type_col, self.mode_col]:
            if col < 0 or col > max_col_index:
                raise ValueError(f"列索引 {col} 超出有效范围 (0~{max_col_index})")
    
    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.data)
    
    def __getitem__(self, idx):
        """获取指定索引的样本数据（应用归一化）"""
        if idx < 0 or idx >= len(self.indices):
            raise IndexError(f"样本索引 {idx} 超出范围")
        
        try:
            # 获取原始数据索引
            data_idx = self.indices[idx]
            
            # 获取输入特征
            input_sample = self.data.iloc[data_idx, self.input_cols].values.astype(float)
            
            # 应用标准归一化
            input_normalized = (input_sample - self.input_mean) / self.input_std
            
            # 获取碗类型和模式类型
            type_sample = self.data.iloc[data_idx, self.type_col]
            mode_sample = self.data.iloc[data_idx, self.mode_col]
            
            # 获取输出序列并解析
            output_sample = self.data.iloc[data_idx, self.output_col]
            # output_list = ast.literal_eval(output_sample) if isinstance(output_sample, str) else output_sample
            
            # 转换为PyTorch张量
            input_tensor = torch.tensor(input_normalized, dtype=torch.float32)
            output_tensor = torch.tensor(output_sample, dtype=torch.float32)
            # 转换为PyTorch张量
            type_tensor = torch.tensor(type_sample, dtype=torch.int64)
            mode_tensor = torch.tensor(mode_sample, dtype=torch.int64)
            
            # # 验证输出长度
            # if len(output_tensor) != self.output_length:
            #     raise ValueError(f"输出序列长度不匹配: 预期 {self.output_length}, 实际 {len(output_tensor)}")
            
            return input_tensor, output_tensor, type_tensor, mode_tensor
            
        except Exception as e:
            print(f"获取样本 {idx} 时出错: {str(e)}")
            raise
    
    def get_validation_dataset(self):
        """创建验证集数据集实例"""
        if self.is_validation:
            raise ValueError("当前实例已是验证集")
            
        # 传递必要的参数给验证集
        val_norm_params = {
            'input_mean': self.input_mean,
            'input_std': self.input_std
        }
        
        val_dataset = SPLdata(
            directory_path=self.directory_path,
            input_cols=self.input_cols,
            output_col=self.output_col,
            type_col=self.type_col,
            mode_col=self.mode_col,
            val_split=self.val_split,
            is_validation=True,
            norm_params=val_norm_params,
            random_seed=self.random_seed
        )
        
        # 共享编码器
        val_dataset.type_encoder = self.type_encoder
        val_dataset.mode_encoder = self.mode_encoder
        
        # 设置验证集索引
        val_dataset.data = self.data
        val_dataset.val_indices = self.val_indices
        val_dataset.indices = self.val_indices
        val_dataset.column_names = self.column_names
        val_dataset.output_length = self.output_length
        
        return val_dataset

# 1/3倍频程频谱预测数据集
class Octave_1_3_data(Dataset):
    def __init__(self, directory_path, input_cols=[4, 5, 6],
                 output_col=12, type_col=3, mode_col=2, val_split=0.2, 
                 is_validation=False, norm_params=None, random_seed=42):
        """
        非线性类型绑定数据集，支持训练/验证集划分和标准归一化
        
        参数:
            directory_path: CSV文件所在目录路径
            input_cols: 输入特征列索引列表
            output_col: 输出序列列索引
            type_col: 空调型号列索引
            mode_col: 模式类型列索引
            val_split: 验证集比例 (仅在非验证集实例中有效)
            is_validation: 是否为验证集
            norm_params: 归一化参数字典 (mean, std)，验证集需从训练集传递
            random_seed: 随机种子，确保可重现性
        """
        self.directory_path = directory_path
        self.input_cols = input_cols
        self.output_col = output_col
        self.type_col = type_col
        self.mode_col = mode_col
        self.val_split = val_split
        self.is_validation = is_validation
        self.norm_params = norm_params
        self.random_seed = random_seed
        
        # 初始化编码器
        self.type_encoder = LabelEncoder()
        self.mode_encoder = LabelEncoder()
        
        # 初始化数据相关变量
        self.data = None
        self.column_names = None
        self.output_length = None
        self.train_indices = None
        self.val_indices = None
        self.indices = None  # 当前使用的索引（训练或验证）
        
        # 归一化参数
        self.input_mean = None
        self.input_std = None
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 加载并处理数据
        self.load_and_concat_csv()
        
        # 划分数据集
        if not self.is_validation:
            self.split_data()
            self.compute_normalization_params()
        else:
            # 验证集使用训练集传递的归一化参数
            self.input_mean = self.norm_params['input_mean']
            self.input_std = self.norm_params['input_std']
    
    def load_and_concat_csv(self):
        """加载并拼接目录下所有CSV文件，执行数据预处理"""
        try:
            # 获取目录下所有CSV文件
            csv_files = [f for f in os.listdir(self.directory_path) if f.endswith('.csv')]
            
            if not csv_files:
                raise ValueError(f"目录 {self.directory_path} 中未找到CSV文件")
            
            # 读取并拼接所有CSV文件
            dfs = []
            for file in csv_files:
                file_path = os.path.join(self.directory_path, file)
                df = pd.read_csv(file_path)
                dfs.append(df)
            
            # 拼接所有数据框
            self.data = pd.concat(dfs, ignore_index=True)
            
            # 获取列名并打印
            self.column_names = self.data.columns.tolist()
            print("列名:", self.column_names)
            
            # 验证列索引是否有效
            self._validate_columns()
            
            # 打印输出列列表的长度
            sample_output = self.data.iloc[0, self.output_col]
            sample_list = ast.literal_eval(sample_output) if isinstance(sample_output, str) else sample_output
            self.output_length = len(sample_list)
            print(f"输出列列表的长度: {self.output_length}")
            
            # 对空调型号和模式列进行标签编码
            self.data.iloc[:, self.type_col] = self.type_encoder.fit_transform(self.data.iloc[:, self.type_col])
            self.data.iloc[:, self.mode_col] = self.mode_encoder.fit_transform(self.data.iloc[:, self.mode_col])
            print(f"空调型号编码: {dict(zip(self.type_encoder.classes_, self.type_encoder.transform(self.type_encoder.classes_)))}")
            print(f"模式类型编码: {dict(zip(self.mode_encoder.classes_, self.mode_encoder.transform(self.mode_encoder.classes_)))}")
            
            print(f"成功加载 {len(self.data)} 条数据记录")
            
        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            raise
    
    def split_data(self):
        """将数据划分为训练集和验证集"""
        n_samples = len(self.data)
        n_val = int(n_samples * self.val_split)
        
        # 随机选择验证集索引
        self.val_indices = np.random.choice(n_samples, size=n_val, replace=False)
        self.train_indices = np.setdiff1d(np.arange(n_samples), self.val_indices)
        
        # 设置当前使用的索引（训练集）
        self.indices = self.train_indices
        
        print(f"数据集划分完成 - 训练集: {len(self.train_indices)} 样本, 验证集: {len(self.val_indices)} 样本")
    
    def compute_normalization_params(self):
        """使用训练集计算输入特征的均值和标准差（标准归一化）"""
        if self.train_indices is None:
            self.split_data()
            
        # 获取训练集输入特征
        train_inputs = self.data.iloc[self.train_indices, self.input_cols].values
        
        # 计算均值和标准差
        self.input_mean = np.mean(train_inputs, axis=0)
        self.input_std = np.std(train_inputs, axis=0)
        
        # 添加微小值避免除零错误
        self.input_std = np.where(self.input_std < 1e-8, 1e-8, self.input_std)
        
        print("输入特征归一化参数计算完成")
        print(f"输入特征均值: {self.input_mean}")
        print(f"输入特征标准差: {self.input_std}")
    
    def _validate_columns(self):
        """验证指定的列索引是否有效"""
        max_col_index = len(self.column_names) - 1
        
        # 检查输入列
        for col in self.input_cols:
            if col < 0 or col > max_col_index:
                raise ValueError(f"输入列索引 {col} 超出有效范围 (0~{max_col_index})")
        
        # 检查其他列
        for col in [self.output_col, self.type_col, self.mode_col]:
            if col < 0 or col > max_col_index:
                raise ValueError(f"列索引 {col} 超出有效范围 (0~{max_col_index})")
    
    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.data)
    
    def __getitem__(self, idx):
        """获取指定索引的样本数据（应用归一化）"""
        if idx < 0 or idx >= len(self.indices):
            raise IndexError(f"样本索引 {idx} 超出范围")
        
        try:
            # 获取原始数据索引
            data_idx = self.indices[idx]
            
            # 获取输入特征
            input_sample = self.data.iloc[data_idx, self.input_cols].values.astype(float)
            
            # 应用标准归一化
            input_normalized = (input_sample - self.input_mean) / self.input_std
            
            # 获取空调型号和模式类型
            type_sample = self.data.iloc[data_idx, self.type_col]
            mode_sample = self.data.iloc[data_idx, self.mode_col]
            
            # 获取输出序列并解析
            output_sample = self.data.iloc[data_idx, self.output_col]
            
            output_list = ast.literal_eval(output_sample) if isinstance(output_sample, str) else output_sample
            
            # 转换为PyTorch张量
            input_tensor = torch.tensor(input_normalized, dtype=torch.float32)
            output_tensor = torch.tensor(output_list, dtype=torch.float32)
            # 转换为PyTorch张量
            type_tensor = torch.tensor(type_sample, dtype=torch.int64)
            mode_tensor = torch.tensor(mode_sample, dtype=torch.int64)
            # 验证输出长度
            if len(output_tensor) != self.output_length:
                raise ValueError(f"输出序列长度不匹配: 预期 {self.output_length}, 实际 {len(output_tensor)},样本详细{output_tensor}样本地址{self.data.iloc[data_idx, 0]}")
            
            return input_tensor, output_tensor, type_tensor, mode_tensor
            
        except Exception as e:
            print(f"获取样本 {idx} 时出错: {str(e)}")
            print(f"样本详细信息: {self.data.iloc[data_idx, :]}")
            raise
    
    def get_validation_dataset(self):
        """创建验证集数据集实例"""
        if self.is_validation:
            raise ValueError("当前实例已是验证集")
            
        # 传递必要的参数给验证集
        val_norm_params = {
            'input_mean': self.input_mean,
            'input_std': self.input_std
        }
        
        val_dataset = Octave_1_3_data(
            directory_path=self.directory_path,
            input_cols=self.input_cols,
            output_col=self.output_col,
            type_col=self.type_col,
            mode_col=self.mode_col,
            val_split=self.val_split,
            is_validation=True,
            norm_params=val_norm_params,
            random_seed=self.random_seed
        )
        
        # 共享编码器
        val_dataset.type_encoder = self.type_encoder
        val_dataset.mode_encoder = self.mode_encoder
        
        # 设置验证集索引
        val_dataset.data = self.data
        val_dataset.val_indices = self.val_indices
        val_dataset.indices = self.val_indices
        val_dataset.column_names = self.column_names
        val_dataset.output_length = self.output_length
        
        return val_dataset



# 线谱预测数据集
class LineOctavedata(Dataset):
    def __init__(self, directory_path, input_cols=[4, 5, 6],
                 output_col=13, type_col=3, mode_col=2, val_split=0.2, 
                 is_validation=False, norm_params=None, random_seed=42):
        """
        非线性类型绑定数据集，支持训练/验证集划分和标准归一化
        
        参数:
            directory_path: CSV文件所在目录路径
            input_cols: 输入特征列索引列表
            output_col: 输出序列列索引
            type_col: 空调型号列索引
            mode_col: 模式类型列索引
            val_split: 验证集比例 (仅在非验证集实例中有效)
            is_validation: 是否为验证集
            norm_params: 归一化参数字典 (mean, std)，验证集需从训练集传递
            random_seed: 随机种子，确保可重现性
        """
        self.directory_path = directory_path
        self.input_cols = input_cols
        self.output_col = output_col
        self.type_col = type_col
        self.mode_col = mode_col
        self.val_split = val_split
        self.is_validation = is_validation
        self.norm_params = norm_params
        self.random_seed = random_seed
        
        # 初始化编码器
        self.type_encoder = LabelEncoder()
        self.mode_encoder = LabelEncoder()
        
        # 初始化数据相关变量
        self.data = None
        self.column_names = None
        self.output_length = None
        self.train_indices = None
        self.val_indices = None
        self.indices = None  # 当前使用的索引（训练或验证）
        
        # 归一化参数
        self.input_mean = None
        self.input_std = None
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 加载并处理数据
        self.load_and_concat_csv()
        
        # 划分数据集
        if not self.is_validation:
            self.split_data()
            self.compute_normalization_params()
        else:
            # 验证集使用训练集传递的归一化参数
            self.input_mean = self.norm_params['input_mean']
            self.input_std = self.norm_params['input_std']
    
    def load_and_concat_csv(self):
        """加载并拼接目录下所有CSV文件，执行数据预处理"""
        try:
            # 获取目录下所有CSV文件
            csv_files = [f for f in os.listdir(self.directory_path) if f.endswith('.csv')]
            
            if not csv_files:
                raise ValueError(f"目录 {self.directory_path} 中未找到CSV文件")
            
            # 读取并拼接所有CSV文件
            dfs = []
            for file in csv_files:
                file_path = os.path.join(self.directory_path, file)
                df = pd.read_csv(file_path)
                dfs.append(df)
            
            # 拼接所有数据框
            self.data = pd.concat(dfs, ignore_index=True)
            
            # 获取列名并打印
            self.column_names = self.data.columns.tolist()
            print("列名:", self.column_names)
            
            # 验证列索引是否有效
            self._validate_columns()
            
            # 打印输出列列表的长度
            sample_output = self.data.iloc[0, self.output_col]
            sample_list = ast.literal_eval(sample_output) if isinstance(sample_output, str) else sample_output
            self.output_length = len(sample_list)
            print(f"输出列列表的长度: {self.output_length}")
            
            # 对空调型号和模式列进行标签编码
            # self.data.iloc[:, self.bowl_col] = self.bowl_encoder.fit_transform(self.data.iloc[:, self.bowl_col])
            self.data.iloc[:, self.type_col] = self.type_encoder.fit_transform(self.data.iloc[:, self.type_col])
            self.data.iloc[:, self.mode_col] = self.mode_encoder.fit_transform(self.data.iloc[:, self.mode_col])
            
            print(f"成功加载 {len(self.data)} 条数据记录")
            
        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            raise
    
    def split_data(self):
        """将数据划分为训练集和验证集"""
        n_samples = len(self.data)
        n_val = int(n_samples * self.val_split)
        
        # 随机选择验证集索引
        self.val_indices = np.random.choice(n_samples, size=n_val, replace=False)
        self.train_indices = np.setdiff1d(np.arange(n_samples), self.val_indices)
        
        # 设置当前使用的索引（训练集）
        self.indices = self.train_indices
        
        print(f"数据集划分完成 - 训练集: {len(self.train_indices)} 样本, 验证集: {len(self.val_indices)} 样本")
    
    def compute_normalization_params(self):
        """使用训练集计算输入特征的均值和标准差（标准归一化）"""
        if self.train_indices is None:
            self.split_data()
            
        # 获取训练集输入特征
        train_inputs = self.data.iloc[self.train_indices, self.input_cols].values
        
        # 计算均值和标准差
        self.input_mean = np.mean(train_inputs, axis=0)
        self.input_std = np.std(train_inputs, axis=0)
        
        # 添加微小值避免除零错误
        self.input_std = np.where(self.input_std < 1e-8, 1e-8, self.input_std)
        
        print("输入特征归一化参数计算完成")
        print(f"输入特征均值: {self.input_mean}")
        print(f"输入特征标准差: {self.input_std}")
    
    def _validate_columns(self):
        """验证指定的列索引是否有效"""
        max_col_index = len(self.column_names) - 1
        
        # 检查输入列
        for col in self.input_cols:
            if col < 0 or col > max_col_index:
                raise ValueError(f"输入列索引 {col} 超出有效范围 (0~{max_col_index})")
        
        # 检查其他列
        for col in [self.output_col, self.type_col, self.mode_col]:
            if col < 0 or col > max_col_index:
                raise ValueError(f"列索引 {col} 超出有效范围 (0~{max_col_index})")
    
    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.data)
    
    def __getitem__(self, idx):
        """获取指定索引的样本数据（应用归一化）"""
        if idx < 0 or idx >= len(self.indices):
            raise IndexError(f"样本索引 {idx} 超出范围")
        
        try:
            # 获取原始数据索引
            data_idx = self.indices[idx]
            
            # 获取输入特征
            input_sample = self.data.iloc[data_idx, self.input_cols].values.astype(float)
            
            # 应用标准归一化
            input_normalized = (input_sample - self.input_mean) / self.input_std
            
            # 获取空调型号和模式类型
            type_sample = self.data.iloc[data_idx, self.type_col]
            mode_sample = self.data.iloc[data_idx, self.mode_col]
            
            # 获取输出序列并解析
            output_sample = self.data.iloc[data_idx, self.output_col]
            output_list = ast.literal_eval(output_sample) if isinstance(output_sample, str) else output_sample

            input_tensor = torch.tensor(input_normalized, dtype=torch.float32)
            output_tensor = torch.tensor(output_list[:2501], dtype=torch.float32)
            type_tensor = torch.tensor(type_sample, dtype=torch.int64)
            mode_tensor = torch.tensor(mode_sample, dtype=torch.int64)
            
            # # 验证输出长度
            # if len(output_tensor) != self.output_length:
            #     raise ValueError(f"输出序列长度不匹配: 预期 {self.output_length}, 实际 {len(output_tensor)}")
            
            return input_tensor, output_tensor, type_tensor, mode_tensor
            
        except Exception as e:
            print(f"获取样本 {idx} 时出错: {str(e)}")
            raise
    def convert_to_spl(self, amplitude_list, max_amplitude):
        """
        将原始频谱幅度值转换为声压级（SPL）
        
        参数:
            amplitude_list: 原始FFT幅度值列表
            
        返回:
            spl_list: 声压级值列表（单位：dB SPL）
        """
        # 系统参数（根据您提供的参数）
        sensitivity = 49.78e-3  # 49.78 mV/Pa 转换为 0.04978 V/Pa
        full_scale_voltage = 10.0  # 满量程电压 10V
        
        # 参考声压（国际标准）
        p_ref = 2e-5  # 20 μPa，人耳可听阈声压[6,8](@ref)
        
        spl_list = []
        
        for amplitude in amplitude_list:
            # 步骤1: 将相对幅度值转换为实际电压
            # 假设原始FFT幅度值是相对值（0到某个范围），需要映射到实际电压
            actual_voltage = amplitude * full_scale_voltage
            
            # 步骤2: 将电压转换为实际声压
            # 根据灵敏度公式：声压 = 电压 / 灵敏度
            actual_pressure = actual_voltage / sensitivity
            actual_pressure = actual_pressure / max_amplitude
            
            # 步骤3: 计算声压级（SPL）
            # SPL = 20 × log₁₀(p / p_ref)[6,7](@ref)
            if actual_pressure <= 0:
                # 处理零或负值，设置为很小的dB值
                spl_value = -120.0  # 远低于可听阈
            else:
                spl_value = 20 * math.log10(actual_pressure / p_ref)
            
            spl_list.append(spl_value)
        
        return spl_list
    def get_validation_dataset(self):
        """创建验证集数据集实例"""
        if self.is_validation:
            raise ValueError("当前实例已是验证集")
            
        # 传递必要的参数给验证集
        val_norm_params = {
            'input_mean': self.input_mean,
            'input_std': self.input_std
        }
        
        val_dataset = LineOctavedata(
            directory_path=self.directory_path,
            input_cols=self.input_cols,
            output_col=self.output_col,
            type_col=self.type_col,
            mode_col=self.mode_col,
            val_split=self.val_split,
            is_validation=True,
            norm_params=val_norm_params,
            random_seed=self.random_seed
        )
        
        # 共享编码器
        val_dataset.type_encoder = self.type_encoder
        val_dataset.mode_encoder = self.mode_encoder
        
        # 设置验证集索引
        val_dataset.data = self.data
        val_dataset.val_indices = self.val_indices
        val_dataset.indices = self.val_indices
        val_dataset.column_names = self.column_names
        val_dataset.output_length = self.output_length
        
        return val_dataset

    # ... 其他现有方法保持不变 ...

# 声品质预测数据集
class Soundquality_one_data(Dataset):
    def __init__(self, directory_path, input_cols=[4, 5, 6, 7, 8, 9, 10],
                bowl_col=4,output_col=[14], mode_col=2, val_split=0.2, 
                 is_validation=False, norm_params=None, random_seed=42):
        """
        非线性类型绑定数据集，支持训练/验证集划分和标准归一化
        
        参数:
            directory_path: CSV文件所在目录路径
            input_cols: 输入特征列索引列表
            output_col: 输出序列列索引
            bowl_col: 碗类型列索引
            mode_col: 模式类型列索引
            val_split: 验证集比例 (仅在非验证集实例中有效)
            is_validation: 是否为验证集
            norm_params: 归一化参数字典 (mean, std)，验证集需从训练集传递
            random_seed: 随机种子，确保可重现性
        """
        self.directory_path = directory_path
        self.input_cols = input_cols
        self.output_col = output_col
        self.bowl_col = bowl_col
        self.mode_col = mode_col
        self.val_split = val_split
        self.is_validation = is_validation
        self.norm_params = norm_params
        self.random_seed = random_seed
        
        # 初始化编码器
        self.bowl_encoder = LabelEncoder()
        self.mode_encoder = LabelEncoder()
        
        # 初始化数据相关变量
        self.data = None
        self.column_names = None
        self.output_length = None
        self.train_indices = None
        self.val_indices = None
        self.indices = None  # 当前使用的索引（训练或验证）
        
        # 归一化参数
        self.input_mean = None
        self.input_std = None
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 加载并处理数据
        self.load_and_concat_csv()
        
        # 划分数据集
        if not self.is_validation:
            self.split_data()
            self.compute_normalization_params()
        else:
            # 验证集使用训练集传递的归一化参数
            self.input_mean = self.norm_params['input_mean']
            self.input_std = self.norm_params['input_std']
    
    def load_and_concat_csv(self):
        """加载并拼接目录下所有CSV文件，执行数据预处理"""
        try:
            # 获取目录下所有CSV文件
            csv_files = [f for f in os.listdir(self.directory_path) if f.endswith('.csv')]
            
            if not csv_files:
                raise ValueError(f"目录 {self.directory_path} 中未找到CSV文件")
            
            # 读取并拼接所有CSV文件
            dfs = []
            for file in csv_files:
                file_path = os.path.join(self.directory_path, file)
                df = pd.read_csv(file_path)
                dfs.append(df)
            
            # 拼接所有数据框
            self.data = pd.concat(dfs, ignore_index=True)
            
            # 获取列名并打印
            self.column_names = self.data.columns.tolist()
            print("列名:", self.column_names)
            
            # 验证列索引是否有效
            self._validate_columns()
            
            # 打印输出列列表的长度
            # sample_output = self.data.iloc[0, self.output_col]
            # sample_list = ast.literal_eval(sample_output) if isinstance(sample_output, str) else sample_output
            # self.output_length = len(sample_list)
            # print(f"输出列列表的长度: {self.output_length}")
            
            # 对碗和模式列进行标签编码
            # self.data.iloc[:, self.bowl_col] = self.bowl_encoder.fit_transform(self.data.iloc[:, self.bowl_col])
            self.data.iloc[:, self.mode_col] = self.mode_encoder.fit_transform(self.data.iloc[:, self.mode_col])
            
            print(f"成功加载 {len(self.data)} 条数据记录")
            
        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            raise
    
    def split_data(self):
        """将数据划分为训练集和验证集"""
        n_samples = len(self.data)
        n_val = int(n_samples * self.val_split)
        
        # 随机选择验证集索引
        self.val_indices = np.random.choice(n_samples, size=n_val, replace=False)
        self.train_indices = np.setdiff1d(np.arange(n_samples), self.val_indices)
        
        # 设置当前使用的索引（训练集）
        self.indices = self.train_indices
        
        print(f"数据集划分完成 - 训练集: {len(self.train_indices)} 样本, 验证集: {len(self.val_indices)} 样本")
    
    def compute_normalization_params(self):
        """使用训练集计算输入特征的均值和标准差（标准归一化）"""
        if self.train_indices is None:
            self.split_data()
            
        # 获取训练集输入特征
        train_inputs = self.data.iloc[self.train_indices, self.input_cols].values
        
        # 计算均值和标准差
        self.input_mean = np.mean(train_inputs, axis=0)
        self.input_std = np.std(train_inputs, axis=0)
        
        # 添加微小值避免除零错误
        self.input_std = np.where(self.input_std < 1e-8, 1e-8, self.input_std)
        
        print("输入特征归一化参数计算完成")
        print(f"输入特征均值: {self.input_mean}")
        print(f"输入特征标准差: {self.input_std}")
    
    def _validate_columns(self):
        """验证指定的列索引是否有效"""
        max_col_index = len(self.column_names) - 1
        
        # 检查输入列
        for col in self.input_cols:
            if col < 0 or col > max_col_index:
                raise ValueError(f"输入列索引 {col} 超出有效范围 (0~{max_col_index})")
        
        # # 检查其他列
        # for col in [self.output_col, self.bowl_col, self.mode_col]:
        #     if col < 0 or col > max_col_index:
        #         raise ValueError(f"列索引 {col} 超出有效范围 (0~{max_col_index})")
    
    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.data)
    
    def __getitem__(self, idx):
        """获取指定索引的样本数据（应用归一化）"""
        if idx < 0 or idx >= len(self.indices):
            raise IndexError(f"样本索引 {idx} 超出范围")
        
        try:
            # 获取原始数据索引
            data_idx = self.indices[idx]
            
            # 获取输入特征
            input_sample = self.data.iloc[data_idx, self.input_cols].values.astype(float)
            
            # 应用标准归一化
            input_normalized = (input_sample - self.input_mean) / self.input_std
            
            # 获取碗类型和模式类型
            # bowl_sample = self.data.iloc[data_idx, self.bowl_col]
            mode_sample = self.data.iloc[data_idx, self.mode_col]
            
            # 获取输出序列并解析
            output_sample = self.data.iloc[data_idx, self.output_col].tolist()
            # output_list = ast.literal_eval(output_sample) if isinstance(output_sample, str) else output_sample
            # output2_sample = self.data.iloc[data_idx, self.output2_col]
            # output_list.extend(output2_sample)

            
            # 转换为PyTorch张量
            input_tensor = torch.tensor(input_normalized, dtype=torch.float32)
            output_tensor = torch.tensor(output_sample, dtype=torch.float32)  # 使用转换后的dB值
            # bowl_tensor = torch.tensor(bowl_sample, dtype=torch.int64)
            mode_tensor = torch.tensor(mode_sample, dtype=torch.int64)
            # 验证输出长度
            # if len(output_tensor) != self.output_length:
            #     raise ValueError(f"输出序列长度不匹配: 预期 {self.output_length}, 实际 {len(output_tensor)}")
            
            return input_tensor, output_tensor, mode_tensor
            
        except Exception as e:
            print(f"获取样本 {idx} 时出错: {str(e)}")
            raise
    
    def get_validation_dataset(self):
        """创建验证集数据集实例"""
        if self.is_validation:
            raise ValueError("当前实例已是验证集")
            
        # 传递必要的参数给验证集
        val_norm_params = {
            'input_mean': self.input_mean,
            'input_std': self.input_std
        }
        
        val_dataset = Soundquality_one_data(
            directory_path=self.directory_path,
            input_cols=self.input_cols,
            output_col=self.output_col,
            bowl_col=self.bowl_col,
            mode_col=self.mode_col,
            val_split=self.val_split,
            is_validation=True,
            norm_params=val_norm_params,
            random_seed=self.random_seed
        )
        
        # 共享编码器
        val_dataset.bowl_encoder = self.bowl_encoder
        val_dataset.mode_encoder = self.mode_encoder
        
        # 设置验证集索引
        val_dataset.data = self.data
        val_dataset.val_indices = self.val_indices
        val_dataset.indices = self.val_indices
        val_dataset.column_names = self.column_names
        val_dataset.output_length = self.output_length
        
        return val_dataset

class Soundquality_two_data(Dataset):
    def __init__(self, directory_path, input_cols=[4, 5, 6, 7, 8, 9, 10],
                bowl_col=4,output_col=[15], mode_col=2, val_split=0.2, 
                 is_validation=False, norm_params=None, random_seed=42):
        """
        非线性类型绑定数据集，支持训练/验证集划分和标准归一化
        
        参数:
            directory_path: CSV文件所在目录路径
            input_cols: 输入特征列索引列表
            output_col: 输出序列列索引
            bowl_col: 碗类型列索引
            mode_col: 模式类型列索引
            val_split: 验证集比例 (仅在非验证集实例中有效)
            is_validation: 是否为验证集
            norm_params: 归一化参数字典 (mean, std)，验证集需从训练集传递
            random_seed: 随机种子，确保可重现性
        """
        self.directory_path = directory_path
        self.input_cols = input_cols
        self.output_col = output_col
        self.bowl_col = bowl_col
        self.mode_col = mode_col
        self.val_split = val_split
        self.is_validation = is_validation
        self.norm_params = norm_params
        self.random_seed = random_seed
        
        # 初始化编码器
        self.bowl_encoder = LabelEncoder()
        self.mode_encoder = LabelEncoder()
        
        # 初始化数据相关变量
        self.data = None
        self.column_names = None
        self.output_length = None
        self.train_indices = None
        self.val_indices = None
        self.indices = None  # 当前使用的索引（训练或验证）
        
        # 归一化参数
        self.input_mean = None
        self.input_std = None
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 加载并处理数据
        self.load_and_concat_csv()
        
        # 划分数据集
        if not self.is_validation:
            self.split_data()
            self.compute_normalization_params()
        else:
            # 验证集使用训练集传递的归一化参数
            self.input_mean = self.norm_params['input_mean']
            self.input_std = self.norm_params['input_std']
    
    def load_and_concat_csv(self):
        """加载并拼接目录下所有CSV文件，执行数据预处理"""
        try:
            # 获取目录下所有CSV文件
            csv_files = [f for f in os.listdir(self.directory_path) if f.endswith('.csv')]
            
            if not csv_files:
                raise ValueError(f"目录 {self.directory_path} 中未找到CSV文件")
            
            # 读取并拼接所有CSV文件
            dfs = []
            for file in csv_files:
                file_path = os.path.join(self.directory_path, file)
                df = pd.read_csv(file_path)
                dfs.append(df)
            
            # 拼接所有数据框
            self.data = pd.concat(dfs, ignore_index=True)
            
            # 获取列名并打印
            self.column_names = self.data.columns.tolist()
            print("列名:", self.column_names)
            
            # 验证列索引是否有效
            self._validate_columns()
            
            # 打印输出列列表的长度
            # sample_output = self.data.iloc[0, self.output_col]
            # sample_list = ast.literal_eval(sample_output) if isinstance(sample_output, str) else sample_output
            # self.output_length = len(sample_list)
            # print(f"输出列列表的长度: {self.output_length}")
            
            # 对碗和模式列进行标签编码
            # self.data.iloc[:, self.bowl_col] = self.bowl_encoder.fit_transform(self.data.iloc[:, self.bowl_col])
            self.data.iloc[:, self.mode_col] = self.mode_encoder.fit_transform(self.data.iloc[:, self.mode_col])
            
            print(f"成功加载 {len(self.data)} 条数据记录")
            
        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            raise
    
    def split_data(self):
        """将数据划分为训练集和验证集"""
        n_samples = len(self.data)
        n_val = int(n_samples * self.val_split)
        
        # 随机选择验证集索引
        self.val_indices = np.random.choice(n_samples, size=n_val, replace=False)
        self.train_indices = np.setdiff1d(np.arange(n_samples), self.val_indices)
        
        # 设置当前使用的索引（训练集）
        self.indices = self.train_indices
        
        print(f"数据集划分完成 - 训练集: {len(self.train_indices)} 样本, 验证集: {len(self.val_indices)} 样本")
    
    def compute_normalization_params(self):
        """使用训练集计算输入特征的均值和标准差（标准归一化）"""
        if self.train_indices is None:
            self.split_data()
            
        # 获取训练集输入特征
        train_inputs = self.data.iloc[self.train_indices, self.input_cols].values
        
        # 计算均值和标准差
        self.input_mean = np.mean(train_inputs, axis=0)
        self.input_std = np.std(train_inputs, axis=0)
        
        # 添加微小值避免除零错误
        self.input_std = np.where(self.input_std < 1e-8, 1e-8, self.input_std)
        
        print("输入特征归一化参数计算完成")
        print(f"输入特征均值: {self.input_mean}")
        print(f"输入特征标准差: {self.input_std}")
    
    def _validate_columns(self):
        """验证指定的列索引是否有效"""
        max_col_index = len(self.column_names) - 1
        
        # 检查输入列
        for col in self.input_cols:
            if col < 0 or col > max_col_index:
                raise ValueError(f"输入列索引 {col} 超出有效范围 (0~{max_col_index})")
        
        # # 检查其他列
        # for col in [self.output_col, self.bowl_col, self.mode_col]:
        #     if col < 0 or col > max_col_index:
        #         raise ValueError(f"列索引 {col} 超出有效范围 (0~{max_col_index})")
    
    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.data)
    
    def __getitem__(self, idx):
        """获取指定索引的样本数据（应用归一化）"""
        if idx < 0 or idx >= len(self.indices):
            raise IndexError(f"样本索引 {idx} 超出范围")
        
        try:
            # 获取原始数据索引
            data_idx = self.indices[idx]
            
            # 获取输入特征
            input_sample = self.data.iloc[data_idx, self.input_cols].values.astype(float)
            
            # 应用标准归一化
            input_normalized = (input_sample - self.input_mean) / self.input_std
            
            # 获取碗类型和模式类型
            # bowl_sample = self.data.iloc[data_idx, self.bowl_col]
            mode_sample = self.data.iloc[data_idx, self.mode_col]
            
            # 获取输出序列并解析
            output_sample = self.data.iloc[data_idx, self.output_col].tolist()
            # output_list = ast.literal_eval(output_sample) if isinstance(output_sample, str) else output_sample
            # output2_sample = self.data.iloc[data_idx, self.output2_col]
            # output_list.extend(output2_sample)

            
            # 转换为PyTorch张量
            input_tensor = torch.tensor(input_normalized, dtype=torch.float32)
            output_tensor = torch.tensor(output_sample, dtype=torch.float32)  # 使用转换后的dB值
            # bowl_tensor = torch.tensor(bowl_sample, dtype=torch.int64)
            mode_tensor = torch.tensor(mode_sample, dtype=torch.int64)
            # 验证输出长度
            # if len(output_tensor) != self.output_length:
            #     raise ValueError(f"输出序列长度不匹配: 预期 {self.output_length}, 实际 {len(output_tensor)}")
            
            return input_tensor, output_tensor, mode_tensor
            
        except Exception as e:
            print(f"获取样本 {idx} 时出错: {str(e)}")
            raise
    
    def get_validation_dataset(self):
        """创建验证集数据集实例"""
        if self.is_validation:
            raise ValueError("当前实例已是验证集")
            
        # 传递必要的参数给验证集
        val_norm_params = {
            'input_mean': self.input_mean,
            'input_std': self.input_std
        }

        val_dataset = Soundquality_two_data(
            directory_path=self.directory_path,
            input_cols=self.input_cols,
            output_col=self.output_col,
            bowl_col=self.bowl_col,
            mode_col=self.mode_col,
            val_split=self.val_split,
            is_validation=True,
            norm_params=val_norm_params,
            random_seed=self.random_seed
        )
        
        # 共享编码器
        val_dataset.bowl_encoder = self.bowl_encoder
        val_dataset.mode_encoder = self.mode_encoder
        
        # 设置验证集索引
        val_dataset.data = self.data
        val_dataset.val_indices = self.val_indices
        val_dataset.indices = self.val_indices
        val_dataset.column_names = self.column_names
        val_dataset.output_length = self.output_length
        
        return val_dataset

class Soundquality_three_data(Dataset):
    def __init__(self, directory_path, input_cols=[4, 5, 6, 7, 8, 9, 10],
                bowl_col=4,output_col=[16], mode_col=2, val_split=0.2, 
                 is_validation=False, norm_params=None, random_seed=42):
        """
        非线性类型绑定数据集，支持训练/验证集划分和标准归一化
        
        参数:
            directory_path: CSV文件所在目录路径
            input_cols: 输入特征列索引列表
            output_col: 输出序列列索引
            bowl_col: 碗类型列索引
            mode_col: 模式类型列索引
            val_split: 验证集比例 (仅在非验证集实例中有效)
            is_validation: 是否为验证集
            norm_params: 归一化参数字典 (mean, std)，验证集需从训练集传递
            random_seed: 随机种子，确保可重现性
        """
        self.directory_path = directory_path
        self.input_cols = input_cols
        self.output_col = output_col
        self.bowl_col = bowl_col
        self.mode_col = mode_col
        self.val_split = val_split
        self.is_validation = is_validation
        self.norm_params = norm_params
        self.random_seed = random_seed
        
        # 初始化编码器
        self.bowl_encoder = LabelEncoder()
        self.mode_encoder = LabelEncoder()
        
        # 初始化数据相关变量
        self.data = None
        self.column_names = None
        self.output_length = None
        self.train_indices = None
        self.val_indices = None
        self.indices = None  # 当前使用的索引（训练或验证）
        
        # 归一化参数
        self.input_mean = None
        self.input_std = None
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 加载并处理数据
        self.load_and_concat_csv()
        
        # 划分数据集
        if not self.is_validation:
            self.split_data()
            self.compute_normalization_params()
        else:
            # 验证集使用训练集传递的归一化参数
            self.input_mean = self.norm_params['input_mean']
            self.input_std = self.norm_params['input_std']
    
    def load_and_concat_csv(self):
        """加载并拼接目录下所有CSV文件，执行数据预处理"""
        try:
            # 获取目录下所有CSV文件
            csv_files = [f for f in os.listdir(self.directory_path) if f.endswith('.csv')]
            
            if not csv_files:
                raise ValueError(f"目录 {self.directory_path} 中未找到CSV文件")
            
            # 读取并拼接所有CSV文件
            dfs = []
            for file in csv_files:
                file_path = os.path.join(self.directory_path, file)
                df = pd.read_csv(file_path)
                dfs.append(df)
            
            # 拼接所有数据框
            self.data = pd.concat(dfs, ignore_index=True)
            
            # 获取列名并打印
            self.column_names = self.data.columns.tolist()
            print("列名:", self.column_names)
            
            # 验证列索引是否有效
            self._validate_columns()
            
            # 打印输出列列表的长度
            # sample_output = self.data.iloc[0, self.output_col]
            # sample_list = ast.literal_eval(sample_output) if isinstance(sample_output, str) else sample_output
            # self.output_length = len(sample_list)
            # print(f"输出列列表的长度: {self.output_length}")
            
            # 对碗和模式列进行标签编码
            # self.data.iloc[:, self.bowl_col] = self.bowl_encoder.fit_transform(self.data.iloc[:, self.bowl_col])
            self.data.iloc[:, self.mode_col] = self.mode_encoder.fit_transform(self.data.iloc[:, self.mode_col])
            
            print(f"成功加载 {len(self.data)} 条数据记录")
            
        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            raise
    
    def split_data(self):
        """将数据划分为训练集和验证集"""
        n_samples = len(self.data)
        n_val = int(n_samples * self.val_split)
        
        # 随机选择验证集索引
        self.val_indices = np.random.choice(n_samples, size=n_val, replace=False)
        self.train_indices = np.setdiff1d(np.arange(n_samples), self.val_indices)
        
        # 设置当前使用的索引（训练集）
        self.indices = self.train_indices
        
        print(f"数据集划分完成 - 训练集: {len(self.train_indices)} 样本, 验证集: {len(self.val_indices)} 样本")
    
    def compute_normalization_params(self):
        """使用训练集计算输入特征的均值和标准差（标准归一化）"""
        if self.train_indices is None:
            self.split_data()
            
        # 获取训练集输入特征
        train_inputs = self.data.iloc[self.train_indices, self.input_cols].values
        
        # 计算均值和标准差
        self.input_mean = np.mean(train_inputs, axis=0)
        self.input_std = np.std(train_inputs, axis=0)
        
        # 添加微小值避免除零错误
        self.input_std = np.where(self.input_std < 1e-8, 1e-8, self.input_std)
        
        print("输入特征归一化参数计算完成")
        print(f"输入特征均值: {self.input_mean}")
        print(f"输入特征标准差: {self.input_std}")
    
    def _validate_columns(self):
        """验证指定的列索引是否有效"""
        max_col_index = len(self.column_names) - 1
        
        # 检查输入列
        for col in self.input_cols:
            if col < 0 or col > max_col_index:
                raise ValueError(f"输入列索引 {col} 超出有效范围 (0~{max_col_index})")
        
        # # 检查其他列
        # for col in [self.output_col, self.bowl_col, self.mode_col]:
        #     if col < 0 or col > max_col_index:
        #         raise ValueError(f"列索引 {col} 超出有效范围 (0~{max_col_index})")
    
    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.data)
    
    def __getitem__(self, idx):
        """获取指定索引的样本数据（应用归一化）"""
        if idx < 0 or idx >= len(self.indices):
            raise IndexError(f"样本索引 {idx} 超出范围")
        
        try:
            # 获取原始数据索引
            data_idx = self.indices[idx]
            
            # 获取输入特征
            input_sample = self.data.iloc[data_idx, self.input_cols].values.astype(float)
            
            # 应用标准归一化
            input_normalized = (input_sample - self.input_mean) / self.input_std
            
            # 获取碗类型和模式类型
            # bowl_sample = self.data.iloc[data_idx, self.bowl_col]
            mode_sample = self.data.iloc[data_idx, self.mode_col]
            
            # 获取输出序列并解析
            output_sample = self.data.iloc[data_idx, self.output_col].tolist()
            # output_list = ast.literal_eval(output_sample) if isinstance(output_sample, str) else output_sample
            # output2_sample = self.data.iloc[data_idx, self.output2_col]
            # output_list.extend(output2_sample)

            
            # 转换为PyTorch张量
            input_tensor = torch.tensor(input_normalized, dtype=torch.float32)
            output_tensor = torch.tensor(output_sample, dtype=torch.float32)  # 使用转换后的dB值
            # bowl_tensor = torch.tensor(bowl_sample, dtype=torch.int64)
            mode_tensor = torch.tensor(mode_sample, dtype=torch.int64)
            # 验证输出长度
            # if len(output_tensor) != self.output_length:
            #     raise ValueError(f"输出序列长度不匹配: 预期 {self.output_length}, 实际 {len(output_tensor)}")
            
            return input_tensor, output_tensor, mode_tensor
            
        except Exception as e:
            print(f"获取样本 {idx} 时出错: {str(e)}")
            raise
    
    def get_validation_dataset(self):
        """创建验证集数据集实例"""
        if self.is_validation:
            raise ValueError("当前实例已是验证集")
            
        # 传递必要的参数给验证集
        val_norm_params = {
            'input_mean': self.input_mean,
            'input_std': self.input_std
        }
        
        val_dataset = Soundquality_three_data(
            directory_path=self.directory_path,
            input_cols=self.input_cols,
            output_col=self.output_col,
            bowl_col=self.bowl_col,
            mode_col=self.mode_col,
            val_split=self.val_split,
            is_validation=True,
            norm_params=val_norm_params,
            random_seed=self.random_seed
        )
        
        # 共享编码器
        val_dataset.bowl_encoder = self.bowl_encoder
        val_dataset.mode_encoder = self.mode_encoder
        
        # 设置验证集索引
        val_dataset.data = self.data
        val_dataset.val_indices = self.val_indices
        val_dataset.indices = self.val_indices
        val_dataset.column_names = self.column_names
        val_dataset.output_length = self.output_length
        
        return val_dataset

class Soundquality_four_data(Dataset):
    def __init__(self, directory_path, input_cols=[4, 5, 6, 7, 8, 9, 10],
                bowl_col=4,output_col=[17], mode_col=2, val_split=0.2, 
                 is_validation=False, norm_params=None, random_seed=42):
        """
        非线性类型绑定数据集，支持训练/验证集划分和标准归一化
        
        参数:
            directory_path: CSV文件所在目录路径
            input_cols: 输入特征列索引列表
            output_col: 输出序列列索引
            bowl_col: 碗类型列索引
            mode_col: 模式类型列索引
            val_split: 验证集比例 (仅在非验证集实例中有效)
            is_validation: 是否为验证集
            norm_params: 归一化参数字典 (mean, std)，验证集需从训练集传递
            random_seed: 随机种子，确保可重现性
        """
        self.directory_path = directory_path
        self.input_cols = input_cols
        self.output_col = output_col
        self.bowl_col = bowl_col
        self.mode_col = mode_col
        self.val_split = val_split
        self.is_validation = is_validation
        self.norm_params = norm_params
        self.random_seed = random_seed
        
        # 初始化编码器
        self.bowl_encoder = LabelEncoder()
        self.mode_encoder = LabelEncoder()
        
        # 初始化数据相关变量
        self.data = None
        self.column_names = None
        self.output_length = None
        self.train_indices = None
        self.val_indices = None
        self.indices = None  # 当前使用的索引（训练或验证）
        
        # 归一化参数
        self.input_mean = None
        self.input_std = None
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 加载并处理数据
        self.load_and_concat_csv()
        
        # 划分数据集
        if not self.is_validation:
            self.split_data()
            self.compute_normalization_params()
        else:
            # 验证集使用训练集传递的归一化参数
            self.input_mean = self.norm_params['input_mean']
            self.input_std = self.norm_params['input_std']
    
    def load_and_concat_csv(self):
        """加载并拼接目录下所有CSV文件，执行数据预处理"""
        try:
            # 获取目录下所有CSV文件
            csv_files = [f for f in os.listdir(self.directory_path) if f.endswith('.csv')]
            
            if not csv_files:
                raise ValueError(f"目录 {self.directory_path} 中未找到CSV文件")
            
            # 读取并拼接所有CSV文件
            dfs = []
            for file in csv_files:
                file_path = os.path.join(self.directory_path, file)
                df = pd.read_csv(file_path)
                dfs.append(df)
            
            # 拼接所有数据框
            self.data = pd.concat(dfs, ignore_index=True)
            
            # 获取列名并打印
            self.column_names = self.data.columns.tolist()
            print("列名:", self.column_names)
            
            # 验证列索引是否有效
            self._validate_columns()
            
            # 打印输出列列表的长度
            # sample_output = self.data.iloc[0, self.output_col]
            # sample_list = ast.literal_eval(sample_output) if isinstance(sample_output, str) else sample_output
            # self.output_length = len(sample_list)
            # print(f"输出列列表的长度: {self.output_length}")
            
            # 对碗和模式列进行标签编码
            # self.data.iloc[:, self.bowl_col] = self.bowl_encoder.fit_transform(self.data.iloc[:, self.bowl_col])
            self.data.iloc[:, self.mode_col] = self.mode_encoder.fit_transform(self.data.iloc[:, self.mode_col])
            
            print(f"成功加载 {len(self.data)} 条数据记录")
            
        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            raise
    
    def split_data(self):
        """将数据划分为训练集和验证集"""
        n_samples = len(self.data)
        n_val = int(n_samples * self.val_split)
        
        # 随机选择验证集索引
        self.val_indices = np.random.choice(n_samples, size=n_val, replace=False)
        self.train_indices = np.setdiff1d(np.arange(n_samples), self.val_indices)
        
        # 设置当前使用的索引（训练集）
        self.indices = self.train_indices
        
        print(f"数据集划分完成 - 训练集: {len(self.train_indices)} 样本, 验证集: {len(self.val_indices)} 样本")
    
    def compute_normalization_params(self):
        """使用训练集计算输入特征的均值和标准差（标准归一化）"""
        if self.train_indices is None:
            self.split_data()
            
        # 获取训练集输入特征
        train_inputs = self.data.iloc[self.train_indices, self.input_cols].values
        
        # 计算均值和标准差
        self.input_mean = np.mean(train_inputs, axis=0)
        self.input_std = np.std(train_inputs, axis=0)
        
        # 添加微小值避免除零错误
        self.input_std = np.where(self.input_std < 1e-8, 1e-8, self.input_std)
        
        print("输入特征归一化参数计算完成")
        print(f"输入特征均值: {self.input_mean}")
        print(f"输入特征标准差: {self.input_std}")
    
    def _validate_columns(self):
        """验证指定的列索引是否有效"""
        max_col_index = len(self.column_names) - 1
        
        # 检查输入列
        for col in self.input_cols:
            if col < 0 or col > max_col_index:
                raise ValueError(f"输入列索引 {col} 超出有效范围 (0~{max_col_index})")
        
        # # 检查其他列
        # for col in [self.output_col, self.bowl_col, self.mode_col]:
        #     if col < 0 or col > max_col_index:
        #         raise ValueError(f"列索引 {col} 超出有效范围 (0~{max_col_index})")
    
    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.data)
    
    def __getitem__(self, idx):
        """获取指定索引的样本数据（应用归一化）"""
        if idx < 0 or idx >= len(self.indices):
            raise IndexError(f"样本索引 {idx} 超出范围")
        
        try:
            # 获取原始数据索引
            data_idx = self.indices[idx]
            
            # 获取输入特征
            input_sample = self.data.iloc[data_idx, self.input_cols].values.astype(float)
            
            # 应用标准归一化
            input_normalized = (input_sample - self.input_mean) / self.input_std
            
            # 获取碗类型和模式类型
            # bowl_sample = self.data.iloc[data_idx, self.bowl_col]
            mode_sample = self.data.iloc[data_idx, self.mode_col]
            
            # 获取输出序列并解析
            output_sample = self.data.iloc[data_idx, self.output_col].tolist()
            # output_list = ast.literal_eval(output_sample) if isinstance(output_sample, str) else output_sample
            # output2_sample = self.data.iloc[data_idx, self.output2_col]
            # output_list.extend(output2_sample)

            
            # 转换为PyTorch张量
            input_tensor = torch.tensor(input_normalized, dtype=torch.float32)
            output_tensor = torch.tensor(output_sample, dtype=torch.float32)  # 使用转换后的dB值
            # bowl_tensor = torch.tensor(bowl_sample, dtype=torch.int64)
            mode_tensor = torch.tensor(mode_sample, dtype=torch.int64)
            # 验证输出长度
            # if len(output_tensor) != self.output_length:
            #     raise ValueError(f"输出序列长度不匹配: 预期 {self.output_length}, 实际 {len(output_tensor)}")
            
            return input_tensor, output_tensor, mode_tensor
            
        except Exception as e:
            print(f"获取样本 {idx} 时出错: {str(e)}")
            raise
    
    def get_validation_dataset(self):
        """创建验证集数据集实例"""
        if self.is_validation:
            raise ValueError("当前实例已是验证集")
            
        # 传递必要的参数给验证集
        val_norm_params = {
            'input_mean': self.input_mean,
            'input_std': self.input_std
        }
        
        val_dataset = Soundquality_four_data(
            directory_path=self.directory_path,
            input_cols=self.input_cols,
            output_col=self.output_col,
            bowl_col=self.bowl_col,
            mode_col=self.mode_col,
            val_split=self.val_split,
            is_validation=True,
            norm_params=val_norm_params,
            random_seed=self.random_seed
        )
        
        # 共享编码器
        val_dataset.bowl_encoder = self.bowl_encoder
        val_dataset.mode_encoder = self.mode_encoder
        
        # 设置验证集索引
        val_dataset.data = self.data
        val_dataset.val_indices = self.val_indices
        val_dataset.indices = self.val_indices
        val_dataset.column_names = self.column_names
        val_dataset.output_length = self.output_length
        
        return val_dataset



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    directory_path = "../csvdata333"
    
    # 创建训练集
    # train_dataset = NonLinearTypeBindata(directory_path, val_split=0.2)
    train_dataset = SPLdata(directory_path, val_split=0.2)
    
    # 获取验证集
    val_dataset = train_dataset.get_validation_dataset()
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 打印编码器类别信息
    # encoder_classes = {
    #     'bowl_classes': train_dataset.bowl_encoder.classes_,
    #     'mode_classes': train_dataset.mode_encoder.classes_
    # }
    encoder_classes = {
        'type_classes': train_dataset.type_encoder.classes_,
        'mode_classes': train_dataset.mode_encoder.classes_
    }
    print("空调型号类别:", encoder_classes['type_classes'])
    print("模式类型类别:", encoder_classes['mode_classes'])
    
    # 查看训练集批次
    print("\n训练集数据:")
    for batch_input, batch_output,batch_type,batch_mode in train_loader:
        print("输入批次形状:", batch_input.shape)
        print("输出批次形状:", batch_output.shape)
        print("类型批次形状:", batch_type.shape)
        print("模式批次形状:", batch_mode.shape)
        break
    
    # 查看验证集批次
    print("\n验证集数据:")
    for batch_input, batch_output, batch_type, batch_mode in val_loader:
        print("输入批次形状:", batch_input.shape)
        print("输入数据范围: [{:.4f}, {:.4f}]".format(batch_input.min(), batch_input.max()))
        break

