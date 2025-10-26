"""
数据处理模块 - 负责数据加载、预处理和分组
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
from config import Config

class RegressionDataset(Dataset):
    """回归数据集类"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.scalers = {'input': {}, 'output': {}}
        self.data_groups = {}
        
    def load_data(self, x_path=None, y_path=None):
        """加载数据"""
        x_path = x_path or self.config.X_INPUT_PATH
        y_path = y_path or self.config.Y_OUTPUT_PATH
        
        try:
            # 加载输入和输出数据
            X = pd.read_csv(x_path).values
            y = pd.read_csv(y_path).values
            
            print(f"数据加载成功:")
            print(f"输入数据形状: {X.shape}")
            print(f"输出数据形状: {y.shape}")
            
            # 验证数据维度
            if X.shape[1] != self.config.INPUT_DIM:
                print(f"警告: 输入维度不匹配，期望{self.config.INPUT_DIM}，实际{X.shape[1]}")
            if y.shape[1] != self.config.OUTPUT_DIM:
                print(f"警告: 输出维度不匹配，期望{self.config.OUTPUT_DIM}，实际{y.shape[1]}")
            
            return X, y
            
        except FileNotFoundError as e:
            print(f"数据文件未找到: {e}")
            # 生成示例数据用于测试
            return self._generate_sample_data()
    
    def _generate_sample_data(self):
        """生成示例数据用于测试"""
        print("生成示例数据用于测试...")
        total_samples = sum(self.config.GROUP_SIZES)
        
        # 生成随机数据
        np.random.seed(42)
        X = np.random.randn(total_samples, self.config.INPUT_DIM)
        
        # 生成具有一定相关性的输出数据
        # 使用线性变换 + 非线性变换 + 噪声
        W = np.random.randn(self.config.INPUT_DIM, self.config.OUTPUT_DIM) * 0.5
        y = X @ W + np.random.randn(total_samples, self.config.OUTPUT_DIM) * 0.1
        
        # 添加非线性关系
        y += np.sin(X[:, :3] @ np.random.randn(3, self.config.OUTPUT_DIM)) * 0.2
        
        print(f"生成数据形状: X={X.shape}, y={y.shape}")
        return X, y
    
    def split_data_by_groups(self, X, y):
        """按组分割数据"""
        # 验证总样本数
        total_expected = sum(self.config.GROUP_SIZES)
        if X.shape[0] != total_expected:
            print(f"警告: 数据总样本数不匹配，期望{total_expected}，实际{X.shape[0]}")
            print("将按实际数据量进行分割")
        
        groups = {}
        start_idx = 0
        
        for i, size in enumerate(self.config.GROUP_SIZES):
            end_idx = min(start_idx + size, X.shape[0])  # 防止越界
            if start_idx >= X.shape[0]:
                print(f"警告: 组 {i+1} 没有足够的数据")
                break
                
            actual_size = end_idx - start_idx
            groups[f'group_{i+1}'] = {
                'X': X[start_idx:end_idx],
                'y': y[start_idx:end_idx]
            }
            start_idx = end_idx
            print(f"组 {i+1}: {actual_size} 个样本 (期望{size})")
        
        self.data_groups = groups
        return groups
    
    def preprocess_data(self, X, y, group_name, scaler_type='standard'):
        """数据预处理（标准化/归一化）"""
        if scaler_type == 'standard':
            input_scaler = StandardScaler()
            output_scaler = StandardScaler()
        elif scaler_type == 'minmax':
            input_scaler = MinMaxScaler()
            output_scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        
        # 拟合并转换数据
        X_scaled = input_scaler.fit_transform(X)
        y_scaled = output_scaler.fit_transform(y)
        
        # 保存缩放器
        self.scalers['input'][group_name] = input_scaler
        self.scalers['output'][group_name] = output_scaler
        
        return X_scaled, y_scaled
    
    def create_data_splits(self, X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """创建训练/验证/测试数据分割"""
        # 首先分离出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=42
        )
        
        # 从剩余数据中分离训练集和验证集
        val_size = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42
        )
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def create_dataloaders(self, data_splits, batch_size=None):
        """创建数据加载器"""
        batch_size = batch_size or self.config.BATCH_SIZE
        
        dataloaders = {}
        for split_name, (X, y) in data_splits.items():
            dataset = RegressionDataset(X, y)
            shuffle = (split_name == 'train')
            dataloaders[split_name] = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle
            )
        
        return dataloaders
    
    def process_all_groups(self, X=None, y=None, scaler_type='standard'):
        """处理所有数据组"""
        if X is None or y is None:
            X, y = self.load_data()
        
        # 按组分割数据
        groups = self.split_data_by_groups(X, y)
        
        processed_groups = {}
        for group_name, group_data in groups.items():
            print(f"\n处理 {group_name}...")
            
            # 预处理数据
            X_scaled, y_scaled = self.preprocess_data(
                group_data['X'], group_data['y'], group_name, scaler_type
            )
            
            # 创建数据分割
            data_splits = self.create_data_splits(X_scaled, y_scaled)
            
            # 创建数据加载器
            dataloaders = self.create_dataloaders(data_splits)
            
            processed_groups[group_name] = {
                'data_splits': data_splits,
                'dataloaders': dataloaders,
                'raw_data': group_data
            }
            
            print(f"训练集: {len(data_splits['train'][0])} 样本")
            print(f"验证集: {len(data_splits['val'][0])} 样本")
            print(f"测试集: {len(data_splits['test'][0])} 样本")
        
        return processed_groups
    
    def inverse_transform_output(self, y_scaled, group_name):
        """反向转换输出数据"""
        if group_name in self.scalers['output']:
            return self.scalers['output'][group_name].inverse_transform(y_scaled)
        else:
            print(f"警告: 未找到组 {group_name} 的输出缩放器")
            return y_scaled
    
    def save_scalers(self, save_dir='scalers'):
        """保存缩放器"""
        os.makedirs(save_dir, exist_ok=True)
        
        for scaler_type in ['input', 'output']:
            for group_name, scaler in self.scalers[scaler_type].items():
                filename = f"{save_dir}/{group_name}_{scaler_type}_scaler.pkl"
                import pickle
                with open(filename, 'wb') as f:
                    pickle.dump(scaler, f)
        
        print(f"缩放器已保存到 {save_dir}")

if __name__ == "__main__":
    # 测试数据处理器
    processor = DataProcessor()
    processed_groups = processor.process_all_groups()
    
    print("\n数据处理完成!")
    print(f"处理了 {len(processed_groups)} 个数据组")
    
    # 保存缩放器
    processor.save_scalers()