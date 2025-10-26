"""
推理和评估模块 - 模型推理、评估和可视化
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle
import json
from typing import Dict, List, Tuple, Optional

from config import Config
from models import ModelFactory
from data_processor import DataProcessor

class ModelInference:
    """模型推理类"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        self.models = {}
        self.scalers = {}
        
    def load_model(self, model_path: str, group_name: str, model_type: str):
        """加载训练好的模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 创建模型
            model = ModelFactory.create_model(
                model_type, 
                self.config.INPUT_DIM, 
                self.config.OUTPUT_DIM, 
                self.config
            )
            
            # 加载权重
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            self.models[f"{group_name}_{model_type}"] = model
            
            print(f"成功加载模型: {group_name}_{model_type}")
            return model
            
        except Exception as e:
            print(f"加载模型失败 {model_path}: {e}")
            return None
    
    def load_scalers(self, scaler_dir: str = 'scalers'):
        """加载数据缩放器"""
        try:
            if not os.path.exists(scaler_dir):
                print(f"缩放器目录不存在: {scaler_dir}")
                return
                
            scaler_files = [f for f in os.listdir(scaler_dir) if f.endswith('.pkl')]
            
            for scaler_file in scaler_files:
                scaler_path = os.path.join(scaler_dir, scaler_file)
                
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                
                # 解析文件名获取组名和类型
                # 文件名格式: group_1_input_scaler.pkl 或 group_1_output_scaler.pkl
                parts = scaler_file.replace('.pkl', '').split('_')
                if len(parts) >= 3:
                    group_name = '_'.join(parts[:-2])  # group_1
                    scaler_type = parts[-2]  # input 或 output
                    
                    if group_name not in self.scalers:
                        self.scalers[group_name] = {}
                    
                    self.scalers[group_name][scaler_type] = scaler
                else:
                    print(f"警告: 无法解析缩放器文件名: {scaler_file}")
            
            print(f"成功加载 {len(scaler_files)} 个缩放器")
            
        except Exception as e:
            print(f"加载缩放器失败: {e}")
    
    def predict(self, model_key: str, X: np.ndarray, group_name: str = None) -> np.ndarray:
        """使用指定模型进行预测"""
        if model_key not in self.models:
            raise ValueError(f"模型 {model_key} 未加载")
        
        model = self.models[model_key]
        
        # 数据预处理
        if group_name and group_name in self.scalers and 'input' in self.scalers[group_name]:
            X_scaled = self.scalers[group_name]['input'].transform(X)
        else:
            X_scaled = X
            print("警告: 未找到输入缩放器，使用原始数据")
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # 预测
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()
        
        # 反向缩放
        if group_name and group_name in self.scalers and 'output' in self.scalers[group_name]:
            predictions = self.scalers[group_name]['output'].inverse_transform(predictions)
        else:
            print("警告: 未找到输出缩放器，返回缩放后的预测")
        
        return predictions
    
    def batch_predict(self, X: np.ndarray, group_name: str, model_types: List[str] = None) -> Dict[str, np.ndarray]:
        """使用多个模型进行批量预测"""
        if model_types is None:
            model_types = ['mlp', 'resnet', 'transformer']
        
        predictions = {}
        
        for model_type in model_types:
            model_key = f"{group_name}_{model_type}"
            
            if model_key in self.models:
                try:
                    pred = self.predict(model_key, X, group_name)
                    predictions[model_type] = pred
                except Exception as e:
                    print(f"预测失败 {model_key}: {e}")
            else:
                print(f"模型 {model_key} 未加载")
        
        return predictions
    
    def ensemble_predict(self, X: np.ndarray, group_name: str, model_types: List[str] = None, weights: List[float] = None) -> np.ndarray:
        """集成预测"""
        predictions = self.batch_predict(X, group_name, model_types)
        
        if not predictions:
            raise ValueError("没有可用的预测结果")
        
        # 默认等权重
        if weights is None:
            weights = [1.0 / len(predictions)] * len(predictions)
        
        # 加权平均
        ensemble_pred = None
        for i, (model_type, pred) in enumerate(predictions.items()):
            if ensemble_pred is None:
                ensemble_pred = pred * weights[i]
            else:
                ensemble_pred += pred * weights[i]
        
        return ensemble_pred

class ModelEvaluator:
    """模型评估类"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.inference = ModelInference(config)
    
    def evaluate_model(self, model_key: str, X_test: np.ndarray, y_test: np.ndarray, group_name: str = None) -> Dict:
        """评估单个模型"""
        try:
            # 预测
            y_pred = self.inference.predict(model_key, X_test, group_name)
            
            # 计算指标
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # 计算每个输出维度的指标
            dim_metrics = {}
            for i in range(y_test.shape[1]):
                dim_mse = mean_squared_error(y_test[:, i], y_pred[:, i])
                dim_mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
                dim_r2 = r2_score(y_test[:, i], y_pred[:, i])
                
                dim_metrics[f'dim_{i}'] = {
                    'mse': dim_mse,
                    'mae': dim_mae,
                    'r2': dim_r2
                }
            
            results = {
                'overall': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': rmse
                },
                'per_dimension': dim_metrics,
                'predictions': y_pred,
                'targets': y_test
            }
            
            return results
            
        except Exception as e:
            print(f"评估模型 {model_key} 失败: {e}")
            return None
    
    def evaluate_all_models(self, test_data: Dict, model_types: List[str] = None) -> Dict:
        """评估所有模型"""
        if model_types is None:
            model_types = ['mlp', 'resnet', 'transformer']
        
        all_results = {}
        
        for group_name, group_data in test_data.items():
            print(f"\n评估 {group_name}...")
            
            X_test = group_data['X']
            y_test = group_data['y']
            
            group_results = {}
            
            for model_type in model_types:
                model_key = f"{group_name}_{model_type}"
                
                if model_key in self.inference.models:
                    print(f"  评估 {model_type} 模型...")
                    results = self.evaluate_model(model_key, X_test, y_test, group_name)
                    
                    if results:
                        group_results[model_type] = results
                        
                        # 打印结果
                        overall = results['overall']
                        print(f"    MSE: {overall['mse']:.6f}")
                        print(f"    MAE: {overall['mae']:.6f}")
                        print(f"    R²: {overall['r2']:.4f}")
                        print(f"    RMSE: {overall['rmse']:.6f}")
                else:
                    print(f"  模型 {model_key} 未加载，跳过评估")
            
            all_results[group_name] = group_results
        
        return all_results
    
    def compare_models(self, evaluation_results: Dict) -> pd.DataFrame:
        """比较不同模型的性能"""
        comparison_data = []
        
        for group_name, group_results in evaluation_results.items():
            for model_type, results in group_results.items():
                if results and 'overall' in results:
                    overall = results['overall']
                    comparison_data.append({
                        'Group': group_name,
                        'Model': model_type,
                        'MSE': overall['mse'],
                        'MAE': overall['mae'],
                        'R²': overall['r2'],
                        'RMSE': overall['rmse']
                    })
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def plot_predictions_vs_targets(self, evaluation_results: Dict, save_dir: str = 'plots'):
        """绘制预测值vs真实值的散点图"""
        os.makedirs(save_dir, exist_ok=True)
        
        for group_name, group_results in evaluation_results.items():
            n_models = len(group_results)
            if n_models == 0:
                continue
            
            fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
            if n_models == 1:
                axes = [axes]
            
            fig.suptitle(f'{group_name} - 预测值 vs 真实值', fontsize=16)
            
            for i, (model_type, results) in enumerate(group_results.items()):
                if not results or 'predictions' not in results:
                    continue
                
                y_pred = results['predictions']
                y_true = results['targets']
                
                # 计算整体的预测值和真实值（所有维度的平均值）
                y_pred_mean = np.mean(y_pred, axis=1)
                y_true_mean = np.mean(y_true, axis=1)
                
                # 散点图
                axes[i].scatter(y_true_mean, y_pred_mean, alpha=0.6, s=20)
                
                # 对角线
                min_val = min(y_true_mean.min(), y_pred_mean.min())
                max_val = max(y_true_mean.max(), y_pred_mean.max())
                axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                
                axes[i].set_xlabel('真实值')
                axes[i].set_ylabel('预测值')
                axes[i].set_title(f'{model_type}\nR² = {results["overall"]["r2"]:.4f}')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = os.path.join(save_dir, f'{group_name}_predictions_vs_targets.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, save_dir: str = 'plots'):
        """绘制模型比较图"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置图形样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型性能比较', fontsize=16)
        
        metrics = ['MSE', 'MAE', 'R²', 'RMSE']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            # 分组柱状图
            sns.barplot(data=comparison_df, x='Group', y=metric, hue='Model', ax=ax)
            ax.set_title(f'{metric} 比较')
            ax.tick_params(axis='x', rotation=45)
            
            # 对于R²，值越大越好，其他指标值越小越好
            if metric == 'R²':
                ax.set_ylim(0, 1)
            
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, 'model_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_evaluation_report(self, evaluation_results: Dict, save_path: str = 'evaluation_report.txt'):
        """生成评估报告"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            for group_name, group_results in evaluation_results.items():
                f.write(f"{group_name} 评估结果:\n")
                f.write("-" * 30 + "\n")
                
                for model_type, results in group_results.items():
                    if results and 'overall' in results:
                        overall = results['overall']
                        f.write(f"\n{model_type} 模型:\n")
                        f.write(f"  MSE: {overall['mse']:.6f}\n")
                        f.write(f"  MAE: {overall['mae']:.6f}\n")
                        f.write(f"  R²: {overall['r2']:.4f}\n")
                        f.write(f"  RMSE: {overall['rmse']:.6f}\n")
                        
                        # 评估模型性能等级
                        mse = overall['mse']
                        if mse <= self.config.TARGET_MSE['excellent']:
                            level = "优秀"
                        elif mse <= self.config.TARGET_MSE['good']:
                            level = "良好"
                        elif mse <= self.config.TARGET_MSE['acceptable']:
                            level = "可接受"
                        else:
                            level = "需要改进"
                        
                        f.write(f"  性能等级: {level}\n")
                
                f.write("\n")
        
        print(f"评估报告已保存到: {save_path}")

def load_all_models(model_dir: str = 'models', scaler_dir: str = 'scalers'):
    """加载所有训练好的模型"""
    evaluator = ModelEvaluator()
    
    # 加载缩放器
    if os.path.exists(scaler_dir):
        evaluator.inference.load_scalers(scaler_dir)
    
    # 加载模型
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            
            # 解析文件名获取组名和模型类型
            parts = model_file.replace('_best.pth', '').split('_')
            if len(parts) >= 2:
                group_name = '_'.join(parts[:-1])
                model_type = parts[-1]
                
                evaluator.inference.load_model(model_path, group_name, model_type)
    
    return evaluator

if __name__ == "__main__":
    # 测试推理和评估
    print("加载模型和评估器...")
    
    evaluator = load_all_models()
    
    # 如果有测试数据，进行评估
    if evaluator.inference.models:
        print(f"成功加载 {len(evaluator.inference.models)} 个模型")
        
        # 这里需要测试数据，实际使用时需要提供
        # test_data = {...}  # 测试数据
        # results = evaluator.evaluate_all_models(test_data)
        # evaluator.generate_evaluation_report(results)
        
    else:
        print("未找到训练好的模型，请先运行训练脚本")