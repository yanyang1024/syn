"""
训练器模块 - 支持多组数据分别训练不同模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
from tqdm import tqdm
from datetime import datetime

from config import Config
from data_processor import DataProcessor
from models import ModelFactory

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=20, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

class Trainer:
    """训练器类"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.train_history = {}
        
        # 创建保存目录
        os.makedirs(self.config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(self.config.LOG_DIR, exist_ok=True)
        
        print(f"使用设备: {self.device}")
    
    def create_model_and_optimizer(self, model_type, input_dim, output_dim, group_name):
        """为指定组创建模型和优化器"""
        
        # 创建模型
        model = ModelFactory.create_model(model_type, input_dim, output_dim, self.config)
        model = model.to(self.device)
        
        # 创建优化器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # 创建学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # 保存到字典
        self.models[group_name] = model
        self.optimizers[group_name] = optimizer
        self.schedulers[group_name] = scheduler
        
        # 打印模型信息
        model_info = ModelFactory.get_model_info(model)
        print(f"\n{group_name} - {model_type} 模型:")
        print(f"参数数量: {model_info['total_parameters']:,}")
        print(f"模型大小: {model_info['model_size_mb']:.2f} MB")
        
        return model, optimizer, scheduler
    
    def train_epoch(self, model, dataloader, optimizer, criterion, epoch, group_name):
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f'{group_name} Epoch {epoch}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.6f}'
            })
        
        return total_loss / num_batches
    
    def validate_epoch(self, model, dataloader, criterion):
        """验证一个epoch"""
        model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # 计算额外指标
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        r2 = self.calculate_r2(targets, predictions)
        
        return total_loss / len(dataloader), mse, mae, r2, predictions, targets
    
    def calculate_r2(self, y_true, y_pred):
        """计算R²分数"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def train_group(self, group_name, group_data, model_type='mlp', epochs=None):
        """训练单个数据组"""
        epochs = epochs or self.config.EPOCHS
        
        print(f"\n开始训练 {group_name} ({model_type} 模型)")
        print("=" * 50)
        
        # 获取数据加载器
        train_loader = group_data['dataloaders']['train']
        val_loader = group_data['dataloaders']['val']
        test_loader = group_data['dataloaders']['test']
        
        # 创建模型和优化器
        model, optimizer, scheduler = self.create_model_and_optimizer(
            model_type, self.config.INPUT_DIM, self.config.OUTPUT_DIM, group_name
        )
        
        # 损失函数
        criterion = nn.MSELoss()
        
        # 早停机制
        early_stopping = EarlyStopping(patience=self.config.PATIENCE)
        
        # TensorBoard记录器
        log_dir = os.path.join(self.config.LOG_DIR, f"{group_name}_{model_type}")
        writer = SummaryWriter(log_dir)
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
            'val_mae': [],
            'val_r2': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        # 训练循环
        for epoch in range(1, epochs + 1):
            # 训练
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, epoch, group_name)
            
            # 验证
            val_loss, val_mse, val_mae, val_r2, val_pred, val_true = self.validate_epoch(
                model, val_loader, criterion
            )
            
            # 学习率调度
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mse'].append(val_mse)
            history['val_mae'].append(val_mae)
            history['val_r2'].append(val_r2)
            history['learning_rate'].append(current_lr)
            
            # TensorBoard记录
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Metrics/MSE', val_mse, epoch)
            writer.add_scalar('Metrics/MAE', val_mae, epoch)
            writer.add_scalar('Metrics/R2', val_r2, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # 打印进度
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, MSE: {val_mse:.6f}, "
                      f"MAE: {val_mae:.6f}, R²: {val_r2:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(model, group_name, model_type, epoch, val_loss)
            
            # 早停检查
            if early_stopping(val_loss, model):
                print(f"早停触发，在第 {epoch} 轮停止训练")
                break
        
        # 训练完成
        training_time = time.time() - start_time
        print(f"\n{group_name} 训练完成!")
        print(f"训练时间: {training_time:.2f} 秒")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        
        # 最终测试
        test_loss, test_mse, test_mae, test_r2, test_pred, test_true = self.validate_epoch(
            model, test_loader, criterion
        )
        
        print(f"测试结果: Loss: {test_loss:.6f}, MSE: {test_mse:.6f}, "
              f"MAE: {test_mae:.6f}, R²: {test_r2:.4f}")
        
        # 保存训练历史
        history['final_test'] = {
            'loss': test_loss,
            'mse': test_mse,
            'mae': test_mae,
            'r2': test_r2
        }
        
        self.train_history[group_name] = history
        
        # 关闭TensorBoard
        writer.close()
        
        # 绘制训练曲线
        self.plot_training_curves(group_name, history)
        
        return model, history
    
    def save_model(self, model, group_name, model_type, epoch, val_loss):
        """保存模型"""
        # 确保保存目录存在
        os.makedirs(self.config.MODEL_SAVE_DIR, exist_ok=True)
        
        model_path = os.path.join(
            self.config.MODEL_SAVE_DIR, 
            f"{group_name}_{model_type}_best.pth"
        )
        
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'model_type': model_type,
                'group_name': group_name,
                'config': self.config.__dict__
            }, model_path)
            print(f"模型已保存到: {model_path}")
        except Exception as e:
            print(f"保存模型失败: {e}")
    
    def plot_training_curves(self, group_name, history):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{group_name} 训练曲线', fontsize=16)
        
        # 损失曲线
        axes[0, 0].plot(history['train_loss'], label='训练损失', color='blue')
        axes[0, 0].plot(history['val_loss'], label='验证损失', color='red')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MSE曲线
        axes[0, 1].plot(history['val_mse'], label='验证MSE', color='green')
        axes[0, 1].set_title('MSE曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # R²曲线
        axes[1, 0].plot(history['val_r2'], label='验证R²', color='purple')
        axes[1, 0].set_title('R²曲线')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 学习率曲线
        axes[1, 1].plot(history['learning_rate'], label='学习率', color='orange')
        axes[1, 1].set_title('学习率曲线')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(self.config.LOG_DIR, f"{group_name}_training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_all_groups(self, processed_groups, model_types=None):
        """训练所有数据组"""
        if model_types is None:
            model_types = ['mlp', 'resnet', 'transformer']
        
        print("开始训练所有数据组...")
        print(f"将训练的模型类型: {model_types}")
        
        all_results = {}
        
        for group_name, group_data in processed_groups.items():
            group_results = {}
            
            for model_type in model_types:
                print(f"\n{'='*60}")
                print(f"训练 {group_name} - {model_type} 模型")
                print(f"{'='*60}")
                
                try:
                    model, history = self.train_group(group_name, group_data, model_type)
                    group_results[model_type] = history
                    
                except Exception as e:
                    print(f"训练 {group_name} - {model_type} 时出错: {e}")
                    continue
            
            all_results[group_name] = group_results
        
        # 保存所有结果
        self.save_all_results(all_results)
        
        # 生成总结报告
        self.generate_summary_report(all_results)
        
        return all_results
    
    def save_all_results(self, results):
        """保存所有训练结果"""
        results_path = os.path.join(self.config.LOG_DIR, "training_results.json")
        
        # 转换numpy数组为列表以便JSON序列化
        json_results = {}
        for group_name, group_results in results.items():
            json_results[group_name] = {}
            for model_type, history in group_results.items():
                json_results[group_name][model_type] = {
                    key: value if not isinstance(value, np.ndarray) else value.tolist()
                    for key, value in history.items()
                }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"训练结果已保存到: {results_path}")
    
    def generate_summary_report(self, results):
        """生成训练总结报告"""
        report_path = os.path.join(self.config.LOG_DIR, "training_summary.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("训练总结报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for group_name, group_results in results.items():
                f.write(f"{group_name} 结果:\n")
                f.write("-" * 30 + "\n")
                
                for model_type, history in group_results.items():
                    if 'final_test' in history:
                        test_results = history['final_test']
                        f.write(f"{model_type}:\n")
                        f.write(f"  测试MSE: {test_results['mse']:.6f}\n")
                        f.write(f"  测试MAE: {test_results['mae']:.6f}\n")
                        f.write(f"  测试R²: {test_results['r2']:.4f}\n")
                        f.write(f"  最终损失: {test_results['loss']:.6f}\n\n")
                
                f.write("\n")
        
        print(f"训练总结报告已保存到: {report_path}")

if __name__ == "__main__":
    # 测试训练器
    print("初始化训练器...")
    
    # 创建数据处理器和训练器
    data_processor = DataProcessor()
    trainer = Trainer()
    
    # 处理数据
    print("处理数据...")
    processed_groups = data_processor.process_all_groups()
    
    # 训练所有组（仅使用MLP进行快速测试）
    print("开始训练...")
    results = trainer.train_all_groups(processed_groups, model_types=['mlp'])
    
    print("训练完成!")