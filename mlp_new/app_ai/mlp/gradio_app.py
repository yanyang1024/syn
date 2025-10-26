"""
Gradio可视化界面 - 训练监控、参数调整和推理测试
"""
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import threading
import time
from datetime import datetime

from config import Config
from data_processor import DataProcessor
from trainer import Trainer
from inference import load_all_models
from models import ModelFactory

class GradioApp:
    """Gradio应用程序类"""
    
    def __init__(self):
        self.config = Config()
        self.data_processor = None
        self.trainer = None
        self.evaluator = None
        self.processed_groups = None
        self.training_thread = None
        self.training_status = {"is_training": False, "progress": 0, "current_group": "", "current_model": ""}
        
        # 初始化组件
        self.initialize_components()
    
    def initialize_components(self):
        """初始化组件"""
        try:
            self.data_processor = DataProcessor(self.config)
            self.trainer = Trainer(self.config)
            self.evaluator = load_all_models()
            print("组件初始化成功")
        except Exception as e:
            print(f"组件初始化失败: {e}")
    
    def load_data(self, x_file, y_file):
        """加载数据"""
        try:
            if x_file is None or y_file is None:
                # 使用示例数据
                X, y = self.data_processor._generate_sample_data()
                message = "使用生成的示例数据"
            else:
                # 直接使用上传文件的路径
                X, y = self.data_processor.load_data(x_file.name, y_file.name)
                message = f"成功加载数据: X={X.shape}, y={y.shape}"
            
            # 处理数据
            self.processed_groups = self.data_processor.process_all_groups(X, y)
            
            # 生成数据统计信息
            stats = self.generate_data_stats()
            
            return message, stats
            
        except Exception as e:
            return f"数据加载失败: {e}", ""
    
    def generate_data_stats(self):
        """生成数据统计信息"""
        if not self.processed_groups:
            return "无数据"
        
        stats = "数据统计信息:\n"
        stats += "=" * 30 + "\n"
        
        for group_name, group_data in self.processed_groups.items():
            train_size = len(group_data['data_splits']['train'][0])
            val_size = len(group_data['data_splits']['val'][0])
            test_size = len(group_data['data_splits']['test'][0])
            
            stats += f"{group_name}:\n"
            stats += f"  训练集: {train_size} 样本\n"
            stats += f"  验证集: {val_size} 样本\n"
            stats += f"  测试集: {test_size} 样本\n"
            stats += f"  总计: {train_size + val_size + test_size} 样本\n\n"
        
        return stats
    
    def update_config(self, learning_rate, batch_size, epochs, hidden_dims, dropout_rate):
        """更新配置参数"""
        try:
            # 验证参数范围
            if learning_rate <= 0 or learning_rate > 1:
                return "学习率必须在 (0, 1] 范围内"
            if batch_size <= 0 or batch_size > 1024:
                return "批次大小必须在 (0, 1024] 范围内"
            if epochs <= 0 or epochs > 1000:
                return "训练轮数必须在 (0, 1000] 范围内"
            if dropout_rate < 0 or dropout_rate >= 1:
                return "Dropout率必须在 [0, 1) 范围内"
            
            self.config.LEARNING_RATE = learning_rate
            self.config.BATCH_SIZE = int(batch_size)
            self.config.EPOCHS = int(epochs)
            self.config.DROPOUT_RATE = dropout_rate
            
            # 解析隐藏层维度
            if hidden_dims:
                try:
                    dims = [int(x.strip()) for x in hidden_dims.split(',')]
                    if any(d <= 0 for d in dims):
                        return "隐藏层维度必须为正整数"
                    self.config.HIDDEN_DIMS = dims
                except ValueError:
                    return "隐藏层维度格式错误，请使用逗号分隔的整数"
            
            return "配置更新成功"
        except Exception as e:
            return f"配置更新失败: {e}"
    
    def start_training(self, selected_groups, selected_models, progress=gr.Progress()):
        """开始训练"""
        if not self.processed_groups:
            return "请先加载数据"
        
        if self.training_status["is_training"]:
            return "训练正在进行中，请等待完成"
        
        try:
            # 过滤选中的组和模型
            groups_to_train = {k: v for k, v in self.processed_groups.items() if k in selected_groups}
            
            if not groups_to_train:
                return "请选择要训练的数据组"
            
            if not selected_models:
                return "请选择要训练的模型类型"
            
            # 启动训练线程
            self.training_thread = threading.Thread(
                target=self._train_models,
                args=(groups_to_train, selected_models, progress)
            )
            self.training_thread.start()
            
            return "训练已开始，请查看训练状态"
            
        except Exception as e:
            return f"启动训练失败: {e}"
    
    def _train_models(self, groups_to_train, selected_models, progress):
        """训练模型（在后台线程中运行）"""
        self.training_status["is_training"] = True
        
        try:
            total_tasks = len(groups_to_train) * len(selected_models)
            current_task = 0
            
            for group_name, group_data in groups_to_train.items():
                for model_type in selected_models:
                    self.training_status["current_group"] = group_name
                    self.training_status["current_model"] = model_type
                    
                    # 训练模型
                    self.trainer.train_group(group_name, group_data, model_type)
                    
                    current_task += 1
                    self.training_status["progress"] = (current_task / total_tasks) * 100
                    
                    if progress:
                        progress((current_task / total_tasks), f"训练 {group_name} - {model_type}")
            
            self.training_status["is_training"] = False
            self.training_status["progress"] = 100
            
        except Exception as e:
            print(f"训练过程中出错: {e}")
            self.training_status["is_training"] = False
    
    def get_training_status(self):
        """获取训练状态"""
        if self.training_status["is_training"]:
            return f"训练中... {self.training_status['progress']:.1f}%\n" \
                   f"当前: {self.training_status['current_group']} - {self.training_status['current_model']}"
        else:
            if self.training_status["progress"] == 100:
                return "训练完成"
            else:
                return "未开始训练"
    
    def load_training_results(self):
        """加载训练结果"""
        try:
            results_path = os.path.join(self.config.LOG_DIR, "training_results.json")
            
            if not os.path.exists(results_path):
                return "未找到训练结果文件"
            
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # 生成结果摘要
            summary = "训练结果摘要:\n"
            summary += "=" * 30 + "\n"
            
            for group_name, group_results in results.items():
                summary += f"{group_name}:\n"
                
                for model_type, history in group_results.items():
                    if 'final_test' in history:
                        test_results = history['final_test']
                        summary += f"  {model_type}:\n"
                        summary += f"    MSE: {test_results['mse']:.6f}\n"
                        summary += f"    MAE: {test_results['mae']:.6f}\n"
                        summary += f"    R²: {test_results['r2']:.4f}\n\n"
                
                summary += "\n"
            
            return summary
            
        except Exception as e:
            return f"加载训练结果失败: {e}"
    
    def create_training_plots(self, selected_group):
        """创建训练曲线图"""
        try:
            results_path = os.path.join(self.config.LOG_DIR, "training_results.json")
            
            if not os.path.exists(results_path):
                return None
            
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            if selected_group not in results:
                return None
            
            group_results = results[selected_group]
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('训练/验证损失', 'MSE', 'R²分数', '学习率'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": True}]]
            )
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            
            for i, (model_type, history) in enumerate(group_results.items()):
                color = colors[i % len(colors)]
                
                epochs = list(range(1, len(history['train_loss']) + 1))
                
                # 训练/验证损失
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['train_loss'], 
                              name=f'{model_type} - 训练损失', 
                              line=dict(color=color, dash='solid')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['val_loss'], 
                              name=f'{model_type} - 验证损失', 
                              line=dict(color=color, dash='dash')),
                    row=1, col=1
                )
                
                # MSE
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['val_mse'], 
                              name=f'{model_type} - MSE', 
                              line=dict(color=color)),
                    row=1, col=2
                )
                
                # R²
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['val_r2'], 
                              name=f'{model_type} - R²', 
                              line=dict(color=color)),
                    row=2, col=1
                )
                
                # 学习率
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['learning_rate'], 
                              name=f'{model_type} - 学习率', 
                              line=dict(color=color)),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, title_text=f"{selected_group} 训练曲线")
            fig.update_yaxes(type="log", row=2, col=2)  # 学习率使用对数坐标
            
            return fig
            
        except Exception as e:
            print(f"创建训练图表失败: {e}")
            return None
    
    def predict_single(self, input_values, selected_group, selected_model):
        """单个预测"""
        try:
            if not self.evaluator or not self.evaluator.inference.models:
                return "请先训练模型或加载已训练的模型"
            
            # 解析输入值
            if isinstance(input_values, str):
                values = [float(x.strip()) for x in input_values.split(',')]
            else:
                values = input_values
            
            if len(values) != self.config.INPUT_DIM:
                return f"输入维度错误，期望 {self.config.INPUT_DIM} 个值"
            
            # 转换为numpy数组
            X = np.array(values).reshape(1, -1)
            
            # 预测
            model_key = f"{selected_group}_{selected_model}"
            
            if model_key not in self.evaluator.inference.models:
                return f"模型 {model_key} 未加载"
            
            prediction = self.evaluator.inference.predict(model_key, X, selected_group)
            
            # 格式化输出
            result = f"预测结果 ({selected_group} - {selected_model}):\n"
            result += "=" * 40 + "\n"
            
            for i, value in enumerate(prediction[0]):
                result += f"输出 {i+1}: {value:.6f}\n"
            
            return result
            
        except Exception as e:
            return f"预测失败: {e}"
    
    def batch_predict(self, csv_file, selected_group, selected_models):
        """批量预测"""
        try:
            if not self.evaluator or not self.evaluator.inference.models:
                return "请先训练模型或加载已训练的模型", None
            
            if csv_file is None:
                return "请上传CSV文件", None
            
            # 读取CSV文件
            df = pd.read_csv(csv_file.name)
            X = df.values
            
            if X.shape[1] != self.config.INPUT_DIM:
                return f"输入维度错误，期望 {self.config.INPUT_DIM} 个特征", None
            
            # 批量预测
            predictions = self.evaluator.inference.batch_predict(X, selected_group, selected_models)
            
            if not predictions:
                return "没有可用的预测结果", None
            
            # 创建结果DataFrame
            result_df = df.copy()
            
            for model_type, pred in predictions.items():
                for i in range(pred.shape[1]):
                    result_df[f'{model_type}_output_{i+1}'] = pred[:, i]
            
            # 保存结果
            output_path = f"batch_predictions_{selected_group}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            result_df.to_csv(output_path, index=False)
            
            return f"批量预测完成，结果已保存到 {output_path}", result_df
            
        except Exception as e:
            return f"批量预测失败: {e}", None
    
    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(title="深度学习回归预测系统", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🧠 深度学习回归预测系统")
            gr.Markdown("多输入多输出回归预测，支持MLP、ResNet、Transformer等多种模型架构")
            
            with gr.Tabs():
                # 数据加载标签页
                with gr.TabItem("📊 数据加载"):
                    gr.Markdown("## 数据加载和预处理")
                    
                    with gr.Row():
                        x_file = gr.File(label="输入数据文件 (x_input.csv)", file_types=[".csv"])
                        y_file = gr.File(label="输出数据文件 (y_output.csv)", file_types=[".csv"])
                    
                    load_btn = gr.Button("加载数据", variant="primary")
                    load_status = gr.Textbox(label="加载状态", lines=2)
                    data_stats = gr.Textbox(label="数据统计", lines=10)
                    
                    load_btn.click(
                        self.load_data,
                        inputs=[x_file, y_file],
                        outputs=[load_status, data_stats]
                    )
                
                # 训练配置标签页
                with gr.TabItem("⚙️ 训练配置"):
                    gr.Markdown("## 模型训练配置")
                    
                    with gr.Row():
                        with gr.Column():
                            learning_rate = gr.Number(label="学习率", value=self.config.LEARNING_RATE)
                            batch_size = gr.Number(label="批次大小", value=self.config.BATCH_SIZE, precision=0)
                            epochs = gr.Number(label="训练轮数", value=self.config.EPOCHS, precision=0)
                        
                        with gr.Column():
                            hidden_dims = gr.Textbox(
                                label="隐藏层维度 (逗号分隔)", 
                                value=",".join(map(str, self.config.HIDDEN_DIMS))
                            )
                            dropout_rate = gr.Number(label="Dropout率", value=self.config.DROPOUT_RATE)
                    
                    config_btn = gr.Button("更新配置", variant="secondary")
                    config_status = gr.Textbox(label="配置状态")
                    
                    config_btn.click(
                        self.update_config,
                        inputs=[learning_rate, batch_size, epochs, hidden_dims, dropout_rate],
                        outputs=[config_status]
                    )
                
                # 模型训练标签页
                with gr.TabItem("🚀 模型训练"):
                    gr.Markdown("## 模型训练")
                    
                    with gr.Row():
                        with gr.Column():
                            selected_groups = gr.CheckboxGroup(
                                choices=["group_1", "group_2", "group_3", "group_4", "group_5"],
                                label="选择数据组",
                                value=["group_1"]
                            )
                            
                            selected_models = gr.CheckboxGroup(
                                choices=["mlp", "resnet", "transformer"],
                                label="选择模型类型",
                                value=["mlp"]
                            )
                        
                        with gr.Column():
                            train_btn = gr.Button("开始训练", variant="primary")
                            training_status = gr.Textbox(label="训练状态", lines=3)
                            
                            # 使用定时器定期更新训练状态
                            def update_status():
                                return self.get_training_status()
                            
                            # 注意：在实际使用中，可能需要手动刷新页面来查看训练状态
                            # 或者使用其他方式实现实时更新
                    
                    train_btn.click(
                        self.start_training,
                        inputs=[selected_groups, selected_models],
                        outputs=[training_status]
                    )
                
                # 训练结果标签页
                with gr.TabItem("📈 训练结果"):
                    gr.Markdown("## 训练结果分析")
                    
                    with gr.Row():
                        results_btn = gr.Button("加载训练结果", variant="secondary")
                        plot_group = gr.Dropdown(
                            choices=["group_1", "group_2", "group_3", "group_4", "group_5"],
                            label="选择组查看图表",
                            value="group_1"
                        )
                    
                    results_text = gr.Textbox(label="训练结果摘要", lines=15)
                    training_plot = gr.Plot(label="训练曲线")
                    
                    results_btn.click(
                        self.load_training_results,
                        outputs=[results_text]
                    )
                    
                    plot_group.change(
                        self.create_training_plots,
                        inputs=[plot_group],
                        outputs=[training_plot]
                    )
                
                # 模型预测标签页
                with gr.TabItem("🔮 模型预测"):
                    gr.Markdown("## 模型预测")
                    
                    with gr.Tabs():
                        # 单个预测
                        with gr.TabItem("单个预测"):
                            with gr.Row():
                                with gr.Column():
                                    input_values = gr.Textbox(
                                        label=f"输入值 (逗号分隔，{self.config.INPUT_DIM}个值)",
                                        placeholder="1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0"
                                    )
                                    
                                    pred_group = gr.Dropdown(
                                        choices=["group_1", "group_2", "group_3", "group_4", "group_5"],
                                        label="选择数据组",
                                        value="group_1"
                                    )
                                    
                                    pred_model = gr.Dropdown(
                                        choices=["mlp", "resnet", "transformer"],
                                        label="选择模型",
                                        value="mlp"
                                    )
                                
                                with gr.Column():
                                    predict_btn = gr.Button("预测", variant="primary")
                                    prediction_result = gr.Textbox(label="预测结果", lines=10)
                            
                            predict_btn.click(
                                self.predict_single,
                                inputs=[input_values, pred_group, pred_model],
                                outputs=[prediction_result]
                            )
                        
                        # 批量预测
                        with gr.TabItem("批量预测"):
                            with gr.Row():
                                with gr.Column():
                                    batch_file = gr.File(label="上传CSV文件", file_types=[".csv"])
                                    
                                    batch_group = gr.Dropdown(
                                        choices=["group_1", "group_2", "group_3", "group_4", "group_5"],
                                        label="选择数据组",
                                        value="group_1"
                                    )
                                    
                                    batch_models = gr.CheckboxGroup(
                                        choices=["mlp", "resnet", "transformer"],
                                        label="选择模型",
                                        value=["mlp"]
                                    )
                                
                                with gr.Column():
                                    batch_predict_btn = gr.Button("批量预测", variant="primary")
                                    batch_status = gr.Textbox(label="预测状态", lines=3)
                            
                            batch_results = gr.Dataframe(label="预测结果")
                            
                            batch_predict_btn.click(
                                self.batch_predict,
                                inputs=[batch_file, batch_group, batch_models],
                                outputs=[batch_status, batch_results]
                            )
        
        return app

def main():
    """主函数"""
    print("初始化Gradio应用...")
    
    app_instance = GradioApp()
    interface = app_instance.create_interface()
    
    print("启动Gradio界面...")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    main()