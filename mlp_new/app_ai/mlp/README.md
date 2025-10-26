# 🧠 深度学习回归预测系统

一个完整的多输入多输出回归预测系统，支持多种神经网络架构（MLP、ResNet、Transformer）和可视化界面。

## 📋 项目概述

本项目实现了一个7维输入到26维输出的回归预测系统，支持5组不同规模的数据分别训练，提供了完整的训练、推理和可视化功能。

### 🎯 主要特性

- **多模型架构**: MLP、ResNet、Transformer、Ensemble
- **多组数据训练**: 支持5组不同规模数据分别训练
- **可视化界面**: 基于Gradio的Web界面
- **完整流程**: 数据处理、模型训练、推理评估
- **超参数优化**: 自动调参和性能分析

## 🏗️ 项目结构

```
mlp/
├── config.py                    # 配置文件
├── data_processor.py           # 数据处理模块
├── models.py                   # 神经网络模型定义
├── trainer.py                  # 训练器
├── inference.py                # 推理和评估
├── gradio_app.py              # Gradio可视化界面
├── hyperparameter_analysis.py  # 超参数分析
├── requirements.txt            # 依赖包
├── README.md                   # 项目说明
├── models/                     # 保存训练好的模型
├── logs/                       # 训练日志和结果
├── scalers/                    # 数据缩放器
└── plots/                      # 可视化图表
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone <repository-url>
cd mlp

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

准备两个CSV文件：
- `x_input.csv`: 输入数据 (N × 7)
- `y_output.csv`: 输出数据 (N × 26)

数据格式要求：
- 输入：7个浮点数特征
- 输出：26个浮点数目标值
- 5组数据样本数：7794/18957/18957/4539/4539

### 3. 启动可视化界面

```bash
python gradio_app.py
```

访问 `http://localhost:7860` 使用Web界面进行训练和预测。

### 4. 命令行训练

```python
from data_processor import DataProcessor
from trainer import Trainer

# 数据处理
processor = DataProcessor()
processed_groups = processor.process_all_groups()

# 模型训练
trainer = Trainer()
results = trainer.train_all_groups(processed_groups, model_types=['mlp', 'resnet'])
```

### 5. 模型推理

```python
from inference import load_all_models
import numpy as np

# 加载模型
evaluator = load_all_models()

# 单个预测
X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
prediction = evaluator.inference.predict('group_1_mlp', X, 'group_1')
print(f"预测结果: {prediction}")
```

## 🎛️ 配置说明

### 主要配置参数 (config.py)

```python
# 数据配置
INPUT_DIM = 7           # 输入维度
OUTPUT_DIM = 26         # 输出维度
DATA_GROUPS = 5         # 数据组数
GROUP_SIZES = [7794, 18957, 18957, 4539, 4539]  # 各组样本数

# 训练配置
BATCH_SIZE = 64         # 批次大小
LEARNING_RATE = 0.001   # 学习率
EPOCHS = 200            # 训练轮数
DROPOUT_RATE = 0.2      # Dropout率

# 模型配置
HIDDEN_DIMS = [128, 256, 512, 256, 128]  # MLP隐藏层维度
RESNET_BLOCKS = 4       # ResNet残差块数量
```

## 🧪 模型架构

### 1. MLP (多层感知机)
- **结构**: 5层全连接网络
- **特点**: 简单快速，适合基线测试
- **预期MSE**: 0.02-0.06

### 2. ResNet (残差网络)
- **结构**: 带残差连接的深度网络
- **特点**: 缓解梯度消失，训练稳定
- **预期MSE**: 0.015-0.04

### 3. Transformer
- **结构**: 基于注意力机制的网络
- **特点**: 捕获复杂关系，表达能力强
- **预期MSE**: 0.01-0.035

### 4. Ensemble (集成模型)
- **结构**: 多模型加权集成
- **特点**: 预测稳定，泛化能力强
- **预期MSE**: 0.008-0.025

## 📊 性能指标

### MSE目标值设定

| 性能等级 | MSE范围 | 应用场景 |
|---------|---------|----------|
| 优秀 | < 0.01 | 高精度工业应用 |
| 良好 | 0.01-0.05 | 大多数实际应用 |
| 可接受 | 0.05-0.1 | 一般精度要求 |
| 需改进 | > 0.1 | 需要优化 |

### 各组预期性能

| 数据组 | 样本数 | 预期MSE | 说明 |
|--------|--------|---------|------|
| Group 1 | 7794 | 0.02-0.05 | 样本适中 |
| Group 2 | 18957 | 0.01-0.03 | 样本最多 |
| Group 3 | 18957 | 0.01-0.03 | 样本最多 |
| Group 4 | 4539 | 0.03-0.08 | 样本较少 |
| Group 5 | 4539 | 0.03-0.08 | 样本较少 |

## 🔧 超参数调优策略

### 学习率调整
- **训练损失下降缓慢**: 增大学习率 (0.003-0.01)
- **训练损失震荡**: 减小学习率 (0.0003-0.0001)
- **过拟合**: 减小学习率 + 增大dropout

### 网络结构调整
- **MSE > 0.1**: 增加隐藏层数和维度
- **0.05 < MSE < 0.1**: 微调当前结构
- **MSE < 0.05**: 可以简化结构防止过拟合

### 正则化策略
- **过拟合**: 增大dropout (0.3-0.5) 和权重衰减
- **欠拟合**: 减小dropout (0.1-0.2) 和权重衰减

## 📈 使用界面

### Gradio Web界面功能

1. **数据加载**: 上传CSV文件或使用示例数据
2. **训练配置**: 调整超参数和选择模型
3. **模型训练**: 实时监控训练进度
4. **结果分析**: 查看训练曲线和性能指标
5. **模型预测**: 单个预测和批量预测

### 界面截图

启动后访问 `http://localhost:7860` 即可看到完整的Web界面。

## 📝 使用示例

### 完整训练流程

```python
# 1. 数据处理
from data_processor import DataProcessor
processor = DataProcessor()
processed_groups = processor.process_all_groups()

# 2. 模型训练
from trainer import Trainer
trainer = Trainer()
results = trainer.train_all_groups(
    processed_groups, 
    model_types=['mlp', 'resnet', 'transformer']
)

# 3. 模型评估
from inference import load_all_models
evaluator = load_all_models()

# 4. 生成报告
evaluator.generate_evaluation_report(results)
```

### 超参数分析

```python
from hyperparameter_analysis import HyperparameterAnalyzer

analyzer = HyperparameterAnalyzer()
analyzer.save_analysis_report()
```

## 🐛 常见问题

### Q1: 训练过程中出现CUDA内存不足
**A**: 减小batch_size或使用CPU训练
```python
config.BATCH_SIZE = 32
config.DEVICE = torch.device('cpu')
```

### Q2: 模型过拟合严重
**A**: 增加正则化和早停
```python
config.DROPOUT_RATE = 0.4
config.WEIGHT_DECAY = 1e-4
config.PATIENCE = 15
```

### Q3: 训练收敛缓慢
**A**: 调整学习率和优化器
```python
config.LEARNING_RATE = 0.003
# 或使用不同的优化器
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
```

### Q4: 预测精度不够
**A**: 尝试更复杂的模型或集成方法
```python
# 使用集成模型
model = ModelFactory.create_model('ensemble', input_dim, output_dim)
```

## 📚 技术栈

- **深度学习框架**: PyTorch 2.0+
- **数据处理**: NumPy, Pandas, Scikit-learn
- **可视化**: Matplotlib, Seaborn, Plotly
- **Web界面**: Gradio 4.0+
- **监控**: TensorBoard

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 项目Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 邮箱: your-email@example.com

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者和研究人员。

---

**注意**: 本项目仅用于学习和研究目的，请确保在实际应用中进行充分的测试和验证。