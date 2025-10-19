# 多输入多输出回归优化项目

## 项目背景
本项目针对 8 维浮点输入到 70 维浮点输出的多输出回归问题，基于 1704 条样本数据（`x_input.csv` / `y_output.csv`）构建和优化了残差 MLP 神经网络。整个流程包括：数据标准化、模型训练、模型评估、推理脚本以及基于 Gradio 的可视化界面，方便进行快速实验与结果对比。

主要优化点：
- **数据管线**：仅在训练集上拟合标准化器，避免验证集泄露。
- **模型结构**：采用预归一化残差块、GELU 激活与可配置宽度/深度，提升表达能力和稳定性。
- **训练策略**：集成 AdamW、梯度裁剪、混合精度、ReduceLROnPlateau 调度与早停机制。
- **推理与可视化**：封装统一推理接口，并提供 Gradio 面板进行训练监控与在线预测。

## 环境依赖
- Python 3.9+
- PyTorch 1.11+（支持 CUDA 可加速训练）
- pandas、numpy、scikit-learn
- gradio

推荐使用 `pip` 安装：
```bash
pip install torch pandas numpy scikit-learn gradio
```

## 项目结构
```
├── dataset.py        # 数据加载与标准化
├── model.py          # 残差 MLP 模型
├── train.py          # 训练入口与配置
├── inference.py      # 推理脚本
├── utils.py          # 训练过程工具函数
├── app.py            # Gradio 可视化界面
├── x_input.csv       # 输入特征（8 维）
├── y_output.csv      # 输出标签（70 维）
└── checkpoints/      # 训练生成的模型、标准化器与日志（运行后自动创建）
```

## 数据准备
确保 `x_input.csv` 与 `y_output.csv` 位于项目根目录，文件不包含表头。若需使用自定义数据：
1. 保持输入列数为 8，输出列数为 70。
2. 使用 UTF-8 编码与逗号分隔。
3. 更新训练脚本参数 `--x-path`、`--y-path` 指向新的数据集。

## 训练流程
命令行训练：
```bash
python train.py \
  --epochs 500 \
  --batch-size 128 \
  --lr 3e-4 \
  --hidden 256 \
  --depth 6 \
  --checkpoint-dir checkpoints
```

训练过程中将输出：
- 归一化尺度下的 train/val MSE 与学习率
- 反标准化后的验证 MAE、RMSE（便于直观评估）
- 早停信息与最佳指标

训练完成后，`checkpoints/` 目录包含：
- `best.pt`：最优模型权重及结构参数
- `scalers.pt`：输入输出的 `StandardScaler`
- `training_summary.json`：训练配置、指标历史与最佳结果

## 推理使用
对新样本进行预测：
```bash
python inference.py --csv new_input.csv --checkpoint-dir checkpoints --out prediction.csv
```
支持直接传入 `numpy` 数组（见 `predict` 函数），默认会对预测结果反标准化并输出到 CSV。

## Gradio 可视化界面
运行命令：
```bash
python app.py
```
界面包含两个标签页：
1. **训练**：通过滑块调整超参数，实时查看训练日志与指标。
2. **推理**：上传 CSV 或在表格内手动输入样本，即可获取预测结果。

## 性能与优化建议
- 在当前数据规模下，归一化后的验证 MSE 目标区间约为 **0.18–0.24**。
- 若训练 MSE 远低于验证 MSE，考虑提高 `dropout`、`weight_decay` 或缩小网络容量，减轻过拟合。
- 若训练/验证同步停滞，可提升 `hidden`、`depth` 或降低学习率，加大模型表达能力或优化稳定性。
- 注意 `ReduceLROnPlateau` 的耐心值（`--scheduler-patience`），确保学习率有时间衰减。

## 二次开发指南
为了方便在项目基础上进行扩展，建议遵循以下步骤：

### 1. 新增或替换模型结构
- 在 `model.py` 中新增自定义网络类，需保证输入/输出维度与数据匹配。
- 通过修改 `train.py` 中的 `model_kwargs` 或引入新的配置参数，实现模型快速切换。
- 若使用不同归一化策略（例如 BatchNorm），请注意与小批量训练的兼容性。

### 2. 扩展数据处理逻辑
- 可在 `dataset.py` 中加入特征工程或数据增强，保持标准化流程只使用训练集。
- 若需要多文件或不同数据格式，扩展 `_load_arrays` 或封装新的数据读取函数。
- 对于在线部署场景，可增加数据缓存或批处理策略。

### 3. 自定义训练流程
- `TrainConfig` 与 `run` 函数可轻松接入自定义调度器、损失函数或评价指标。
- 需要额外日志或可视化时，可在 `history` 中添加更多字段，或结合 TensorBoard/Weights & Biases。
- 若要进行超参数搜索，可在外部脚本中循环调用 `run(config)` 并记录 `TrainSummary`。

### 4. 推理与部署
- 在 `inference.py` 中可以加入批量预测、模型集成或置信区间估计逻辑。
- 若部署到 Web 服务或微服务框架，直接复用 `predict` 函数并管理好 `checkpoint_dir`。
- Gradio 界面可扩展上传模型、下载结果等功能，满足更复杂的业务需求。

### 5. 代码规范与协作
- 保持模块化设计，新增功能时优先编写单元测试或脚本验证。
- 使用 `training_summary.json` 记录实验配置，便于多次迭代与结果复现。
- 若团队协作，建议增加 `requirements.txt`、`Makefile` 或 `pre-commit` 钩子以统一环境与风格。

## 常见问题
| 问题 | 说明与处理 |
| --- | --- |
| 训练报错找不到 CUDA | 安装对应版本 PyTorch，或使用 `--no-amp` 强制禁用混合精度。 |
| 预测结果全为 NaN/inf | 检查输入是否包含非法值，必要时在 `_table_to_array` 中添加清洗逻辑。 |
| Gradio 打开缓慢 | 首次安装或加载模型耗时较长，可提前预热或使用持久进程部署。 |

## 许可
本项目无附加许可证限制，可根据自身需求二次开发与部署。欢迎在此基础上继续优化模型与业务流程。祝实验顺利！
