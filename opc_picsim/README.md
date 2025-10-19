# 多边形唯一性检测工具

## 项目简介

这是一个基于深度学习和计算机视觉的多边形唯一性检测工具。该工具能够从PNG图像中检测多边形，分析它们的相似度，并找出最具唯一性的多边形区域。

## 功能特性

- **多边形检测**: 自动检测图像中的所有多边形
- **区域过滤**: 根据面积大小过滤多边形
- **相似度计算**: 多维度计算多边形相似度（形状、尺寸、方向）
- **唯一性分析**: 评估每个多边形的唯一性程度
- **可视化输出**: 生成带标注的结果图像
- **详细报告**: 输出JSON格式的分析报告

## 算法原理

### 1. 多边形检测
- 图像预处理：灰度化、高斯模糊、二值化
- 轮廓检测：使用OpenCV的轮廓检测算法
- 多边形近似：将轮廓近似为多边形

### 2. 相似度计算
采用多维度相似度计算方法：

- **形状相似度** (权重40%):
  - Hausdorff距离：衡量两个多边形形状的相似程度
  - Hu矩：旋转、缩放、平移不变的形状描述符

- **尺寸相似度** (权重30%):
  - 基于面积比例的相似度计算

- **方向相似度** (权重30%):
  - 基于主方向角度差异的相似度计算

### 3. 唯一性评分
- 计算每个多边形与其他所有多边形的相似度
- 唯一性评分 = 1 - 最大相似度
- 考虑平均相似度的影响进行加权

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 命令行使用

```bash
# 基本使用
python main.py image.png

# 设置面积过滤
python main.py image.png --min-area 100 --max-area 5000

# 使用自定义配置
python main.py image.png --config config.json
```

### 交互式演示

```bash
python demo.py
```

### 编程接口

```python
from main import PolygonUniquenessDetector
from config import Config

# 创建检测器
detector = PolygonUniquenessDetector()

# 处理图像
result = detector.process_image("image.png")

if result['success']:
    coords = result['result']
    print(f"最唯一多边形坐标: {coords['top_left']} -> {coords['bottom_right']}")
    print(f"唯一性评分: {coords['uniqueness_score']}")
```

## 输出格式

### 标准输出
```
最唯一多边形边界矩形坐标:
左上角: (x1, y1)
右下角: (x2, y2)
唯一性评分: 0.xxx

输出结果: (x1, y1), (x2, y2)
```

### 可视化文件
- `{image_name}_result.png`: 带标注的结果图像
- `{image_name}_report.json`: 详细分析报告

## 配置参数

可以通过修改 `config.py` 调整以下参数：

```python
# 图像处理参数
BINARY_THRESHOLD = 127      # 二值化阈值
MIN_CONTOUR_AREA = 100      # 最小轮廓面积
MAX_CONTOUR_AREA = 50000    # 最大轮廓面积

# 相似度计算权重
SHAPE_SIMILARITY_WEIGHT = 0.4       # 形状相似度权重
SIZE_SIMILARITY_WEIGHT = 0.3        # 尺寸相似度权重
ORIENTATION_SIMILARITY_WEIGHT = 0.3  # 方向相似度权重

# 唯一性阈值
SIMILARITY_THRESHOLD = 0.7  # 相似度阈值
```

## 项目结构

```
opc_picsim/
├── main.py                 # 主程序入口
├── demo.py                 # 交互式演示程序
├── config.py               # 配置文件
├── polygon_detector.py     # 多边形检测模块
├── similarity_calculator.py # 相似度计算模块
├── uniqueness_analyzer.py  # 唯一性分析模块
├── requirements.txt        # 依赖包列表
├── README.md              # 项目说明文档
└── output/                # 输出目录
    ├── *_result.png       # 可视化结果
    └── *_report.json      # 分析报告
```

## 技术栈

- **Python 3.7+**
- **OpenCV**: 图像处理和计算机视觉
- **NumPy**: 数值计算
- **SciPy**: 科学计算（Hausdorff距离）
- **Shapely**: 几何计算
- **Matplotlib**: 可视化
- **Pillow**: 图像处理

## 应用场景

- 工业质检中的零件唯一性识别
- 图案设计中的重复元素检测
- 生物学研究中的细胞形态分析
- 建筑设计中的结构元素分析

## 注意事项

1. 输入图像应为PNG格式
2. 图像中的多边形应有明显的边界
3. 建议图像背景为单色以提高检测精度
4. 可根据具体应用场景调整配置参数

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。