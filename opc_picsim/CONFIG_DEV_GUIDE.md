# 多边形唯一性检测项目：配置调整与二次开发指南

本文档面向二次开发者，覆盖项目的配置项说明、相似度与唯一性算法的可调入口、模块扩展方法、性能与资源评估，以及建议的开发实践。结合当前代码结构与实现，帮助你快速精准地做定制和扩展。

---

## 1. 项目概览与运行

- 主要模块：
  - `polygon_detector.py`：多边形检测（预处理、轮廓、近似）
  - `similarity_calculator.py`：相似度（形状/尺寸/方向）与相似度矩阵
  - `uniqueness_analyzer.py`：唯一性评分（基于相似度矩阵）
  - `main.py`：主流程与 CLI
  - `debug_utils.py`：可视化与性能分析工具
  - `config.py`：全局配置
- 依赖安装：
  - `pip install -r requirements.txt`
- 基本运行：
  - `python main.py image.png --min-area 100 --max-area 5000`
  - 交互演示：`python demo.py`

---

## 2. 配置项总览（调整指南）

配置文件：`config.py`

- 图像与检测
  - `BINARY_THRESHOLD`（默认 127）：二值化阈值。图像偏暗/偏亮时调整（建议范围：80–180）。
  - `MIN_CONTOUR_AREA` / `MAX_CONTOUR_AREA`（100 / 50000）：过滤噪声与异常大区域（根据像素尺度调整）。
  - `EPSILON_FACTOR`（0.02）：多边形近似精度。过大导致过度简化，过小顶点过多（建议 0.01–0.05）。
- 相似度权重（综合相似度 = 线性加权）
  - `SHAPE_SIMILARITY_WEIGHT`（0.4）：轮廓形状重要性（Hausdorff + Hu）。
  - `SIZE_SIMILARITY_WEIGHT`（0.3）：面积比例的重要性。
  - `ORIENTATION_SIMILARITY_WEIGHT`（0.3）：主方向角度差的权重。
  - 建议保持权重总和≈1，便于直觉对比（代码不强制）。
- 唯一性与阈值
  - `SIMILARITY_THRESHOLD`（0.7）：统计“高相似对”时使用；可用于结果筛选/报警。
- 可视化与输出
  - `CONTOUR_COLOR` / `UNIQUE_COLOR` / `LINE_THICKNESS`：绘制参数。
  - `SAVE_INTERMEDIATE_RESULTS`（True）：保存中间与可视化结果。
  - `OUTPUT_DIR`（"output"）：输出目录。

进阶配置（可选，参考 `DEVELOPMENT_GUIDE.md`）
- 环境变量注入：将配置读自 `os.getenv`，适合部署环境差异化。
- 日志轮转：使用 `RotatingFileHandler` 控制日志大小与备份。

---

## 3. 相似度计算逻辑与权重调整

综合相似度由三部分组成：

- 形状相似度（`shape_similarity`）
  - 标准化：顶点平移到质心，按最大半径缩放到单位尺度（`normalize_polygon()`）。
  - Hausdorff 距离：取双向定向距离最大值 `d`，映射为 `1 / (1 + d)`。
  - Hu 矩：将标准化多边形栅格化到 500×500 掩码，计算 `cv2.HuMoments`，对数变换后欧氏距离再映射为 `1 / (1 + distance)`。
  - 组合：`shape_similarity = (hausdorff_sim + hu_sim) / 2`。
- 尺寸相似度（`size_similarity`）
  - 面积比例：`min(area1, area2) / max(area1, area2)`。
- 方向相似度（`orientation_similarity`）
  - PCA 主方向：对顶点协方差矩阵做特征分解取最大特征向量。
  - 角度映射：`angle_diff = arccos(|dot|)`，相似度 = `1 - angle_diff / (pi/2)`。

加权合成（`calculate_overall_similarity`）：
```
overall = w_shape * shape + w_size * size + w_orient * orient
```
权重调整建议：
- 场景 A（更关注轮廓形状）：提升 `w_shape` 到 0.6–0.7，降低 `w_size`。
- 场景 B（尺寸为关键约束）：提升 `w_size` 到 0.4–0.5，用于快速筛分。
- 场景 C（方向严格一致）：提升 `w_orient` 到 0.4+；若图案具旋转对称，降低到 0.1–0.2。
- 同步阈值：若整体相似度随权重显著升降，可调整 `SIMILARITY_THRESHOLD` 到 0.6–0.8。

动态权重与早停（建议扩展）：
```python
# similarity_calculator.py 中重写 calculate_overall_similarity：
def calculate_overall_similarity(self, p1, p2):
    shape = self.calculate_shape_similarity(p1['polygon'], p2['polygon'])
    size  = self.calculate_size_similarity(p1['area'], p2['area'])
    orient = self.calculate_orientation_similarity(p1['polygon'], p2['polygon'])

    # 早期筛分：尺寸或方向极低时直接返回较低相似度，避免昂贵形状计算
    if size < 0.3 or orient < 0.3:
        return 0.2 * shape + 0.5 * size + 0.3 * orient

    # 动态权重示例：近圆形（方向不稳定）降低方向权重
    w_shape, w_size, w_orient = 0.5, 0.2, 0.3
    return w_shape * shape + w_size * size + w_orient * orient
```

---

## 4. 唯一性评分（调整与替代）

- 现有公式（`uniqueness_analyzer.py`）：
  - 计算相似度矩阵后，针对第 `i` 个多边形取其与其它的最大相似度 `max` 与平均相似度 `avg`；
  - 唯一性评分：`score = 0.7 * (1 - max) + 0.3 * (1 - avg)`。
- 可调参数：将权重比调整为不同场景（例如更强调离群度可改为 0.9/0.1）。
- 替代策略（扩展建议）：
  - 聚类离群度：对相似度矩阵做聚类（DBSCAN/层次聚类），使用簇密度/距离作为唯一性。
  - 阈值计数：统计与 `SIMILARITY_THRESHOLD` 之上的邻居数量，邻居少者更唯一。

---

## 5. 模块二次开发指南

### 5.1 PolygonDetector（检测）
- 可扩展预处理：在 `preprocess_image()` 中加入自适应阈值、开闭运算去噪等。
- 调参路径：`BINARY_THRESHOLD`、面积过滤、`EPSILON_FACTOR`。
- 建议：在复杂背景与噪声环境中先做形态学处理，再近似轮廓。

### 5.2 SimilarityCalculator（相似度）
- 新特征：周长比、凸包比、圆形度、傅里叶描述子等，作为额外分量加入综合权重。
- 性能优化：
  - 缓存标准化结果与 Hu 矩特征（基于顶点列表的哈希）。
  - 降低掩码尺寸（500×500 → 256×256）。
  - 早停与剪枝（先用尺寸/方向筛分后再算形状）。

### 5.3 UniquenessAnalyzer（唯一性）
- 替换评分函数：实现聚类或基于邻居计数的评分，并保留当前 API 兼容性。
- 输出扩展：附加每个多边形的“相似邻居列表”和“离群度分数”。

### 5.4 主流程与 CLI（`main.py`）
- 当前 CLI：`--min-area`/`--max-area`/`--config`（`--config` 读取未实现）。
- 建议扩展：增加 `--debug` 与 `--visualize` 参数透出，完善 `--config` 读取。
```python
# 示例：在 main() 的 argparse 中添加
parser.add_argument('--debug', action='store_true', help='启用调试与性能报告')
parser.add_argument('--visualize', action='store_true', help='启用可视化输出')

# 读取 JSON 配置（若存在）：
if args.config and os.path.exists(args.config):
    with open(args.config, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cfg = Config()
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    config = cfg
else:
    config = Config()

# 传入检测器：
detector = PolygonUniquenessDetector(config, debug=args.debug, enable_visualization=args.visualize)
```

### 5.5 调试与可视化（`debug_utils.py`）
- 可视化：轮廓、相似度矩阵、唯一性评分柱状图。
- 性能分析：在关键步骤前后调用 `PerformanceProfiler.start_timer()` / `end_timer()`，生成时间报告。

---

## 6. 批处理与并行计算

- 并行相似度：对上三角配对列表使用 `multiprocessing.Pool` 并行映射。
- 注意事项：Windows 下需放入 `if __name__ == "__main__":` 保护；谨慎共享可视化与日志句柄。
- 大规模优化：
  - 顶点重采样为固定长度（如 64）以降低 Hausdorff 复杂度。
  - 先用尺寸/方向阈值剪枝，减少后续昂贵对比的数量。

---

## 7. 资源消耗评估与性能建议

- 时间复杂度：相似度矩阵为 `O(P^2)`；形状相似度（Hausdorff）近似 `O(V1*V2)`。
- 经验估算（单线程 CPU）：
  - `P=100, V≈50`：约 2–6 秒；`P=300`：约 20–60 秒。
- 内存：相似度矩阵 `P×P` 浮点（`P=1000` ≈ 8MB；`P=5000` ≈ 200MB）。Hu 掩码每次 ≈ 0.25MB（500×500）。
- 存储：每张图的结果 PNG/JSON 通常数百 KB–数 MB；批量时建议定期清理或压缩。
- 建议：尽量开启并行、降低掩码分辨率、启用剪枝与缓存；必要时按图块处理大图。

---

## 8. 质量保障与测试策略

- 单元测试：
  - 边界与异常：顶点 <3、面积=0、空输入、数值稳定性。
  - 不变性：平移/缩放/旋转后形状相似度应保持高；方向相似度对旋转对称图形稳定性。
- 基准测试：构造 `P=50/200/1000` 的数据集，记录时间与峰值内存，指导默认参数与并发设置。
- 日志与监控：在 CI 或批处理脚本中自动汇总时间与内存指标，滚动保存日志。

---

## 9. 常见改造场景与落地示例

- 工业件方向刚性：`w_orient=0.45, w_shape=0.35, w_size=0.20`；`SIMILARITY_THRESHOLD=0.75`。
- 纹样重复检测：`w_shape=0.6, w_size=0.2, w_orient=0.2`；Hu 掩码改为 256×256；启用剪枝（先看尺寸/方向）。
- 噪声多的影像：降低 `BINARY_THRESHOLD`、提升形态学滤波；`MIN_CONTOUR_AREA` 提高到 300–500。

---

## 10. 交付建议与下一步

- 先按目标场景设定权重模板与阈值，跑一组代表样例，观察相似度分布与唯一性评分。
- 如果处理时间过长：启用并行与剪枝；压测 `P` 与 `V` 的规模上限并记录资源曲线。
- 如果结果不稳定：
  - 检查顶点近似精度与预处理质量；
  - 对旋转对称/近圆形做方向权重自适应；
  - 优化 Hu 掩码分辨率与标准化策略。

如需，我可以提供：`--debug/--visualize` 参数透出、JSON 配置读取实现、动态权重与早停版本的 `SimilarityCalculator`、以及基准测试脚本示例。