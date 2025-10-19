# 调试检查清单

## 快速问题诊断

### 🔍 问题分类
- [ ] **输入问题**: 图像文件、参数设置
- [ ] **处理问题**: 算法执行、性能问题  
- [ ] **输出问题**: 结果格式、可视化

---

## 📋 输入问题检查清单

### 图像文件检查
- [ ] 文件是否存在且可读
- [ ] 文件格式是否支持 (PNG, JPG, JPEG, BMP, TIFF)
- [ ] 文件是否损坏
- [ ] 图像尺寸是否合理 (不要过大或过小)
- [ ] 图像内容是否包含明显的多边形

### 参数设置检查
- [ ] `BINARY_THRESHOLD` 是否适合当前图像
- [ ] `MIN_CONTOUR_AREA` 和 `MAX_CONTOUR_AREA` 范围是否合理
- [ ] `EPSILON_FACTOR` 是否过大或过小
- [ ] 面积过滤参数 `min_area`, `max_area` 是否设置正确

### 快速验证命令
```bash
# 检查文件
ls -la your_image.png
file your_image.png

# 基础测试
python -c "import cv2; img=cv2.imread('your_image.png'); print('图像尺寸:', img.shape if img is not None else '加载失败')"
```

---

## ⚙️ 处理问题检查清单

### 多边形检测问题
- [ ] 启用调试模式查看中间结果
- [ ] 检查二值化效果是否理想
- [ ] 验证轮廓检测是否找到目标
- [ ] 确认多边形近似是否合理

### 相似度计算问题
- [ ] 检查多边形顶点数是否足够 (≥3)
- [ ] 验证标准化过程是否正常
- [ ] 确认相似度值是否在合理范围 [0,1]
- [ ] 检查是否有数值计算异常

### 性能问题
- [ ] 监控内存使用情况
- [ ] 检查处理时间是否过长
- [ ] 确认是否有无限循环或死锁
- [ ] 验证算法复杂度是否合理

### 调试命令
```bash
# 启用详细日志
python main.py your_image.png --debug

# 内存监控
python -c "
import psutil, os
process = psutil.Process(os.getpid())
print(f'内存使用: {process.memory_info().rss/1024/1024:.1f} MB')
"
```

---

## 📤 输出问题检查清单

### 结果验证
- [ ] 检查返回的坐标是否在图像范围内
- [ ] 验证唯一性评分是否合理 [0,1]
- [ ] 确认最唯一多边形是否符合预期
- [ ] 检查输出格式是否正确

### 可视化问题
- [ ] 确认可视化图像是否正确显示
- [ ] 检查标注是否清晰可见
- [ ] 验证颜色编码是否正确
- [ ] 确认保存路径是否正确

---

## 🛠️ 常见问题快速修复

### 问题1: 未检测到多边形
```python
# 解决方案
config.BINARY_THRESHOLD = 100  # 尝试不同阈值
config.MIN_CONTOUR_AREA = 50   # 降低最小面积
config.MAX_CONTOUR_AREA = 100000  # 增加最大面积

# 调试代码
detector = PolygonDetector(debug=True)
binary = detector.preprocess_image(image)
cv2.imshow('Binary Result', binary)
cv2.waitKey(0)
```

### 问题2: 相似度计算错误
```python
# 检查多边形有效性
for i, poly in enumerate(polygons):
    if len(poly['polygon']) < 3:
        print(f"警告: 多边形 {i} 顶点数不足: {len(poly['polygon'])}")
    
    if poly['area'] <= 0:
        print(f"警告: 多边形 {i} 面积异常: {poly['area']}")
```

### 问题3: 内存不足
```python
# 减少内存使用
import gc

# 处理大图像时分块处理
def process_large_image(image_path):
    # 缩放图像
    image = cv2.imread(image_path)
    if image.shape[0] > 2000 or image.shape[1] > 2000:
        scale = min(2000/image.shape[0], 2000/image.shape[1])
        new_size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
        image = cv2.resize(image, new_size)
    
    # 强制垃圾回收
    gc.collect()
    
    return image
```

### 问题4: 处理时间过长
```python
# 性能优化
from debug_utils import PerformanceProfiler

profiler = PerformanceProfiler()
profiler.start_timer("总处理时间")

# 在关键步骤添加计时
profiler.start_timer("图像预处理")
# ... 处理代码 ...
profiler.end_timer("图像预处理")

print(profiler.get_timing_report())
```

---

## 📊 调试工具使用

### 1. 启用详细日志
```python
import logging

# 设置日志级别
logging.getLogger('polygon_detector').setLevel(logging.DEBUG)
logging.getLogger('similarity_calculator').setLevel(logging.DEBUG)
logging.getLogger('uniqueness_analyzer').setLevel(logging.DEBUG)

# 或者全局设置
logging.basicConfig(level=logging.DEBUG)
```

### 2. 可视化调试
```python
from debug_utils import DebugVisualizer

visualizer = DebugVisualizer()

# 可视化预处理步骤
visualizer.visualize_preprocessing_steps(original, gray, binary)

# 可视化检测结果
visualizer.visualize_contours_and_polygons(image, contours, polygons)

# 可视化相似度矩阵
visualizer.visualize_similarity_matrix(similarity_matrix, polygon_ids)
```

### 3. 性能分析
```python
from debug_utils import PerformanceProfiler
import cProfile

# 方法1: 使用内置性能分析器
profiler = PerformanceProfiler()
# ... 使用profiler ...

# 方法2: 使用cProfile
cProfile.run('detector.process_image("image.png")', 'profile_stats')

# 分析结果
import pstats
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(10)
```

---

## 🔧 环境问题排查

### Python环境检查
```bash
# 检查Python版本
python --version

# 检查依赖包
pip list | grep -E "(opencv|numpy|scipy|matplotlib)"

# 检查包版本兼容性
pip check
```

### 系统资源检查
```bash
# 检查可用内存
free -h

# 检查磁盘空间
df -h

# 检查CPU使用率
top -p $(pgrep python)
```

### OpenCV安装问题
```bash
# 重新安装OpenCV
pip uninstall opencv-python
pip install opencv-python==4.8.1.78

# 测试OpenCV
python -c "import cv2; print('OpenCV版本:', cv2.__version__)"
```

---

## 📝 问题报告模板

当遇到无法解决的问题时，请按以下格式提供信息：

```
## 问题描述
简要描述遇到的问题

## 环境信息
- Python版本: 
- 操作系统: 
- 依赖包版本: 

## 重现步骤
1. 
2. 
3. 

## 预期结果
描述期望的结果

## 实际结果
描述实际发生的情况

## 错误信息
```
粘贴完整的错误信息和堆栈跟踪
```

## 已尝试的解决方案
列出已经尝试过的解决方法

## 附加信息
- 输入图像特征
- 配置参数
- 日志输出
```

---

## 🚀 性能优化建议

### 图像预处理优化
- 使用合适的图像尺寸 (建议 < 2000x2000)
- 选择最优的二值化阈值
- 考虑使用自适应阈值
- 添加形态学操作去噪

### 算法优化
- 缓存重复计算结果
- 使用向量化操作
- 考虑并行处理
- 优化内存使用

### 系统优化
- 增加系统内存
- 使用SSD存储
- 优化Python环境
- 考虑使用GPU加速

---

**记住**: 调试是一个系统性的过程，按照检查清单逐步排查，通常能快速定位问题所在。保持耐心，善用工具！ 🎯