# 多边形唯一性检测配置文件

class Config:
    # 图像处理参数
    BINARY_THRESHOLD = 127  # 二值化阈值
    MIN_CONTOUR_AREA = 100  # 最小轮廓面积
    MAX_CONTOUR_AREA = 50000  # 最大轮廓面积
    
    # 多边形近似参数
    EPSILON_FACTOR = 0.02  # 多边形近似精度因子
    
    # 相似度计算参数
    SHAPE_SIMILARITY_WEIGHT = 0.4  # 形状相似度权重
    SIZE_SIMILARITY_WEIGHT = 0.3   # 尺寸相似度权重
    ORIENTATION_SIMILARITY_WEIGHT = 0.3  # 方向相似度权重
    
    # 唯一性阈值
    SIMILARITY_THRESHOLD = 0.7  # 相似度阈值，超过此值认为相似
    
    # 可视化参数
    CONTOUR_COLOR = (0, 255, 0)  # 轮廓颜色 (BGR)
    UNIQUE_COLOR = (0, 0, 255)   # 唯一多边形颜色 (BGR)
    LINE_THICKNESS = 2           # 线条粗细
    
    # 输出参数
    SAVE_INTERMEDIATE_RESULTS = True  # 是否保存中间结果
    OUTPUT_DIR = "output"            # 输出目录