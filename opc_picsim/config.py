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
    # 非联通体相似度的碎裂惩罚参数
    FRAGMENTATION_PENALTY_ALPHA = 0.5  # 组件数量的惩罚强度（越大惩罚越强）
    
    # 唯一性阈值
    SIMILARITY_THRESHOLD = 0.7  # 相似度阈值，超过此值认为相似
    
    # 可视化参数
    CONTOUR_COLOR = (0, 255, 0)  # 轮廓颜色 (BGR)
    UNIQUE_COLOR = (0, 0, 255)   # 唯一多边形颜色 (BGR)
    LINE_THICKNESS = 2           # 线条粗细
    
    # 输出参数
    SAVE_INTERMEDIATE_RESULTS = True  # 是否保存中间结果
    OUTPUT_DIR = "output"            # 输出目录

    # 区域扫描与匹配参数
    AREA_TOLERANCE_RATIO = 0.05       # 同面积匹配的相对容差（±比例）
    REGION_SCAN_STRIDE = 16           # 非联通体滑窗扫描步长（像素）
    REGION_MAX_CANDIDATES = 50        # 每个多边形的最大候选区域数（用于控制计算量）

    # 唯一性评分权重（基于区域相似）
    UNIQUENESS_WEIGHT_CONNECTED = 0.6     # 与同面积联通体的最大相似度的权重
    UNIQUENESS_WEIGHT_NONCONNECTED = 0.4  # 与同面积非联通区域的平均相似度的权重