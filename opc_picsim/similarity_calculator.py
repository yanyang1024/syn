import cv2
import numpy as np
import logging
from typing import List, Tuple
from scipy.spatial.distance import directed_hausdorff
from config import Config

# 配置日志
logger = logging.getLogger(__name__)

class SimilarityCalculator:
    """多边形相似度计算器"""
    
    def __init__(self, config: Config = None, debug: bool = False):
        self.config = config or Config()
        self.debug = debug
        if self.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("SimilarityCalculator初始化完成，调试模式已启用")
    
    def normalize_polygon(self, polygon: np.ndarray) -> np.ndarray:
        """标准化多边形：移动到原点并缩放"""
        # 转换为浮点数
        polygon = polygon.astype(np.float32).reshape(-1, 2)
        
        # 计算质心
        centroid = np.mean(polygon, axis=0)
        
        # 移动到原点
        normalized = polygon - centroid
        
        # 计算到质心的最大距离进行缩放
        max_dist = np.max(np.linalg.norm(normalized, axis=1))
        if max_dist > 0:
            normalized = normalized / max_dist
        
        return normalized
    
    def calculate_shape_similarity(self, poly1: np.ndarray, poly2: np.ndarray) -> float:
        """计算形状相似度（基于Hausdorff距离）"""
        try:
            if self.debug:
                logger.debug(f"计算形状相似度: poly1形状={poly1.shape}, poly2形状={poly2.shape}")
            
            # 检查输入有效性
            if len(poly1) < 3 or len(poly2) < 3:
                logger.warning("多边形顶点数少于3，返回0相似度")
                return 0.0
            
            # 标准化多边形
            norm_poly1 = self.normalize_polygon(poly1)
            norm_poly2 = self.normalize_polygon(poly2)
            
            if self.debug:
                logger.debug(f"标准化后: poly1范围=[{norm_poly1.min():.3f}, {norm_poly1.max():.3f}], "
                           f"poly2范围=[{norm_poly2.min():.3f}, {norm_poly2.max():.3f}]")
            
            # 计算双向Hausdorff距离
            dist1 = directed_hausdorff(norm_poly1, norm_poly2)[0]
            dist2 = directed_hausdorff(norm_poly2, norm_poly1)[0]
            
            # 取最大值作为Hausdorff距离
            hausdorff_dist = max(dist1, dist2)
            
            # 转换为相似度（0-1，1表示完全相似）
            similarity = 1.0 / (1.0 + hausdorff_dist)
            
            if self.debug:
                logger.debug(f"Hausdorff距离: {hausdorff_dist:.4f}, 形状相似度: {similarity:.4f}")
            
            return similarity
            
        except Exception as e:
            logger.error(f"计算形状相似度时发生错误: {str(e)}")
            return 0.0
    
    def calculate_size_similarity(self, area1: float, area2: float) -> float:
        """计算尺寸相似度"""
        if area1 == 0 or area2 == 0:
            return 0.0
        
        # 计算面积比例
        ratio = min(area1, area2) / max(area1, area2)
        return ratio
    
    def calculate_orientation_similarity(self, poly1: np.ndarray, poly2: np.ndarray) -> float:
        """计算方向相似度（考虑旋转不变性）"""
        # 计算主方向
        def get_principal_direction(polygon):
            points = polygon.reshape(-1, 2).astype(np.float32)
            # 计算协方差矩阵
            cov_matrix = np.cov(points.T)
            # 计算特征值和特征向量
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            # 主方向是最大特征值对应的特征向量
            principal_direction = eigenvectors[:, np.argmax(eigenvalues)]
            return principal_direction
        
        try:
            dir1 = get_principal_direction(poly1)
            dir2 = get_principal_direction(poly2)
            
            # 计算角度差异
            dot_product = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
            angle_diff = np.arccos(np.abs(dot_product))  # 使用绝对值考虑方向不变性
            
            # 转换为相似度
            similarity = 1.0 - (angle_diff / (np.pi / 2))
            return max(0.0, similarity)
        
        except:
            # 如果计算失败，返回中等相似度
            return 0.5
    
    def calculate_hu_moments_similarity(self, poly1: np.ndarray, poly2: np.ndarray) -> float:
        """使用Hu矩计算形状相似度"""
        def get_hu_moments(polygon):
            # 创建掩码图像
            mask = np.zeros((500, 500), dtype=np.uint8)
            # 将多边形坐标缩放到图像范围内
            scaled_poly = ((polygon.reshape(-1, 2) + 1) * 200 + 50).astype(np.int32)
            cv2.fillPoly(mask, [scaled_poly], 255)
            
            # 计算图像矩
            moments = cv2.moments(mask)
            # 计算Hu矩
            hu_moments = cv2.HuMoments(moments).flatten()
            # 对数变换
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            return hu_moments
        
        try:
            hu1 = get_hu_moments(self.normalize_polygon(poly1))
            hu2 = get_hu_moments(self.normalize_polygon(poly2))
            
            # 计算欧氏距离
            distance = np.linalg.norm(hu1 - hu2)
            
            # 转换为相似度
            similarity = 1.0 / (1.0 + distance)
            return similarity
        
        except:
            return 0.0
    
    def calculate_overall_similarity(self, poly1_info: dict, poly2_info: dict) -> float:
        """计算综合相似度"""
        # 形状相似度（结合Hausdorff距离和Hu矩）
        shape_sim1 = self.calculate_shape_similarity(poly1_info['polygon'], poly2_info['polygon'])
        shape_sim2 = self.calculate_hu_moments_similarity(poly1_info['polygon'], poly2_info['polygon'])
        shape_similarity = (shape_sim1 + shape_sim2) / 2
        
        # 尺寸相似度
        size_similarity = self.calculate_size_similarity(poly1_info['area'], poly2_info['area'])
        
        # 方向相似度
        orientation_similarity = self.calculate_orientation_similarity(
            poly1_info['polygon'], poly2_info['polygon']
        )
        
        # 加权综合相似度
        overall_similarity = (
            self.config.SHAPE_SIMILARITY_WEIGHT * shape_similarity +
            self.config.SIZE_SIMILARITY_WEIGHT * size_similarity +
            self.config.ORIENTATION_SIMILARITY_WEIGHT * orientation_similarity
        )
        
        return overall_similarity
    
    def calculate_similarity_matrix(self, polygons: List[dict]) -> np.ndarray:
        """计算所有多边形之间的相似度矩阵"""
        n = len(polygons)
        logger.info(f"开始计算 {n}x{n} 相似度矩阵")
        
        if n == 0:
            logger.warning("输入多边形列表为空")
            return np.array([])
        
        if n == 1:
            logger.info("只有一个多边形，返回单元素矩阵")
            return np.array([[1.0]])
        
        similarity_matrix = np.zeros((n, n))
        total_comparisons = n * (n - 1) // 2
        completed_comparisons = 0
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0  # 自己与自己完全相似
                elif i < j:
                    # 只计算上三角矩阵，然后对称填充
                    try:
                        similarity = self.calculate_overall_similarity(polygons[i], polygons[j])
                        similarity_matrix[i][j] = similarity
                        similarity_matrix[j][i] = similarity
                        
                        completed_comparisons += 1
                        
                        if self.debug:
                            logger.debug(f"多边形 {i} vs {j}: 相似度 = {similarity:.4f}")
                        
                        # 进度报告
                        if completed_comparisons % max(1, total_comparisons // 10) == 0:
                            progress = (completed_comparisons / total_comparisons) * 100
                            logger.info(f"相似度计算进度: {progress:.1f}% ({completed_comparisons}/{total_comparisons})")
                    
                    except Exception as e:
                        logger.error(f"计算多边形 {i} 和 {j} 的相似度时发生错误: {str(e)}")
                        similarity_matrix[i][j] = 0.0
                        similarity_matrix[j][i] = 0.0
        
        # 统计相似度矩阵
        non_diagonal = similarity_matrix[np.triu_indices(n, k=1)]
        if len(non_diagonal) > 0:
            logger.info(f"相似度统计 - 最小: {non_diagonal.min():.4f}, "
                       f"最大: {non_diagonal.max():.4f}, "
                       f"平均: {non_diagonal.mean():.4f}")
            
            # 统计高相似度对
            high_similarity_threshold = 0.7
            high_similarity_count = np.sum(non_diagonal > high_similarity_threshold)
            logger.info(f"高相似度对数 (>{high_similarity_threshold}): {high_similarity_count}")
        
        logger.info("相似度矩阵计算完成")
        return similarity_matrix