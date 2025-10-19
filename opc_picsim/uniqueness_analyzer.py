import numpy as np
import logging
from typing import List, Tuple, Dict
from config import Config
from similarity_calculator import SimilarityCalculator

# 配置日志
logger = logging.getLogger(__name__)

class UniquenessAnalyzer:
    """多边形唯一性分析器"""
    
    def __init__(self, config: Config = None, debug: bool = False):
        self.config = config or Config()
        self.debug = debug
        self.similarity_calculator = SimilarityCalculator(config, debug)
        if self.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("UniquenessAnalyzer初始化完成，调试模式已启用")
    
    def filter_by_size(self, polygons: List[dict], min_area: float = None, max_area: float = None) -> List[dict]:
        """根据区域大小过滤多边形"""
        if min_area is None:
            min_area = self.config.MIN_CONTOUR_AREA
        if max_area is None:
            max_area = self.config.MAX_CONTOUR_AREA
        
        filtered_polygons = []
        for polygon_info in polygons:
            area = polygon_info['area']
            if min_area <= area <= max_area:
                filtered_polygons.append(polygon_info)
        
        return filtered_polygons
    
    def calculate_uniqueness_scores(self, polygons: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        """计算每个多边形的唯一性评分"""
        if len(polygons) == 0:
            return np.array([]), np.array([])
        
        if len(polygons) == 1:
            return np.array([1.0]), np.array([[1.0]])
        
        # 计算相似度矩阵
        similarity_matrix = self.similarity_calculator.calculate_similarity_matrix(polygons)
        
        # 计算唯一性评分
        uniqueness_scores = np.zeros(len(polygons))
        
        for i in range(len(polygons)):
            # 获取与其他多边形的相似度
            similarities_with_others = similarity_matrix[i, :]
            
            # 排除自己（相似度为1.0）
            similarities_with_others = similarities_with_others[similarities_with_others < 1.0]
            
            if len(similarities_with_others) == 0:
                # 如果只有一个多边形，唯一性最高
                uniqueness_scores[i] = 1.0
            else:
                # 计算唯一性评分：1 - 最大相似度
                max_similarity = np.max(similarities_with_others)
                uniqueness_scores[i] = 1.0 - max_similarity
                
                # 也可以考虑平均相似度的影响
                avg_similarity = np.mean(similarities_with_others)
                uniqueness_scores[i] = 0.7 * (1.0 - max_similarity) + 0.3 * (1.0 - avg_similarity)
        
        return uniqueness_scores, similarity_matrix
    
    def find_most_unique_polygon(self, polygons: List[dict]) -> Tuple[int, dict]:
        """找到最唯一的多边形"""
        if len(polygons) == 0:
            return -1, None
        
        if len(polygons) == 1:
            return 0, polygons[0]
        
        # 计算唯一性评分
        uniqueness_scores, _ = self.calculate_uniqueness_scores(polygons)
        
        # 找到评分最高的多边形
        most_unique_idx = np.argmax(uniqueness_scores)
        most_unique_polygon = polygons[most_unique_idx]
        
        return most_unique_idx, most_unique_polygon
    
    def get_uniqueness_report(self, polygons: List[dict]) -> Dict:
        """生成唯一性分析报告"""
        if len(polygons) == 0:
            return {
                'total_polygons': 0,
                'uniqueness_scores': [],
                'similarity_matrix': np.array([]),
                'most_unique_index': -1,
                'most_unique_score': 0.0,
                'analysis_summary': "未检测到多边形"
            }
        
        # 计算唯一性评分和相似度矩阵
        uniqueness_scores, similarity_matrix = self.calculate_uniqueness_scores(polygons)
        
        # 找到最唯一的多边形
        most_unique_idx = np.argmax(uniqueness_scores)
        most_unique_score = uniqueness_scores[most_unique_idx]
        
        # 统计相似多边形数量
        similar_pairs = []
        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                if similarity_matrix[i][j] > self.config.SIMILARITY_THRESHOLD:
                    similar_pairs.append((i, j, similarity_matrix[i][j]))
        
        # 生成分析摘要
        summary = f"检测到 {len(polygons)} 个多边形。"
        if len(similar_pairs) > 0:
            summary += f" 发现 {len(similar_pairs)} 对相似多边形（相似度 > {self.config.SIMILARITY_THRESHOLD}）。"
        else:
            summary += " 未发现高度相似的多边形对。"
        
        summary += f" 最唯一多边形ID: {most_unique_idx}，唯一性评分: {most_unique_score:.3f}。"
        
        report = {
            'total_polygons': len(polygons),
            'uniqueness_scores': uniqueness_scores.tolist(),
            'similarity_matrix': similarity_matrix.tolist(),
            'most_unique_index': int(most_unique_idx),
            'most_unique_score': float(most_unique_score),
            'similar_pairs': similar_pairs,
            'analysis_summary': summary,
            'most_unique_polygon_info': {
                'id': polygons[most_unique_idx]['id'],
                'bbox': polygons[most_unique_idx]['bbox'],
                'area': polygons[most_unique_idx]['area'],
                'vertices': polygons[most_unique_idx]['vertices']
            }
        }
        
        return report
    
    def get_bounding_box_coordinates(self, polygon_info: dict) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """获取多边形边界矩形的两点坐标"""
        x, y, w, h = polygon_info['bbox']
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        return top_left, bottom_right