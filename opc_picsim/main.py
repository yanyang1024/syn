#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多边形唯一性检测主程序
输入：PNG图像
输出：最唯一多边形的边界矩形坐标
"""

import os
import sys
import cv2
import json
import logging
import argparse
from typing import Tuple

import numpy as np

from config import Config
from polygon_detector import PolygonDetector
from similarity_calculator import SimilarityCalculator
from uniqueness_analyzer import UniquenessAnalyzer
from debug_utils import DebugVisualizer, PerformanceProfiler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('polygon_detection.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class PolygonUniquenessDetector:
    """多边形唯一性检测主类"""
    
    def __init__(self, config: Config = None, debug: bool = False, enable_visualization: bool = False):
        self.config = config or Config()
        self.debug = debug
        self.enable_visualization = enable_visualization
        
        # 初始化组件
        self.detector = PolygonDetector(self.config, debug)
        self.analyzer = UniquenessAnalyzer(self.config, debug)
        
        # 初始化调试工具
        if self.debug or self.enable_visualization:
            self.visualizer = DebugVisualizer()
            self.profiler = PerformanceProfiler()
        
        # 创建输出目录
        if not os.path.exists(self.config.OUTPUT_DIR):
            os.makedirs(self.config.OUTPUT_DIR)
            logger.info(f"创建输出目录: {self.config.OUTPUT_DIR}")

    def _compute_integral_image(self, binary: np.ndarray) -> np.ndarray:
        """计算二值图的积分图（用于快速滑窗面积统计）"""
        mask = (binary > 0).astype(np.uint8)
        integral = cv2.integral(mask)
        return integral

    def _rect_sum(self, integral: np.ndarray, x: int, y: int, w: int, h: int) -> int:
        """使用积分图计算矩形区域的前景像素数"""
        # 注意：cv2.integral 的尺寸为 (H+1, W+1)，索引需偏移 +1
        x2, y2 = x + w, y + h
        return int(
            integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x]
        )

    def _generate_region_candidates(self, binary: np.ndarray, polygons: list, polygon_info: dict):
        """为指定多边形生成同面积区域候选（联通/非联通）"""
        target_area = float(polygon_info['area'])
        tol = self.config.AREA_TOLERANCE_RATIO
        min_area = target_area * (1.0 - tol)
        max_area = target_area * (1.0 + tol)

        connected_candidates = []
        non_connected_candidates = []

        # 1) 联通体候选：利用其他已检测多边形的面积接近者
        for other in polygons:
            if other['id'] == polygon_info['id']:
                continue
            a = float(other['area'])
            if min_area <= a <= max_area:
                connected_candidates.append({
                    'is_connected': True,
                    'polygon': other['polygon'],
                    'area': a
                })

        # 2) 非联通候选：在二值图上按滑窗采样，选择同面积（前景像素数）近似的patch
        h_img, w_img = binary.shape[:2]
        integral = self._compute_integral_image(binary)

        # 两组窗口尺寸：使用目标bbox尺寸与平方近似尺寸
        bx, by, bw, bh = polygon_info['bbox']
        # 保证正值
        bw = max(1, int(bw))
        bh = max(1, int(bh))
        side = int(max(1, np.sqrt(target_area)))
        window_sizes = [(bw, bh), (side, side)]

        stride = max(1, int(self.config.REGION_SCAN_STRIDE))
        candidate_records = []  # (diff, x, y, w, h, area)
        for (ww, hh) in window_sizes:
            if ww <= 0 or hh <= 0:
                continue
            if ww > w_img or hh > h_img:
                continue
            for y in range(0, h_img - hh + 1, stride):
                for x in range(0, w_img - ww + 1, stride):
                    area_sum = self._rect_sum(integral, x, y, ww, hh)
                    if min_area <= area_sum <= max_area:
                        diff = abs(area_sum - target_area)
                        candidate_records.append((diff, x, y, ww, hh, area_sum))

        # 选取前 N 个最接近面积的候选
        candidate_records.sort(key=lambda t: t[0])
        max_cand = int(self.config.REGION_MAX_CANDIDATES)
        candidate_records = candidate_records[:max_cand]

        for _, x, y, ww, hh, area_sum in candidate_records:
            patch = binary[y:y+hh, x:x+ww]
            # 计算连通组件数量（排除背景）
            num_labels, labels = cv2.connectedComponents((patch > 0).astype(np.uint8), connectivity=8)
            num_components = max(0, num_labels - 1)
            if num_components <= 1:
                # 联通patch不在此列表中，避免与已检测多边形重复；如需也可添加到connected候选
                continue
            non_connected_candidates.append({
                'is_connected': False,
                'mask': patch,
                'num_components': int(num_components),
                'area': float(area_sum),
                'bbox': (x, y, ww, hh)
            })

        return connected_candidates, non_connected_candidates

    
    def process_image(self, image_path: str, min_area: float = None, max_area: float = None) -> dict:
        """处理图像并返回结果"""
        logger.info(f"开始处理图像: {image_path}")
        
        if self.debug:
            self.profiler.start_timer("总处理时间")
        
        try:
            # 验证输入
            if not self._validate_input(image_path):
                return self._create_error_result("输入验证失败")
            
            # 检测多边形
            if self.debug:
                self.profiler.start_timer("多边形检测")
            
            logger.info("1. 检测多边形...")
            polygons, original_image = self.detector.detect_polygons(image_path)
            
            if self.debug:
                self.profiler.end_timer("多边形检测")
            
            logger.info(f"   检测到 {len(polygons)} 个多边形")
            
            if len(polygons) == 0:
                return self._create_error_result('未检测到任何多边形', {
                    'suggestion': '尝试调整二值化阈值或轮廓面积范围',
                    'current_threshold': self.config.BINARY_THRESHOLD,
                    'current_area_range': [self.config.MIN_CONTOUR_AREA, self.config.MAX_CONTOUR_AREA]
                })
            
            # 根据面积过滤多边形
            if min_area is not None or max_area is not None:
                logger.info("2. 过滤多边形...")
                filtered_polygons = self.analyzer.filter_by_size(polygons, min_area, max_area)
                logger.info(f"   过滤后剩余 {len(filtered_polygons)} 个多边形")
            else:
                filtered_polygons = polygons
            
            if len(filtered_polygons) == 0:
                return self._create_error_result('过滤后未剩余任何多边形', {
                    'suggestion': '调整面积过滤范围',
                    'detected_areas': [p['area'] for p in polygons],
                    'filter_range': [min_area, max_area]
                })
            
            # 分析唯一性
            if self.debug:
                self.profiler.start_timer("唯一性分析")
            
            logger.info("3. 分析唯一性...")
            # 新逻辑：基于“多边形 vs 同面积区域（联通/非联通）”的相似
            # 1) 准备二值图供滑窗与连通性分析
            binary_image = self.detector.preprocess_image(original_image)

            # 2) 计算每个多边形与同面积候选区域的相似向量
            sim_calculator = SimilarityCalculator(self.config, self.debug)
            per_polygon_connected_sims = []
            per_polygon_nonconnected_sims = []
            uniqueness_scores = []

            for poly in filtered_polygons:
                connected_cands, non_connected_cands = self._generate_region_candidates(
                    binary_image, filtered_polygons, poly
                )

                # 计算相似度
                conn_sims = [sim_calculator.calculate_region_similarity(poly, rc) for rc in connected_cands]
                nonconn_sims = [sim_calculator.calculate_region_similarity(poly, rc) for rc in non_connected_cands]
                per_polygon_connected_sims.append(conn_sims)
                per_polygon_nonconnected_sims.append(nonconn_sims)

                # 唯一性评分：区分联通/非联通影响
                if len(conn_sims) > 0:
                    max_connected = float(np.max(conn_sims))
                else:
                    max_connected = 0.0
                if len(nonconn_sims) > 0:
                    avg_nonconnected = float(np.mean(nonconn_sims))
                else:
                    avg_nonconnected = 0.0

                score = (
                    self.config.UNIQUENESS_WEIGHT_CONNECTED * (1.0 - max_connected) +
                    self.config.UNIQUENESS_WEIGHT_NONCONNECTED * (1.0 - avg_nonconnected)
                )
                uniqueness_scores.append(score)

            uniqueness_scores = np.array(uniqueness_scores, dtype=np.float32)
            most_unique_idx = int(np.argmax(uniqueness_scores))

            # 3) 组装报告
            uniqueness_report = {
                'total_polygons': len(filtered_polygons),
                'uniqueness_scores': uniqueness_scores.tolist(),
                'most_unique_index': most_unique_idx,
                'most_unique_score': float(uniqueness_scores[most_unique_idx]) if len(uniqueness_scores) > 0 else 0.0,
                'analysis_summary': (
                    f"采用同面积区域相似度（联通/非联通）计算。最唯一ID: {most_unique_idx}, "
                    f"评分: {float(uniqueness_scores[most_unique_idx]) if len(uniqueness_scores)>0 else 0.0:.3f}"
                ),
                'connected_similarity_vectors': per_polygon_connected_sims,
                'nonconnected_similarity_vectors': per_polygon_nonconnected_sims,
            }
            
            if self.debug:
                self.profiler.end_timer("唯一性分析")
            
            # 获取最唯一多边形的坐标
            most_unique_idx = uniqueness_report['most_unique_index']
            most_unique_polygon = filtered_polygons[most_unique_idx]
            top_left, bottom_right = self.analyzer.get_bounding_box_coordinates(most_unique_polygon)
            
            logger.info(f"   最唯一多边形ID: {most_unique_idx}")
            logger.info(f"   唯一性评分: {uniqueness_report['most_unique_score']:.3f}")
            logger.info(f"   边界矩形坐标: {top_left} -> {bottom_right}")
            
            # 可视化和保存结果
            if self.config.SAVE_INTERMEDIATE_RESULTS or self.enable_visualization:
                logger.info("4. 保存可视化结果...")
                self._save_results(original_image, filtered_polygons, most_unique_idx, 
                                 uniqueness_report, image_path)
            
            # 性能报告
            if self.debug:
                self.profiler.end_timer("总处理时间")
                logger.info("\n" + self.profiler.get_timing_report())
            
            return self._create_success_result(top_left, bottom_right, uniqueness_report, most_unique_polygon)
            
        except Exception as e:
            logger.error(f"处理图像时发生未预期的错误: {str(e)}", exc_info=True)
            return self._create_error_result(f"处理失败: {str(e)}", {'exception_type': type(e).__name__})
    
    def _validate_input(self, image_path: str) -> bool:
        """验证输入参数"""
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                logger.error(f"图像文件不存在: {image_path}")
                return False
            
            # 检查文件大小
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                logger.error("图像文件为空")
                return False
            
            if file_size > 100 * 1024 * 1024:  # 100MB
                logger.warning(f"图像文件较大 ({file_size / 1024 / 1024:.1f}MB)，处理可能较慢")
            
            # 检查文件权限
            if not os.access(image_path, os.R_OK):
                logger.error("无法读取图像文件，请检查文件权限")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"输入验证时发生错误: {str(e)}")
            return False
    
    def _create_success_result(self, top_left: Tuple[int, int], bottom_right: Tuple[int, int], 
                             uniqueness_report: dict, polygon_info: dict) -> dict:
        """创建成功结果"""
        return {
            'success': True,
            'message': '处理成功',
            'result': {
                'top_left': top_left,
                'bottom_right': bottom_right,
                'uniqueness_score': uniqueness_report['most_unique_score'],
                'polygon_info': polygon_info,
                'full_report': uniqueness_report
            }
        }
    
    def _create_error_result(self, message: str, details: dict = None) -> dict:
        """创建错误结果"""
        result = {
            'success': False,
            'message': message,
            'result': None
        }
        if details:
            result['details'] = details
        return result

    def _save_results(self, image, polygons, unique_idx, uniqueness_report, image_path):
        """保存可视化结果和分析报告"""
        try:
            self._save_visualization(image, polygons, unique_idx, image_path)
        except Exception as e:
            logger.error(f"保存可视化结果时发生错误: {e}")

        try:
            self._save_unique_polygon(image, polygons[unique_idx], image_path)
        except Exception as e:
            logger.error(f"保存唯一联通体图像时发生错误: {e}")
        
        try:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            report_path = os.path.join(self.config.OUTPUT_DIR, f"{base_name}_analysis.json")
            
            sanitized_polygons = []
            for polygon in polygons:
                sanitized_polygons.append({
                    'id': int(polygon['id']),
                    'bbox': [int(polygon['bbox'][0]), int(polygon['bbox'][1]), 
                             int(polygon['bbox'][2]), int(polygon['bbox'][3])],
                    'area': float(polygon['area']),
                    'perimeter': float(polygon['perimeter']),
                    'vertices': int(polygon['vertices']),
                    'aspect_ratio': float(polygon['aspect_ratio']),
                    'extent': float(polygon['extent']),
                    'solidity': float(polygon['solidity'])
                })
            
            report_content = {
                'image': os.path.basename(image_path),
                'most_unique_index': int(unique_idx),
                'uniqueness_report': uniqueness_report,
                'polygons': sanitized_polygons
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_content, f, ensure_ascii=False, indent=2)
            
            logger.info(f"分析报告已保存: {report_path}")
        except Exception as e:
            logger.error(f"保存分析报告时发生错误: {e}")

    def _save_visualization(self, image, polygons, unique_idx, original_path):
        """保存可视化结果"""
        # 绘制所有多边形
        result_image = self.detector.visualize_polygons(image, polygons, unique_idx)
        
        # 保存结果图像
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        output_path = os.path.join(self.config.OUTPUT_DIR, f"{base_name}_result.png")
        cv2.imwrite(output_path, result_image)
        print(f"   可视化结果已保存: {output_path}")
        
        # 保存详细报告
        report_path = os.path.join(self.config.OUTPUT_DIR, f"{base_name}_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            # 准备可序列化的报告数据
            serializable_report = {}
            for key, value in polygons[unique_idx].items():
                if key in ['contour', 'polygon']:
                    serializable_report[key] = value.tolist()
                else:
                    serializable_report[key] = value

            json.dump(serializable_report, f, ensure_ascii=False, indent=2)
        print(f"   详细报告已保存: {report_path}")

    def _save_unique_polygon(self, image, polygon_info, original_path):
        """将最终选中的联通体单独导出为PNG"""
        if image is None or polygon_info is None:
            raise ValueError("原始图像或多边形信息为空，无法保存唯一联通体")

        polygon = polygon_info.get('polygon')
        bbox = polygon_info.get('bbox')
        if polygon is None or bbox is None:
            raise ValueError("多边形数据缺失，无法导出唯一联通体")

        # 创建掩码并填充唯一多边形区域
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [polygon], -1, 255, thickness=-1)

        # 使用掩码提取该联通体
        isolated = cv2.bitwise_and(image, image, mask=mask)

        x, y, w, h = bbox
        x = max(0, int(x))
        y = max(0, int(y))
        w = max(1, int(w))
        h = max(1, int(h))
        cropped = isolated[y:y + h, x:x + w]

        base_name = os.path.splitext(os.path.basename(original_path))[0]
        unique_path = os.path.join(self.config.OUTPUT_DIR, f"{base_name}_unique.png")
        cv2.imwrite(unique_path, cropped)
        logger.info(f"唯一联通体图像已保存: {unique_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多边形唯一性检测工具')
    parser.add_argument('image_path', help='输入PNG图像路径')
    parser.add_argument('--min-area', type=float, help='最小多边形面积')
    parser.add_argument('--max-area', type=float, help='最大多边形面积')
    parser.add_argument('--config', help='配置文件路径')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.image_path):
        print(f"错误: 图像文件不存在: {args.image_path}")
        sys.exit(1)
    
    # 加载配置
    config = Config()
    if args.config and os.path.exists(args.config):
        # 这里可以添加从文件加载配置的逻辑
        pass
    
    # 创建检测器
    detector = PolygonUniquenessDetector(config)
    
    try:
        # 处理图像
        result = detector.process_image(args.image_path, args.min_area, args.max_area)
        
        if result['success']:
            print("\n=== 处理结果 ===")
            coords = result['result']
            print(f"最唯一多边形边界矩形坐标:")
            print(f"左上角: {coords['top_left']}")
            print(f"右下角: {coords['bottom_right']}")
            print(f"唯一性评分: {coords['uniqueness_score']:.3f}")
            
            # 输出标准格式结果
            print(f"\n输出结果: {coords['top_left']}, {coords['bottom_right']}")
        else:
            print(f"处理失败: {result['message']}")
            sys.exit(1)
    
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
