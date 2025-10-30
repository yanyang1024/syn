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
            uniqueness_report = self.analyzer.get_uniqueness_report(filtered_polygons)
            
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
