#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio可视化处理器
用于支持步骤化的图像处理可视化
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
import io
import base64
from PIL import Image

from config import Config
from polygon_detector import PolygonDetector
from similarity_calculator import SimilarityCalculator
from uniqueness_analyzer import UniquenessAnalyzer


class GradioVisualizer:
    """Gradio可视化处理器"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.detector = PolygonDetector(self.config, debug=False)
        self.analyzer = UniquenessAnalyzer(self.config, debug=False)
        
        # 存储处理步骤的结果
        self.processing_steps = []
        
    def process_image_with_steps(self, image_path: str, min_area: float = None, max_area: float = None) -> Dict[str, Any]:
        """
        处理图像并返回每个步骤的可视化结果
        """
        self.processing_steps = []
        
        try:
            # 步骤1: 加载原始图像
            original_image = cv2.imread(image_path)
            if original_image is None:
                return {"error": "无法加载图像文件"}
            
            step1_result = self._create_step_result(
                "步骤1: 加载原始图像",
                "成功加载图像文件",
                self._cv2_to_pil(original_image)
            )
            self.processing_steps.append(step1_result)
            
            # 步骤2: 图像预处理和二值化
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, self.config.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
            
            step2_result = self._create_step_result(
                "步骤2: 图像预处理",
                f"灰度化和二值化处理 (阈值: {self.config.BINARY_THRESHOLD})",
                self._cv2_to_pil(binary)
            )
            self.processing_steps.append(step2_result)
            
            # 步骤3: 轮廓检测
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = original_image.copy()
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
            
            step3_result = self._create_step_result(
                "步骤3: 轮廓检测",
                f"检测到 {len(contours)} 个轮廓",
                self._cv2_to_pil(contour_image)
            )
            self.processing_steps.append(step3_result)
            
            # 步骤4: 多边形检测和过滤
            polygons, _ = self.detector.detect_polygons(image_path)
            
            # 绘制所有检测到的多边形
            polygon_image = original_image.copy()
            for i, polygon in enumerate(polygons):
                color = (0, 255, 0)  # 绿色
                cv2.drawContours(polygon_image, [polygon['polygon']], -1, color, 2)
                # 添加编号
                bbox = polygon['bbox']
                cv2.putText(polygon_image, str(i), 
                           (int(bbox[0]), int(bbox[1]-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            step4_result = self._create_step_result(
                "步骤4: 多边形检测",
                f"检测到 {len(polygons)} 个有效多边形",
                self._cv2_to_pil(polygon_image)
            )
            self.processing_steps.append(step4_result)
            
            if len(polygons) == 0:
                return {
                    "steps": self.processing_steps,
                    "error": "未检测到任何多边形"
                }
            
            # 步骤5: 面积过滤
            if min_area is not None or max_area is not None:
                filtered_polygons = self.analyzer.filter_by_size(polygons, min_area, max_area)
                
                # 绘制过滤后的多边形
                filtered_image = original_image.copy()
                for i, polygon in enumerate(filtered_polygons):
                    color = (255, 0, 0)  # 蓝色
                    cv2.drawContours(filtered_image, [polygon['polygon']], -1, color, 2)
                    bbox = polygon['bbox']
                    cv2.putText(filtered_image, str(polygon['id']), 
                               (int(bbox[0]), int(bbox[1]-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                step5_result = self._create_step_result(
                    "步骤5: 面积过滤",
                    f"过滤后剩余 {len(filtered_polygons)} 个多边形",
                    self._cv2_to_pil(filtered_image)
                )
                self.processing_steps.append(step5_result)
            else:
                filtered_polygons = polygons
            
            if len(filtered_polygons) == 0:
                return {
                    "steps": self.processing_steps,
                    "error": "过滤后未剩余任何多边形"
                }
            
            # 步骤6: 相似度计算和唯一性分析
            uniqueness_report = self.analyzer.get_uniqueness_report(filtered_polygons)
            
            # 创建相似度矩阵可视化
            similarity_viz = self._create_similarity_matrix_visualization(uniqueness_report)
            
            step6_result = self._create_step_result(
                "步骤6: 相似度分析",
                f"计算了 {len(filtered_polygons)} 个多边形之间的相似度",
                similarity_viz
            )
            self.processing_steps.append(step6_result)
            
            # 步骤7: 最终结果
            most_unique_idx = uniqueness_report['most_unique_index']
            most_unique_polygon = filtered_polygons[most_unique_idx]
            
            # 绘制最终结果
            final_image = original_image.copy()
            
            # 绘制所有多边形（灰色）
            for polygon in filtered_polygons:
                cv2.drawContours(final_image, [polygon['polygon']], -1, (128, 128, 128), 1)
            
            # 高亮最唯一的多边形（红色）
            cv2.drawContours(final_image, [most_unique_polygon['polygon']], -1, (0, 0, 255), 3)
            
            # 绘制边界矩形
            top_left, bottom_right = self.analyzer.get_bounding_box_coordinates(most_unique_polygon)
            cv2.rectangle(final_image, top_left, bottom_right, (255, 0, 0), 2)
            
            # 添加标注
            cv2.putText(final_image, f"Most Unique (Score: {uniqueness_report['most_unique_score']:.3f})", 
                       (top_left[0], top_left[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            step7_result = self._create_step_result(
                "步骤7: 最终结果",
                f"最唯一多边形ID: {most_unique_idx}, 唯一性评分: {uniqueness_report['most_unique_score']:.3f}",
                self._cv2_to_pil(final_image)
            )
            self.processing_steps.append(step7_result)
            
            return {
                "steps": self.processing_steps,
                "result": {
                    "top_left": top_left,
                    "bottom_right": bottom_right,
                    "uniqueness_score": uniqueness_report['most_unique_score'],
                    "polygon_info": most_unique_polygon,
                    "full_report": uniqueness_report
                }
            }
            
        except Exception as e:
            error_step = self._create_step_result(
                "错误",
                f"处理过程中发生错误: {str(e)}",
                None
            )
            self.processing_steps.append(error_step)
            return {
                "steps": self.processing_steps,
                "error": str(e)
            }
    
    def _create_step_result(self, title: str, description: str, image: Image.Image = None) -> Dict[str, Any]:
        """创建步骤结果"""
        return {
            "title": title,
            "description": description,
            "image": image
        }
    
    def _cv2_to_pil(self, cv2_image) -> Image.Image:
        """将OpenCV图像转换为PIL图像"""
        if len(cv2_image.shape) == 3:
            # BGR to RGB
            rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        else:
            # 灰度图像
            rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(rgb_image)
    
    def _create_similarity_matrix_visualization(self, uniqueness_report: Dict) -> Image.Image:
        """创建相似度矩阵可视化"""
        try:
            similarity_matrix = uniqueness_report.get('similarity_matrix', [])
            if not similarity_matrix:
                # 创建一个简单的文本图像
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.text(0.5, 0.5, '相似度矩阵数据不可用', 
                       ha='center', va='center', fontsize=14)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            else:
                # 创建相似度矩阵热图
                fig, ax = plt.subplots(figsize=(8, 6))
                
                matrix = np.array(similarity_matrix)
                im = ax.imshow(matrix, cmap='viridis', aspect='auto')
                
                # 添加颜色条
                plt.colorbar(im, ax=ax, label='相似度')
                
                # 设置标签
                ax.set_xlabel('多边形ID')
                ax.set_ylabel('多边形ID')
                ax.set_title('多边形相似度矩阵')
                
                # 添加数值标注
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                     ha="center", va="center", color="white", fontsize=8)
            
            # 将matplotlib图像转换为PIL图像
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            pil_image = Image.open(buf)
            plt.close(fig)
            
            return pil_image
            
        except Exception as e:
            # 如果出错，返回错误信息图像
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f'相似度矩阵生成失败:\n{str(e)}', 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            pil_image = Image.open(buf)
            plt.close(fig)
            
            return pil_image