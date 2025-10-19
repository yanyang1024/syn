#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试工具模块
提供可视化和调试功能
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DebugVisualizer:
    """调试可视化工具"""
    
    def __init__(self, output_dir: str = "debug_output", save_plots: bool = True):
        self.output_dir = output_dir
        self.save_plots = save_plots
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"创建调试输出目录: {output_dir}")
    
    def visualize_preprocessing_steps(self, original: np.ndarray, gray: np.ndarray, 
                                    binary: np.ndarray, image_name: str = "image"):
        """可视化图像预处理步骤"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图
        if len(original.shape) == 3:
            axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(original, cmap='gray')
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 灰度图
        axes[1].imshow(gray, cmap='gray')
        axes[1].set_title('灰度图像')
        axes[1].axis('off')
        
        # 二值图
        axes[2].imshow(binary, cmap='gray')
        axes[2].set_title('二值化图像')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = os.path.join(self.output_dir, f"{image_name}_preprocessing.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"预处理步骤可视化已保存: {save_path}")
        
        plt.show()
        plt.close()
    
    def visualize_contours_and_polygons(self, image: np.ndarray, contours: List[np.ndarray], 
                                      polygons: List[dict], image_name: str = "image"):
        """可视化轮廓和多边形"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 显示轮廓
        contour_img = image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        
        if len(contour_img.shape) == 3:
            axes[0].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(contour_img, cmap='gray')
        axes[0].set_title(f'检测到的轮廓 ({len(contours)}个)')
        axes[0].axis('off')
        
        # 显示多边形
        polygon_img = image.copy()
        for i, poly_info in enumerate(polygons):
            # 绘制多边形
            cv2.drawContours(polygon_img, [poly_info['polygon']], -1, (255, 0, 0), 2)
            
            # 绘制边界矩形
            x, y, w, h = poly_info['bbox']
            cv2.rectangle(polygon_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            
            # 添加标签
            cv2.putText(polygon_img, f"ID:{i}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        if len(polygon_img.shape) == 3:
            axes[1].imshow(cv2.cvtColor(polygon_img, cv2.COLOR_BGR2RGB))
        else:
            axes[1].imshow(polygon_img, cmap='gray')
        axes[1].set_title(f'近似多边形 ({len(polygons)}个)')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = os.path.join(self.output_dir, f"{image_name}_contours_polygons.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"轮廓和多边形可视化已保存: {save_path}")
        
        plt.show()
        plt.close()
    
    def visualize_similarity_matrix(self, similarity_matrix: np.ndarray, 
                                  polygon_ids: List[int], image_name: str = "image"):
        """可视化相似度矩阵"""
        if similarity_matrix.size == 0:
            logger.warning("相似度矩阵为空，跳过可视化")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建热力图
        im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        
        # 设置标签
        ax.set_xticks(range(len(polygon_ids)))
        ax.set_yticks(range(len(polygon_ids)))
        ax.set_xticklabels([f"P{id}" for id in polygon_ids])
        ax.set_yticklabels([f"P{id}" for id in polygon_ids])
        
        # 添加数值标注
        for i in range(len(polygon_ids)):
            for j in range(len(polygon_ids)):
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                             ha="center", va="center", color="white" if similarity_matrix[i, j] < 0.5 else "black")
        
        ax.set_title("多边形相似度矩阵")
        ax.set_xlabel("多边形ID")
        ax.set_ylabel("多边形ID")
        
        # 添加颜色条
        cbar = plt.colorbar(im)
        cbar.set_label('相似度', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = os.path.join(self.output_dir, f"{image_name}_similarity_matrix.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"相似度矩阵可视化已保存: {save_path}")
        
        plt.show()
        plt.close()
    
    def visualize_uniqueness_scores(self, polygons: List[dict], uniqueness_scores: np.ndarray, 
                                  most_unique_idx: int, image_name: str = "image"):
        """可视化唯一性评分"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 柱状图显示唯一性评分
        polygon_ids = [p['id'] for p in polygons]
        colors = ['red' if i == most_unique_idx else 'blue' for i in range(len(polygons))]
        
        bars = ax1.bar(polygon_ids, uniqueness_scores, color=colors, alpha=0.7)
        ax1.set_xlabel('多边形ID')
        ax1.set_ylabel('唯一性评分')
        ax1.set_title('多边形唯一性评分')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, uniqueness_scores)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 散点图显示面积vs唯一性评分
        areas = [p['area'] for p in polygons]
        scatter = ax2.scatter(areas, uniqueness_scores, c=uniqueness_scores, 
                            cmap='viridis', s=100, alpha=0.7)
        
        # 高亮最唯一的多边形
        if most_unique_idx < len(areas):
            ax2.scatter(areas[most_unique_idx], uniqueness_scores[most_unique_idx], 
                       c='red', s=200, marker='*', label='最唯一多边形')
        
        ax2.set_xlabel('多边形面积')
        ax2.set_ylabel('唯一性评分')
        ax2.set_title('面积 vs 唯一性评分')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('唯一性评分', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = os.path.join(self.output_dir, f"{image_name}_uniqueness_analysis.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"唯一性分析可视化已保存: {save_path}")
        
        plt.show()
        plt.close()
    
    def save_debug_report(self, report_data: Dict, image_name: str = "image"):
        """保存调试报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"{image_name}_debug_report_{timestamp}.json")
        
        # 确保数据可序列化
        serializable_data = self._make_serializable(report_data)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"调试报告已保存: {report_path}")
        return report_path
    
    def _make_serializable(self, obj):
        """将对象转换为可序列化格式"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        else:
            return obj

class PerformanceProfiler:
    """性能分析工具"""
    
    def __init__(self):
        self.timings = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """开始计时"""
        import time
        self.start_times[name] = time.time()
        logger.debug(f"开始计时: {name}")
    
    def end_timer(self, name: str):
        """结束计时"""
        import time
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.timings[name] = elapsed
            logger.info(f"计时结束: {name} - 耗时 {elapsed:.3f} 秒")
            del self.start_times[name]
            return elapsed
        else:
            logger.warning(f"未找到计时器: {name}")
            return None
    
    def get_timing_report(self) -> str:
        """获取计时报告"""
        if not self.timings:
            return "无计时数据"
        
        report = "性能分析报告:\n"
        report += "=" * 40 + "\n"
        
        total_time = sum(self.timings.values())
        
        for name, time_taken in sorted(self.timings.items(), key=lambda x: x[1], reverse=True):
            percentage = (time_taken / total_time) * 100 if total_time > 0 else 0
            report += f"{name:25s}: {time_taken:8.3f}s ({percentage:5.1f}%)\n"
        
        report += "-" * 40 + "\n"
        report += f"{'总计':25s}: {total_time:8.3f}s (100.0%)\n"
        
        return report
    
    def reset(self):
        """重置计时器"""
        self.timings.clear()
        self.start_times.clear()
        logger.debug("性能分析器已重置")