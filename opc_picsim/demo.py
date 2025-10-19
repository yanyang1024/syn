#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多边形唯一性检测演示程序
提供简单的交互式界面
"""

import os
import cv2
import numpy as np
from main import PolygonUniquenessDetector
from config import Config

def create_sample_image():
    """创建一个包含多个多边形的示例图像"""
    # 创建白色背景
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # 绘制几个不同的多边形
    # 三角形
    triangle = np.array([[100, 100], [200, 100], [150, 50]], np.int32)
    cv2.fillPoly(img, [triangle], (0, 0, 0))
    
    # 矩形
    cv2.rectangle(img, (300, 80), (400, 120), (0, 0, 0), -1)
    
    # 五边形
    pentagon = np.array([[500, 100], [550, 80], [570, 120], [530, 150], [480, 130]], np.int32)
    cv2.fillPoly(img, [pentagon], (0, 0, 0))
    
    # 另一个三角形（相似但不同位置）
    triangle2 = np.array([[150, 300], [250, 300], [200, 250]], np.int32)
    cv2.fillPoly(img, [triangle2], (0, 0, 0))
    
    # 圆形（近似多边形）
    center = (400, 300)
    radius = 40
    cv2.circle(img, center, radius, (0, 0, 0), -1)
    
    # 六边形
    hexagon = np.array([[600, 300], [640, 280], [680, 300], [680, 340], [640, 360], [600, 340]], np.int32)
    cv2.fillPoly(img, [hexagon], (0, 0, 0))
    
    # 不规则多边形
    irregular = np.array([[100, 450], [180, 420], [220, 480], [160, 520], [80, 500]], np.int32)
    cv2.fillPoly(img, [irregular], (0, 0, 0))
    
    return img

def interactive_demo():
    """交互式演示"""
    print("=== 多边形唯一性检测演示程序 ===\n")
    
    # 创建检测器
    config = Config()
    detector = PolygonUniquenessDetector(config)
    
    while True:
        print("请选择操作:")
        print("1. 使用示例图像")
        print("2. 输入图像路径")
        print("3. 退出")
        
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == '1':
            # 创建并保存示例图像
            sample_img = create_sample_image()
            sample_path = "sample_polygons.png"
            cv2.imwrite(sample_path, sample_img)
            print(f"已创建示例图像: {sample_path}")
            
            # 处理示例图像
            process_image_interactive(detector, sample_path)
            
        elif choice == '2':
            image_path = input("请输入图像路径: ").strip()
            if os.path.exists(image_path):
                process_image_interactive(detector, image_path)
            else:
                print("错误: 文件不存在!")
                
        elif choice == '3':
            print("退出程序")
            break
            
        else:
            print("无效选择，请重新输入")
        
        print("\n" + "="*50 + "\n")

def process_image_interactive(detector, image_path):
    """交互式处理图像"""
    print(f"\n正在处理图像: {image_path}")
    
    # 询问是否设置面积过滤
    use_filter = input("是否设置面积过滤? (y/n): ").strip().lower()
    min_area, max_area = None, None
    
    if use_filter == 'y':
        try:
            min_area = float(input("请输入最小面积 (默认100): ") or "100")
            max_area = float(input("请输入最大面积 (默认50000): ") or "50000")
        except ValueError:
            print("输入无效，使用默认值")
            min_area, max_area = 100, 50000
    
    try:
        # 处理图像
        result = detector.process_image(image_path, min_area, max_area)
        
        if result['success']:
            print("\n=== 处理成功 ===")
            coords = result['result']
            report = coords['full_report']
            
            print(f"检测到多边形数量: {report['total_polygons']}")
            print(f"最唯一多边形ID: {report['most_unique_index']}")
            print(f"唯一性评分: {coords['uniqueness_score']:.3f}")
            print(f"边界矩形坐标: {coords['top_left']} -> {coords['bottom_right']}")
            
            # 显示详细信息
            polygon_info = coords['polygon_info']
            print(f"\n多边形详细信息:")
            print(f"  面积: {polygon_info['area']:.2f}")
            print(f"  周长: {polygon_info['perimeter']:.2f}")
            print(f"  顶点数: {polygon_info['vertices']}")
            
            # 显示相似度分析
            if len(report['similar_pairs']) > 0:
                print(f"\n发现 {len(report['similar_pairs'])} 对相似多边形:")
                for i, (id1, id2, similarity) in enumerate(report['similar_pairs'][:3]):  # 只显示前3对
                    print(f"  多边形 {id1} 与 {id2}: 相似度 {similarity:.3f}")
            else:
                print("\n未发现高度相似的多边形对")
            
            print(f"\n分析摘要: {report['analysis_summary']}")
            
        else:
            print(f"处理失败: {result['message']}")
    
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")

if __name__ == "__main__":
    interactive_demo()