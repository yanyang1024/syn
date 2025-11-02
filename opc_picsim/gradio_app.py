#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio应用：多边形唯一性检测可视化
上传图片后，展示从预处理、轮廓/多边形、相似度矩阵、唯一性评分到最终unique pattern的全过程。
"""

import os
import cv2
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Tuple, List

from config import Config
from polygon_detector import PolygonDetector
from uniqueness_analyzer import UniquenessAnalyzer
from debug_utils import DebugVisualizer


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def crop_unique(image: np.ndarray, polygon_info: dict) -> np.ndarray:
    polygon = polygon_info.get("polygon")
    bbox = polygon_info.get("bbox")
    if polygon is None or bbox is None:
        return None

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [polygon], -1, 255, thickness=-1)
    isolated = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = bbox
    x = max(0, int(x))
    y = max(0, int(y))
    w = max(1, int(w))
    h = max(1, int(h))
    cropped = isolated[y : y + h, x : x + w]
    return cropped


def analyze_image(
    image_path: str,
    min_area: float = None,
    max_area: float = None,
    binary_threshold: int = None,
):
    # 配置与组件
    cfg = Config()
    if binary_threshold is not None:
        cfg.BINARY_THRESHOLD = int(binary_threshold)

    detector = PolygonDetector(cfg, debug=True)
    analyzer = UniquenessAnalyzer(cfg, debug=True)
    visualizer = DebugVisualizer(output_dir="debug_output", save_plots=True)

    # 加载原图与预处理
    original = detector.load_image(image_path)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
    binary = detector.preprocess_image(original)

    # 轮廓与多边形
    contours = detector.find_contours(binary)

    # 复制detect_polygons逻辑以生成polygons
    polygons: List[dict] = []
    for i, contour in enumerate(contours):
        try:
            polygon = detector.approximate_polygon(contour)
            if len(polygon) < 3:
                continue
            x, y, w, h = detector.get_bounding_rect(polygon)
            area = cv2.contourArea(polygon)
            perimeter = cv2.arcLength(polygon, True)
            aspect_ratio = w / h if h > 0 else 0
            extent = area / (w * h) if (w * h) > 0 else 0
            solidity = area / cv2.contourArea(cv2.convexHull(polygon)) if cv2.contourArea(cv2.convexHull(polygon)) > 0 else 0
            polygons.append(
                {
                    "id": i,
                    "contour": contour,
                    "polygon": polygon,
                    "bbox": (x, y, w, h),
                    "area": area,
                    "perimeter": perimeter,
                    "vertices": len(polygon),
                    "aspect_ratio": aspect_ratio,
                    "extent": extent,
                    "solidity": solidity,
                }
            )
        except Exception:
            continue

    # 面积过滤
    filtered_polygons = analyzer.filter_by_size(polygons, min_area, max_area)

    # 唯一性分析
    report = analyzer.get_uniqueness_report(filtered_polygons)
    most_unique_idx = report["most_unique_index"]

    overlay_np = None
    cropped_unique_np = None
    bbox_text = ""
    score_text = ""

    if most_unique_idx is not None and most_unique_idx >= 0 and len(filtered_polygons) > 0:
        unique_id = filtered_polygons[most_unique_idx]["id"]
        overlay_bgr = detector.visualize_polygons(original, filtered_polygons, unique_polygon_id=unique_id)
        overlay_np = bgr_to_rgb(overlay_bgr)
        most_unique_polygon = filtered_polygons[most_unique_idx]
        cropped_bgr = crop_unique(original, most_unique_polygon)
        cropped_unique_np = bgr_to_rgb(cropped_bgr)
        x, y, w, h = most_unique_polygon["bbox"]
        bbox_text = f"唯一联通体边界框: (x={x}, y={y}, w={w}, h={h})"
        score_text = f"唯一性评分: {report['most_unique_score']:.3f}"
    else:
        overlay_np = bgr_to_rgb(original)
        cropped_unique_np = None
        bbox_text = "未找到最唯一多边形"
        score_text = ""

    # 可视化保存（采用DebugVisualizer生成图像文件路径）
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    visualizer.visualize_preprocessing_steps(original, gray, binary, image_name=base_name)
    visualizer.visualize_contours_and_polygons(original, contours, filtered_polygons, image_name=base_name)

    # 唯一性评分与相似度矩阵
    # 需要从report中取similarity_matrix与scores
    similarity_matrix = np.array(report.get("similarity_matrix", []))
    uniqueness_scores = np.array(report.get("uniqueness_scores", []))
    polygon_ids = [p["id"] for p in filtered_polygons]

    if similarity_matrix.size > 0:
        visualizer.visualize_similarity_matrix(similarity_matrix, polygon_ids, image_name=base_name)
    if uniqueness_scores.size > 0 and len(filtered_polygons) > 0:
        visualizer.visualize_uniqueness_scores(filtered_polygons, uniqueness_scores, most_unique_idx, image_name=base_name)

    # 构造保存路径
    preproc_path = os.path.join(visualizer.output_dir, f"{base_name}_preprocessing.png")
    contours_polys_path = os.path.join(visualizer.output_dir, f"{base_name}_contours_polygons.png")
    sim_mat_path = os.path.join(visualizer.output_dir, f"{base_name}_similarity_matrix.png")
    uniq_scores_path = os.path.join(visualizer.output_dir, f"{base_name}_uniqueness_analysis.png")

    summary_text = report.get("analysis_summary", "")

    return (
        preproc_path,
        contours_polys_path,
        sim_mat_path if os.path.exists(sim_mat_path) else None,
        uniq_scores_path if os.path.exists(uniq_scores_path) else None,
        overlay_np,
        cropped_unique_np,
        summary_text,
        bbox_text,
        score_text,
    )


def build_app():
    with gr.Blocks(title="多边形唯一性检测可视化") as demo:
        gr.Markdown("# 多边形唯一性检测可视化")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.File(label="上传图片", file_types=[".png", ".jpg", ".jpeg", ".bmp"], type="filepath")
                min_area = gr.Number(label="最小面积(可选)")
                max_area = gr.Number(label="最大面积(可选)")
                binary_th = gr.Slider(label="二值化阈值", minimum=0, maximum=255, step=1, value=Config.BINARY_THRESHOLD)
                btn = gr.Button("开始分析", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown("### 预处理: 原图/灰度/二值化")
                out_preproc = gr.Image(type="filepath")
                gr.Markdown("### 轮廓与多边形")
                out_contours_polys = gr.Image(type="filepath")
                gr.Markdown("### 相似度矩阵与唯一性评分")
                out_sim_mat = gr.Image(type="filepath")
                out_uniq_scores = gr.Image(type="filepath")
                gr.Markdown("### 高亮唯一联通体与裁剪结果")
                out_overlay = gr.Image(type="numpy")
                out_cropped = gr.Image(type="numpy")
                gr.Markdown("### 结果摘要")
                out_summary = gr.Textbox()
                out_bbox = gr.Textbox()
                out_score = gr.Textbox()

        def _run(image_path, min_a, max_a, th):
            return analyze_image(image_path, min_a, max_a, th)

        btn.click(
            fn=_run,
            inputs=[inp, min_area, max_area, binary_th],
            outputs=[
                out_preproc,
                out_contours_polys,
                out_sim_mat,
                out_uniq_scores,
                out_overlay,
                out_cropped,
                out_summary,
                out_bbox,
                out_score,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()