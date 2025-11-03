#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多边形唯一性检测 Gradio Web界面
提供交互式的图像处理和可视化功能
"""

import os
import gradio as gr
import tempfile
from typing import List, Tuple, Any
from PIL import Image

from config import Config
from gradio_visualizer import GradioVisualizer


class PolygonDetectionApp:
    """多边形检测Gradio应用"""
    
    def __init__(self):
        self.config = Config()
        self.visualizer = GradioVisualizer(self.config)
        
    def process_image(self, image: Image.Image, min_area: float = None, max_area: float = None) -> Tuple[List[Any], str]:
        """
        处理上传的图像
        
        Args:
            image: 上传的PIL图像
            min_area: 最小面积过滤
            max_area: 最大面积过滤
            
        Returns:
            处理步骤的图像列表和结果文本
        """
        if image is None:
            return [], "请先上传图像文件"
        
        # 保存临时图像文件
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            image.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            # 处理图像
            result = self.visualizer.process_image_with_steps(
                temp_path, min_area, max_area
            )
            
            # 清理临时文件
            os.unlink(temp_path)
            
            if "error" in result:
                return [], f"处理失败: {result['error']}"
            
            # 提取步骤图像
            step_images = []
            step_descriptions = []
            
            for step in result["steps"]:
                if step["image"] is not None:
                    step_images.append(step["image"])
                    step_descriptions.append(f"{step['title']}: {step['description']}")
            
            # 生成结果文本
            if "result" in result:
                res = result["result"]
                result_text = f"""
## 处理完成！

### 最终结果:
- **最唯一多边形边界矩形坐标:**
  - 左上角: {res['top_left']}
  - 右下角: {res['bottom_right']}
- **唯一性评分:** {res['uniqueness_score']:.3f}

### 多边形信息:
- **ID:** {res['polygon_info']['id']}
- **面积:** {res['polygon_info']['area']:.2f}
- **周长:** {res['polygon_info']['perimeter']:.2f}
- **顶点数:** {res['polygon_info']['vertices']}
- **长宽比:** {res['polygon_info']['aspect_ratio']:.3f}

### 处理步骤:
{chr(10).join([f"{i+1}. {desc}" for i, desc in enumerate(step_descriptions)])}
                """
            else:
                result_text = "处理完成，但未找到结果数据"
            
            return step_images, result_text
            
        except Exception as e:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return [], f"处理过程中发生错误: {str(e)}"
    
    def create_interface(self):
        """创建Gradio界面"""
        
        # 自定义CSS样式
        custom_css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .step-gallery {
            height: 400px;
        }
        .result-text {
            font-family: 'Courier New', monospace;
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #007acc;
        }
        """
        
        with gr.Blocks(css=custom_css, title="多边形唯一性检测") as interface:
            
            # 标题和说明
            gr.Markdown("""
            # 🔍 多边形唯一性检测系统
            
            上传PNG图像，系统将自动检测图像中的多边形，分析它们的相似性，并找出最具唯一性的多边形。
            
            ## 使用说明:
            1. 上传PNG格式的图像文件
            2. 可选：设置面积过滤范围
            3. 点击"开始处理"按钮
            4. 查看处理步骤和最终结果
            """)
            
            with gr.Row():
                # 左侧：输入区域
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 输入设置")
                    
                    # 图像上传
                    input_image = gr.Image(
                        label="上传图像",
                        type="pil",
                        height=300
                    )
                    
                    # 参数设置
                    with gr.Group():
                        gr.Markdown("#### 过滤参数 (可选)")
                        min_area = gr.Number(
                            label="最小面积",
                            value=None,
                            placeholder="留空表示不限制"
                        )
                        max_area = gr.Number(
                            label="最大面积", 
                            value=None,
                            placeholder="留空表示不限制"
                        )
                    
                    # 处理按钮
                    process_btn = gr.Button(
                        "🚀 开始处理",
                        variant="primary",
                        size="lg"
                    )
                
                # 右侧：结果区域
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 处理结果")
                    
                    # 结果文本
                    result_text = gr.Markdown(
                        value="等待处理...",
                        elem_classes=["result-text"]
                    )
            
            # 处理步骤可视化
            gr.Markdown("### 🔄 处理步骤可视化")
            step_gallery = gr.Gallery(
                label="处理步骤",
                show_label=True,
                elem_id="step-gallery",
                columns=3,
                rows=3,
                height=500,
                object_fit="contain"
            )
            
            # 配置参数显示
            with gr.Accordion("⚙️ 当前配置参数", open=False):
                config_info = gr.Markdown(f"""
                **图像处理参数:**
                - 二值化阈值: {self.config.BINARY_THRESHOLD}
                - 最小轮廓面积: {self.config.MIN_CONTOUR_AREA}
                - 最大轮廓面积: {self.config.MAX_CONTOUR_AREA}
                
                **相似度计算权重:**
                - 形状相似度: {self.config.SHAPE_SIMILARITY_WEIGHT}
                - 尺寸相似度: {self.config.SIZE_SIMILARITY_WEIGHT}
                - 方向相似度: {self.config.ORIENTATION_SIMILARITY_WEIGHT}
                
                **其他参数:**
                - 相似度阈值: {self.config.SIMILARITY_THRESHOLD}
                - 多边形近似精度: {self.config.EPSILON_FACTOR}
                """)
            
            # 示例图像
            with gr.Accordion("📋 示例图像", open=False):
                gr.Markdown("""
                如果您没有测试图像，可以尝试以下类型的图像：
                - 包含多个几何形状的技术图纸
                - 电路板布局图
                - 建筑平面图
                - 包含多边形的示意图
                
                **注意:** 图像应该是PNG格式，背景最好是白色或浅色，多边形为深色。
                """)
            
            # 绑定事件
            process_btn.click(
                fn=self.process_image,
                inputs=[input_image, min_area, max_area],
                outputs=[step_gallery, result_text],
                show_progress=True
            )
            
            # 示例处理（如果有示例图像）
            examples_dir = "examples"
            if os.path.exists(examples_dir):
                example_files = [os.path.join(examples_dir, f) 
                               for f in os.listdir(examples_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if example_files:
                    gr.Examples(
                        examples=[[f] for f in example_files[:3]],  # 最多显示3个示例
                        inputs=[input_image],
                        label="示例图像"
                    )
        
        return interface


def main():
    """主函数"""
    app = PolygonDetectionApp()
    interface = app.create_interface()
    
    # 启动应用
    interface.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,       # 默认端口
        share=False,            # 不创建公共链接
        debug=True,             # 启用调试模式
        show_error=True         # 显示错误信息
    )


if __name__ == "__main__":
    main()