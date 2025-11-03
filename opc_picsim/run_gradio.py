#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多边形唯一性检测 Gradio应用启动脚本
"""

import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from gradio_app import main
    
    if __name__ == "__main__":
        print("🚀 启动多边形唯一性检测Web界面...")
        print("📍 界面将在浏览器中打开: http://localhost:7860")
        print("⏹️  按 Ctrl+C 停止服务")
        print("-" * 50)
        
        main()
        
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("💡 请确保已安装所有依赖包:")
    print("   pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ 启动失败: {e}")
    sys.exit(1)