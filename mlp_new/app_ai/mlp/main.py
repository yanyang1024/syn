"""
主程序入口 - 深度学习回归预测系统
"""
import argparse
import sys
import os
from datetime import datetime

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='深度学习回归预测系统')
    parser.add_argument('--mode', type=str, default='gui', 
                       choices=['gui', 'train', 'inference', 'analysis'],
                       help='运行模式: gui(界面), train(训练), inference(推理), analysis(分析)')
    parser.add_argument('--models', type=str, nargs='+', default=['mlp'],
                       choices=['mlp', 'resnet', 'transformer', 'ensemble'],
                       help='选择模型类型')
    parser.add_argument('--groups', type=str, nargs='+', 
                       default=['group_1', 'group_2', 'group_3', 'group_4', 'group_5'],
                       help='选择数据组')
    parser.add_argument('--x_file', type=str, default=None,
                       help='输入数据文件路径')
    parser.add_argument('--y_file', type=str, default=None,
                       help='输出数据文件路径')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='学习率')
    
    args = parser.parse_args()
    
    print("🧠 深度学习回归预测系统")
    print("=" * 50)
    print(f"运行模式: {args.mode}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    if args.mode == 'gui':
        run_gui()
    elif args.mode == 'train':
        run_training(args)
    elif args.mode == 'inference':
        run_inference(args)
    elif args.mode == 'analysis':
        run_analysis()
    else:
        print(f"未知模式: {args.mode}")
        sys.exit(1)

def run_gui():
    """运行GUI界面"""
    try:
        print("启动Gradio界面...")
        from gradio_app import main as gradio_main
        gradio_main()
    except ImportError as e:
        print(f"导入Gradio模块失败: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
    except Exception as e:
        print(f"启动GUI失败: {e}")

def run_training(args):
    """运行训练"""
    try:
        print("开始训练模式...")
        
        from config import Config
        from data_processor import DataProcessor
        from trainer import Trainer
        
        # 更新配置
        config = Config()
        if args.epochs:
            if args.epochs <= 0:
                print("错误: 训练轮数必须为正整数")
                return
            config.EPOCHS = args.epochs
        if args.batch_size:
            if args.batch_size <= 0:
                print("错误: 批次大小必须为正整数")
                return
            config.BATCH_SIZE = args.batch_size
        if args.learning_rate:
            if args.learning_rate <= 0:
                print("错误: 学习率必须为正数")
                return
            config.LEARNING_RATE = args.learning_rate
        
        print(f"配置: epochs={config.EPOCHS}, batch_size={config.BATCH_SIZE}, lr={config.LEARNING_RATE}")
        
        # 数据处理
        print("处理数据...")
        processor = DataProcessor(config)
        
        if args.x_file and args.y_file:
            if not os.path.exists(args.x_file):
                print(f"错误: 输入文件不存在: {args.x_file}")
                return
            if not os.path.exists(args.y_file):
                print(f"错误: 输出文件不存在: {args.y_file}")
                return
            X, y = processor.load_data(args.x_file, args.y_file)
            processed_groups = processor.process_all_groups(X, y)
        else:
            processed_groups = processor.process_all_groups()
        
        # 过滤选中的组
        if args.groups:
            valid_groups = [g for g in args.groups if g in processed_groups]
            if not valid_groups:
                print(f"错误: 没有找到有效的数据组: {args.groups}")
                return
            processed_groups = {k: v for k, v in processed_groups.items() if k in valid_groups}
        
        # 验证模型类型
        valid_models = ['mlp', 'resnet', 'transformer', 'ensemble']
        invalid_models = [m for m in args.models if m not in valid_models]
        if invalid_models:
            print(f"错误: 无效的模型类型: {invalid_models}")
            print(f"支持的模型类型: {valid_models}")
            return
        
        # 训练
        print(f"训练模型: {args.models}")
        print(f"数据组: {list(processed_groups.keys())}")
        
        trainer = Trainer(config)
        results = trainer.train_all_groups(processed_groups, args.models)
        
        print("训练完成!")
        print("结果已保存到 logs/ 目录")
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()

def run_inference(args):
    """运行推理"""
    try:
        print("开始推理模式...")
        
        from inference import load_all_models
        import numpy as np
        
        # 加载模型
        print("加载模型...")
        evaluator = load_all_models()
        
        if not evaluator.inference.models:
            print("未找到训练好的模型，请先运行训练")
            return
        
        print(f"成功加载 {len(evaluator.inference.models)} 个模型")
        
        # 示例推理
        print("执行示例推理...")
        X_sample = np.random.randn(5, 7)  # 5个样本
        
        for group in args.groups:
            for model_type in args.models:
                model_key = f"{group}_{model_type}"
                
                if model_key in evaluator.inference.models:
                    try:
                        predictions = evaluator.inference.predict(model_key, X_sample, group)
                        print(f"{model_key} 预测完成，输出形状: {predictions.shape}")
                    except Exception as e:
                        print(f"{model_key} 预测失败: {e}")
        
        print("推理完成!")
        
    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()

def run_analysis():
    """运行分析"""
    try:
        print("开始分析模式...")
        
        from hyperparameter_analysis import HyperparameterAnalyzer
        
        # 生成超参数分析报告
        print("生成超参数分析报告...")
        analyzer = HyperparameterAnalyzer()
        analyzer.save_analysis_report()
        
        # 如果有训练结果，生成评估报告
        if os.path.exists('logs/training_results.json'):
            print("生成训练结果分析...")
            from inference import load_all_models
            
            evaluator = load_all_models()
            # 这里可以添加更多分析功能
        
        print("分析完成!")
        print("报告已保存到当前目录")
        
    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()

def check_dependencies():
    """检查依赖"""
    required_packages = [
        'torch', 'numpy', 'pandas', 'sklearn', 
        'matplotlib', 'seaborn', 'gradio', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def show_help():
    """显示帮助信息"""
    help_text = """
🧠 深度学习回归预测系统 - 使用指南

基本用法:
  python main.py [选项]

运行模式:
  --mode gui          启动Web界面 (默认)
  --mode train        命令行训练模式
  --mode inference    推理模式
  --mode analysis     分析模式

训练选项:
  --models mlp resnet transformer    选择模型类型
  --groups group_1 group_2          选择数据组
  --epochs 200                      训练轮数
  --batch_size 64                   批次大小
  --learning_rate 0.001             学习率
  --x_file path/to/x_input.csv      输入数据文件
  --y_file path/to/y_output.csv     输出数据文件

使用示例:
  # 启动Web界面
  python main.py

  # 训练MLP和ResNet模型
  python main.py --mode train --models mlp resnet

  # 训练指定组的数据
  python main.py --mode train --groups group_1 group_2 --epochs 100

  # 使用自定义数据训练
  python main.py --mode train --x_file data/x.csv --y_file data/y.csv

  # 运行推理
  python main.py --mode inference --models mlp --groups group_1

  # 生成分析报告
  python main.py --mode analysis

更多信息请查看 README.md
"""
    print(help_text)

if __name__ == "__main__":
    # 检查是否需要显示帮助
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_help()
        sys.exit(0)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 运行主程序
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)