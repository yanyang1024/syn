"""
超参数分析和调整策略
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config

class HyperparameterAnalyzer:
    """超参数分析器"""
    
    def __init__(self, config=None):
        self.config = config or Config()
    
    def analyze_target_mse(self):
        """分析预期MSE目标值"""
        analysis = {
            "数据规模分析": {
                "输入维度": self.config.INPUT_DIM,
                "输出维度": self.config.OUTPUT_DIM,
                "数据组数": self.config.DATA_GROUPS,
                "总样本数": sum(self.config.GROUP_SIZES),
                "各组样本数": self.config.GROUP_SIZES
            },
            
            "MSE目标值设定": {
                "优秀水平 (excellent)": self.config.TARGET_MSE['excellent'],
                "良好水平 (good)": self.config.TARGET_MSE['good'],
                "可接受水平 (acceptable)": self.config.TARGET_MSE['acceptable']
            },
            
            "目标值设定依据": {
                "数据复杂度": "7输入->26输出的高维映射关系",
                "样本规模": "最小组4539样本，最大组18957样本",
                "预期精度": "基于回归问题的一般精度要求",
                "实际应用": "考虑实际应用场景的容错性"
            }
        }
        
        return analysis
    
    def generate_hyperparameter_strategies(self):
        """生成超参数调整策略"""
        strategies = {
            "学习率调整策略": {
                "初始学习率": 0.001,
                "调整原则": [
                    "训练损失下降缓慢 -> 增大学习率 (0.003-0.01)",
                    "训练损失震荡 -> 减小学习率 (0.0003-0.0001)",
                    "验证损失不下降 -> 减小学习率并增加正则化",
                    "过拟合严重 -> 减小学习率 + 增大dropout"
                ],
                "自适应策略": "使用ReduceLROnPlateau，验证损失停止改善时自动降低"
            },
            
            "网络结构调整": {
                "隐藏层数量": {
                    "MSE > 0.1": "增加隐藏层 (5-7层)",
                    "0.05 < MSE < 0.1": "保持当前结构或微调",
                    "MSE < 0.05": "可以尝试减少层数防止过拟合"
                },
                "隐藏层维度": {
                    "欠拟合": "增大隐藏层维度 [256, 512, 1024, 512, 256]",
                    "过拟合": "减小隐藏层维度 [64, 128, 256, 128, 64]",
                    "平衡": "当前配置 [128, 256, 512, 256, 128]"
                }
            },
            
            "正则化策略": {
                "Dropout调整": {
                    "过拟合": "增大dropout (0.3-0.5)",
                    "欠拟合": "减小dropout (0.1-0.2)",
                    "当前值": 0.2
                },
                "权重衰减": {
                    "过拟合": "增大weight_decay (1e-4 to 1e-3)",
                    "欠拟合": "减小weight_decay (1e-6 to 1e-5)",
                    "当前值": 1e-5
                }
            },
            
            "训练策略": {
                "批次大小": {
                    "内存充足": "增大batch_size (128-256) 提高训练稳定性",
                    "内存不足": "减小batch_size (32-64) 但可能需要调整学习率",
                    "当前值": 64
                },
                "训练轮数": {
                    "快速收敛": "可以减少epochs到100-150",
                    "收敛缓慢": "增加epochs到300-500",
                    "当前值": 200
                },
                "早停策略": {
                    "patience": "验证损失20轮不改善则停止",
                    "min_delta": "最小改善阈值0.001"
                }
            }
        }
        
        return strategies
    
    def mse_performance_analysis(self):
        """MSE性能分析"""
        analysis = {
            "不同MSE水平的含义": {
                "MSE < 0.01 (优秀)": {
                    "含义": "预测值与真实值非常接近",
                    "应用场景": "高精度要求的工业应用",
                    "达成难度": "需要优质数据和精心调优",
                    "模型要求": "复杂模型 + 充分训练"
                },
                "0.01 ≤ MSE < 0.05 (良好)": {
                    "含义": "预测精度较高，实用性强",
                    "应用场景": "大多数实际应用场景",
                    "达成难度": "中等，需要合理的模型设计",
                    "模型要求": "标准深度网络即可达到"
                },
                "0.05 ≤ MSE < 0.1 (可接受)": {
                    "含义": "基本满足预测需求",
                    "应用场景": "对精度要求不太严格的场景",
                    "达成难度": "较容易，基础模型即可",
                    "模型要求": "简单MLP即可达到"
                },
                "MSE ≥ 0.1 (需要改进)": {
                    "含义": "预测精度不足",
                    "可能原因": "数据质量差、模型欠拟合、超参数不当",
                    "改进方向": "检查数据、增加模型复杂度、调整超参数"
                }
            },
            
            "各组数据预期MSE": {
                "group_1 (7794样本)": {
                    "预期MSE": "0.02-0.05",
                    "原因": "样本数适中，训练相对充分"
                },
                "group_2 (18957样本)": {
                    "预期MSE": "0.01-0.03",
                    "原因": "样本数最多，训练最充分"
                },
                "group_3 (18957样本)": {
                    "预期MSE": "0.01-0.03",
                    "原因": "样本数最多，训练最充分"
                },
                "group_4 (4539样本)": {
                    "预期MSE": "0.03-0.08",
                    "原因": "样本数较少，可能存在过拟合风险"
                },
                "group_5 (4539样本)": {
                    "预期MSE": "0.03-0.08",
                    "原因": "样本数较少，可能存在过拟合风险"
                }
            }
        }
        
        return analysis
    
    def model_comparison_analysis(self):
        """模型比较分析"""
        comparison = {
            "MLP (多层感知机)": {
                "优势": [
                    "结构简单，训练快速",
                    "参数相对较少",
                    "适合中等规模数据",
                    "容易调试和理解"
                ],
                "劣势": [
                    "表达能力有限",
                    "容易过拟合",
                    "对超参数敏感"
                ],
                "预期性能": "MSE: 0.02-0.06",
                "适用场景": "基线模型，快速验证"
            },
            
            "ResNet (残差网络)": {
                "优势": [
                    "残差连接缓解梯度消失",
                    "可以训练更深的网络",
                    "表达能力强",
                    "训练稳定"
                ],
                "劣势": [
                    "参数较多",
                    "训练时间较长",
                    "可能过拟合小数据集"
                ],
                "预期性能": "MSE: 0.015-0.04",
                "适用场景": "复杂映射关系，大数据集"
            },
            
            "Transformer": {
                "优势": [
                    "注意力机制捕获复杂关系",
                    "并行计算效率高",
                    "表达能力最强"
                ],
                "劣势": [
                    "参数最多",
                    "训练时间最长",
                    "需要大量数据",
                    "容易过拟合"
                ],
                "预期性能": "MSE: 0.01-0.035",
                "适用场景": "复杂非线性关系，充足数据"
            },
            
            "Ensemble (集成模型)": {
                "优势": [
                    "结合多模型优势",
                    "预测更稳定",
                    "泛化能力强"
                ],
                "劣势": [
                    "计算开销大",
                    "模型复杂度高",
                    "部署困难"
                ],
                "预期性能": "MSE: 0.008-0.025",
                "适用场景": "追求最高精度"
            }
        }
        
        return comparison
    
    def generate_tuning_recommendations(self, current_mse_results=None):
        """生成调优建议"""
        recommendations = {
            "通用调优流程": [
                "1. 数据质量检查：检查异常值、缺失值、数据分布",
                "2. 基线模型：先用简单MLP建立基线",
                "3. 逐步优化：依次调整学习率、网络结构、正则化",
                "4. 模型选择：比较不同架构的性能",
                "5. 集成优化：使用最佳模型进行集成"
            ],
            
            "针对不同MSE水平的调优策略": {
                "MSE > 0.1": [
                    "检查数据预处理是否正确",
                    "增加模型复杂度（更多层、更大维度）",
                    "降低学习率，延长训练时间",
                    "检查是否存在数据泄露或标签错误"
                ],
                "0.05 < MSE ≤ 0.1": [
                    "尝试不同的激活函数（ReLU -> LeakyReLU -> GELU）",
                    "调整网络结构，增加残差连接",
                    "使用学习率调度器",
                    "增加数据增强或正则化"
                ],
                "0.02 < MSE ≤ 0.05": [
                    "微调超参数（学习率、dropout、权重衰减）",
                    "尝试不同的优化器（Adam -> AdamW -> RMSprop）",
                    "使用集成方法",
                    "增加模型深度或宽度"
                ],
                "MSE ≤ 0.02": [
                    "检查是否过拟合",
                    "增加正则化防止过拟合",
                    "使用交叉验证确保泛化性能",
                    "考虑模型压缩和加速"
                ]
            },
            
            "数据组特定建议": {
                "小数据组 (group_4, group_5)": [
                    "使用较小的网络防止过拟合",
                    "增大dropout率 (0.3-0.4)",
                    "使用数据增强技术",
                    "考虑迁移学习或预训练"
                ],
                "大数据组 (group_2, group_3)": [
                    "可以使用更复杂的模型",
                    "适当减小dropout率 (0.1-0.2)",
                    "增加训练轮数",
                    "使用更大的批次大小"
                ]
            }
        }
        
        return recommendations
    
    def create_hyperparameter_grid(self):
        """创建超参数搜索网格"""
        grid = {
            "learning_rate": [0.0001, 0.0003, 0.001, 0.003, 0.01],
            "batch_size": [32, 64, 128, 256],
            "hidden_dims": [
                [64, 128, 64],
                [128, 256, 128],
                [128, 256, 512, 256, 128],
                [256, 512, 1024, 512, 256]
            ],
            "dropout_rate": [0.1, 0.2, 0.3, 0.4],
            "weight_decay": [1e-6, 1e-5, 1e-4, 1e-3],
            "activation": ["relu", "leaky_relu", "gelu", "swish"]
        }
        
        return grid
    
    def save_analysis_report(self, filename="hyperparameter_analysis_report.txt"):
        """保存分析报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("深度学习回归预测项目 - 超参数分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            # MSE目标分析
            f.write("1. MSE目标值分析\n")
            f.write("-" * 30 + "\n")
            target_analysis = self.analyze_target_mse()
            for section, content in target_analysis.items():
                f.write(f"\n{section}:\n")
                if isinstance(content, dict):
                    for key, value in content.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {content}\n")
            
            # 超参数策略
            f.write("\n\n2. 超参数调整策略\n")
            f.write("-" * 30 + "\n")
            strategies = self.generate_hyperparameter_strategies()
            for strategy, details in strategies.items():
                f.write(f"\n{strategy}:\n")
                if isinstance(details, dict):
                    for key, value in details.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {details}\n")
            
            # 性能分析
            f.write("\n\n3. MSE性能分析\n")
            f.write("-" * 30 + "\n")
            performance = self.mse_performance_analysis()
            for section, content in performance.items():
                f.write(f"\n{section}:\n")
                if isinstance(content, dict):
                    for key, value in content.items():
                        f.write(f"  {key}:\n")
                        if isinstance(value, dict):
                            for k, v in value.items():
                                f.write(f"    {k}: {v}\n")
                        else:
                            f.write(f"    {value}\n")
            
            # 模型比较
            f.write("\n\n4. 模型比较分析\n")
            f.write("-" * 30 + "\n")
            comparison = self.model_comparison_analysis()
            for model, details in comparison.items():
                f.write(f"\n{model}:\n")
                for key, value in details.items():
                    f.write(f"  {key}: {value}\n")
            
            # 调优建议
            f.write("\n\n5. 调优建议\n")
            f.write("-" * 30 + "\n")
            recommendations = self.generate_tuning_recommendations()
            for section, content in recommendations.items():
                f.write(f"\n{section}:\n")
                if isinstance(content, list):
                    for item in content:
                        f.write(f"  - {item}\n")
                elif isinstance(content, dict):
                    for key, value in content.items():
                        f.write(f"  {key}:\n")
                        if isinstance(value, list):
                            for item in value:
                                f.write(f"    - {item}\n")
                        else:
                            f.write(f"    {value}\n")
        
        print(f"超参数分析报告已保存到: {filename}")

if __name__ == "__main__":
    # 生成超参数分析报告
    analyzer = HyperparameterAnalyzer()
    
    print("生成超参数分析报告...")
    analyzer.save_analysis_report()
    
    print("分析完成!")
    
    # 打印关键信息
    print("\n关键信息摘要:")
    print("=" * 40)
    
    target_analysis = analyzer.analyze_target_mse()
    print("MSE目标值:")
    for level, value in target_analysis["MSE目标值设定"].items():
        print(f"  {level}: {value}")
    
    print("\n预期各组MSE范围:")
    performance = analyzer.mse_performance_analysis()
    for group, info in performance["各组数据预期MSE"].items():
        print(f"  {group}: {info['预期MSE']}")
    
    print("\n模型预期性能:")
    comparison = analyzer.model_comparison_analysis()
    for model, info in comparison.items():
        print(f"  {model}: {info['预期性能']}")