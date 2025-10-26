"""
配置文件 - 管理模型超参数和训练设置
"""
import torch

class Config:
    # 数据配置
    INPUT_DIM = 7
    OUTPUT_DIM = 26
    DATA_GROUPS = 5
    GROUP_SIZES = [7794, 18957, 18957, 4539, 4539]
    
    # 模型配置
    HIDDEN_DIMS = [128, 256, 512, 256, 128]  # MLP隐藏层维度
    DROPOUT_RATE = 0.2
    ACTIVATION = 'relu'
    
    # ResNet配置
    RESNET_BLOCKS = 4
    RESNET_HIDDEN_DIM = 256
    
    # 训练配置
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 200
    WEIGHT_DECAY = 1e-5
    PATIENCE = 20  # 早停耐心值
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据路径
    X_INPUT_PATH = 'x_input.csv'
    Y_OUTPUT_PATH = 'y_output.csv'
    
    # 模型保存路径
    MODEL_SAVE_DIR = 'models'
    LOG_DIR = 'logs'
    
    # 评估配置
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # 目标MSE值（根据数据规模和复杂度估算）
    TARGET_MSE = {
        'excellent': 0.01,    # 优秀水平
        'good': 0.05,         # 良好水平
        'acceptable': 0.1     # 可接受水平
    }