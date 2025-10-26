"""
神经网络模型定义 - 包含MLP、ResNet和其他推荐架构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class MLP(nn.Module):
    """多层感知机 (Multi-Layer Perceptron)"""
    
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout_rate=0.2, activation='relu'):
        super(MLP, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 512, 256, 128]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # 激活函数选择
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # 残差连接
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """残差网络 (Residual Network)"""
    
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_blocks=4, dropout_rate=0.2):
        super(ResNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入投影
        x = self.input_projection(x)
        
        # 通过残差块
        for block in self.residual_blocks:
            x = block(x)
        
        # 输出
        x = self.output_layer(x)
        return x

class AttentionBlock(nn.Module):
    """注意力块"""
    
    def __init__(self, dim, num_heads=8, dropout_rate=0.2):
        super(AttentionBlock, self).__init__()
        
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(dim)
        
        self.output_projection = nn.Linear(dim, dim)
    
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        
        # 计算Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        
        # 输出投影和残差连接
        output = self.output_projection(attended)
        output = self.norm(output + x)
        
        return output

class TransformerMLP(nn.Module):
    """基于Transformer的MLP"""
    
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4, num_heads=8, dropout_rate=0.2):
        super(TransformerMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # 输入嵌入
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads, dropout_rate) for _ in range(num_layers)
        ])
        
        # 前馈网络
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入嵌入 (batch_size, input_dim) -> (batch_size, 1, hidden_dim)
        x = self.input_embedding(x).unsqueeze(1)
        
        # 通过Transformer层
        for attention_layer, ffn_layer in zip(self.transformer_layers, self.ffn):
            # 注意力
            x = attention_layer(x)
            # 前馈网络
            x = ffn_layer(x.squeeze(1)).unsqueeze(1) + x
        
        # 输出 (batch_size, 1, hidden_dim) -> (batch_size, output_dim)
        x = x.squeeze(1)
        x = self.output_layer(x)
        
        return x

class EnsembleModel(nn.Module):
    """集成模型 - 结合多个模型的预测"""
    
    def __init__(self, models, weights=None):
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        else:
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
    
    def forward(self, x):
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # 加权平均
        predictions = torch.stack(predictions, dim=0)  # (num_models, batch_size, output_dim)
        weights = F.softmax(self.weights, dim=0).view(-1, 1, 1)  # (num_models, 1, 1)
        
        ensemble_pred = torch.sum(predictions * weights, dim=0)
        
        return ensemble_pred

class ModelFactory:
    """模型工厂 - 用于创建不同类型的模型"""
    
    @staticmethod
    def create_model(model_type, input_dim, output_dim, config=None):
        """创建指定类型的模型"""
        
        if config is None:
            config = Config()
        
        if model_type == 'mlp':
            return MLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=config.HIDDEN_DIMS,
                dropout_rate=config.DROPOUT_RATE,
                activation=config.ACTIVATION
            )
        
        elif model_type == 'resnet':
            return ResNet(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=config.RESNET_HIDDEN_DIM,
                num_blocks=config.RESNET_BLOCKS,
                dropout_rate=config.DROPOUT_RATE
            )
        
        elif model_type == 'transformer':
            return TransformerMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=config.RESNET_HIDDEN_DIM,
                num_layers=4,
                num_heads=8,
                dropout_rate=config.DROPOUT_RATE
            )
        
        elif model_type == 'ensemble':
            # 创建多个基础模型
            models = [
                ModelFactory.create_model('mlp', input_dim, output_dim, config),
                ModelFactory.create_model('resnet', input_dim, output_dim, config),
                ModelFactory.create_model('transformer', input_dim, output_dim, config)
            ]
            return EnsembleModel(models)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_info(model):
        """获取模型信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
            'model_type': type(model).__name__
        }

if __name__ == "__main__":
    # 测试模型创建
    config = Config()
    
    print("测试模型创建...")
    
    model_types = ['mlp', 'resnet', 'transformer', 'ensemble']
    
    for model_type in model_types:
        print(f"\n创建 {model_type} 模型:")
        model = ModelFactory.create_model(
            model_type, config.INPUT_DIM, config.OUTPUT_DIM, config
        )
        
        info = ModelFactory.get_model_info(model)
        print(f"参数数量: {info['total_parameters']:,}")
        print(f"模型大小: {info['model_size_mb']:.2f} MB")
        
        # 测试前向传播
        x = torch.randn(32, config.INPUT_DIM)
        with torch.no_grad():
            y = model(x)
            print(f"输入形状: {x.shape}")
            print(f"输出形状: {y.shape}")
    
    print("\n模型测试完成!")