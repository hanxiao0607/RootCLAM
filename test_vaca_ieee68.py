import torch
import numpy as np
from models.vaca.vaca import VACA

# 示例配置
vaca_config = {
    "is_heterogeneous": True,
    "likelihood_x": ["gaussian"] * 136,  # 假设输入维度为 68 * 2 (Vm + Va)
    "deg": [2] * 136,
    "num_nodes": 136,
    "edge_dim": 0,
    "scaler": None,
    # "hidden_dim": 64,
    "z_dim": 16,
    # "num_layers": 2,
    # "dropout": 0.1,
}

# 初始化模型
model = VACA(**vaca_config)

# 生成随机输入
x = torch.randn(100, 136)  # 批大小为100，特征维度为136
u = torch.randn(100, 10)   # 假设干预向量为10维

# 前向传播
output = model(x, u)
print("Forward pass output keys:", output.keys())
print("Sample reconstructed x[0]:", output['xhat'][0])