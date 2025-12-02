"""
几何认知模型 - 核心实现
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class ReflexiveNeuron(nn.Module):
    """反身性神经元（认知元）"""
    
    def __init__(self, input_dim: int, output_dim: int, reflexive_dim: int):
        super().__init__()
        
        # 前馈部分
        self.linear = nn.Linear(input_dim, output_dim)
        
        # 反身性部分
        self.reflexive_layer = nn.Linear(output_dim, reflexive_dim)
        self.reflexive_activation = nn.Tanh()
        
        # 自指约束
        self.self_awareness = nn.Parameter(torch.randn(reflexive_dim))
        
    def forward(self, x: torch.Tensor, reflexive_load: float = 0.0) -> torch.Tensor:
        # 前馈计算
        output = torch.relu(self.linear(x))
        
        # 反身性处理
        if reflexive_load > 0:
            reflexive = self.reflexive_layer(output)
            reflexive = self.reflexive_activation(reflexive)
            
            # 自指约束影响
            output = output + reflexive_load * torch.matmul(
                reflexive, self.self_awareness
            ).unsqueeze(-1)
        
        return output


class GeometricCognitiveNetwork(nn.Module):
    """几何认知网络 - 反身性自适应网络的核心"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 reflexive_dims: List[int], output_dim: int = 1):
        super().__init__()
        
        # 网络层
        self.layers = nn.ModuleList()
        self.reflexive_layers = nn.ModuleList()
        
        # 构建前馈层
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        
        # 构建反身性层
        for dim in reflexive_dims:
            self.reflexive_layers.append(ReflexiveNeuron(dim, dim, dim // 2))
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # 认知参数
        self.cognitive_energy = nn.Parameter(torch.tensor(1.0))
        self.cognitive_cost = nn.Parameter(torch.tensor(0.0))
        self.reflexive_load = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播
        for layer in self.layers:
            x = torch.relu(layer(x))
        
        # 反身性处理
        if self.reflexive_load > 0:
            for r_layer in self.reflexive_layers:
                x = r_layer(x, self.reflexive_load.item())
        
        # 输出
        return self.output_layer(x)
    
    def compute_reflexivity_score(self) -> float:
        """计算反身性分数（曲率代理）"""
        # 基于权重变化率
        score = 0.0
        for param in self.parameters():
            if param.grad is not None:
                score += torch.norm(param.grad).item()
        
        return score
    
    def adapt_reflexive_load(self, loss: float, threshold: float = 0.1):
        """自适应调整反身性负荷"""
        if loss < threshold:
            # 任务简单，降低反身性
            self.reflexive_load.data *= 0.9
        else:
            # 任务困难，增加反身性
            self.reflexive_load.data *= 1.1
    
    def reset_state(self):
        """重置认知状态"""
        self.cognitive_energy.data = torch.tensor(1.0)
        self.cognitive_cost.data = torch.tensor(0.0)
        self.reflexive_load.data = torch.tensor(0.0)
