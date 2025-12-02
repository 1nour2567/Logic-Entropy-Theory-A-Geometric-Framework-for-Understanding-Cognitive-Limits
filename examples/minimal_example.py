"""
最小示例：快速体验几何认知模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 导入几何认知模型
from src.models.geometric_cognitive import GeometricCognitiveNetwork

# 创建模型
model = GeometricCognitiveNetwork(
    input_dim=10,
    hidden_dims=[32, 64],
    reflexive_dims=[16, 32],
    output_dim=1
)

# 创建随机数据
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# 训练配置
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 训练循环
losses = []
for epoch in range(100):
    # 前向传播
    predictions = model(X)
    loss = criterion(predictions, y)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 记录损失
    losses.append(loss.item())
    
    # 自适应调整反身性负荷
    model.adapt_reflexive_load(loss.item())
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Reflexive Load = {model.reflexive_load.item():.3f}")

# 可视化训练过程
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Geometric Cognitive Network Training')
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n训练完成！最终损失: {losses[-1]:.4f}")
print(f"最终反身性负荷: {model.reflexive_load.item():.3f}")
print(f"认知能量: {model.cognitive_energy.item():.3f}")
print(f"认知成本: {model.cognitive_cost.item():.3f}")
