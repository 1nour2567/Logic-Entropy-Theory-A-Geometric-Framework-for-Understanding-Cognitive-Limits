"""
深度学习模型实现
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class StandardDeepLearningModel:
    """标准深度学习模型 - 用于对比实验"""
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.criterion = nn.MSELoss()
        self.loss_history = []

    def train_addition_task(self, epochs=800):
        """训练加法任务"""
        np.random.seed(42)
        X = np.random.randint(0, 1000, (5000, 2)).astype(np.float32) / 1000.0
        y = (X[:, 0] + X[:, 1]).reshape(-1, 1)
        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            pred = self.model(X_tensor)
            loss = self.criterion(pred, y_tensor)
            loss.backward()
            self.optimizer.step()
            self.loss_history.append(loss.item())
            
        return self.loss_history
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        with torch.no_grad():
            X_tensor = torch.tensor(X_test.astype(np.float32))
            y_tensor = torch.tensor(y_test.astype(np.float32))
            pred = self.model(X_tensor)
            loss = self.criterion(pred, y_tensor)
        return loss.item()
