"""
几何模型测试
"""
import unittest
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.geometric_cognitive import GeometricCognitiveNetwork, ReflexiveNeuron


class TestGeometricModel(unittest.TestCase):
    
    def setUp(self):
        self.model = GeometricCognitiveNetwork(
            input_dim=10,
            hidden_dims=[32, 64],
            reflexive_dims=[16, 32],
            output_dim=1
        )
        
    def test_model_creation(self):
        """测试模型是否能正确创建"""
        self.assertIsNotNone(self.model)
        self.assertEqual(len(self.model.layers), 2)
        self.assertEqual(len(self.model.reflexive_layers), 2)
    
    def test_forward_pass(self):
        """测试前向传播"""
        X = torch.randn(5, 10)
        output = self.model(X)
        
        self.assertEqual(output.shape, (5, 1))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_reflexive_load(self):
        """测试反身性负荷调整"""
        initial_load = self.model.reflexive_load.item()
        
        # 高损失应该增加反身性
        self.model.adapt_reflexive_load(0.5)
        self.assertGreater(self.model.reflexive_load.item(), initial_load)
        
        # 低损失应该减少反身性
        self.model.reflexive_load.data = torch.tensor(1.0)
        self.model.adapt_reflexive_load(0.05)
        self.assertLess(self.model.reflexive_load.item(), 1.0)
    
    def test_reset_state(self):
        """测试状态重置"""
        self.model.cognitive_energy.data = torch.tensor(0.5)
        self.model.cognitive_cost.data = torch.tensor(2.0)
        self.model.reflexive_load.data = torch.tensor(0.8)
        
        self.model.reset_state()
        
        self.assertAlmostEqual(self.model.cognitive_energy.item(), 1.0, places=5)
        self.assertAlmostEqual(self.model.cognitive_cost.item(), 0.0, places=5)
        self.assertAlmostEqual(self.model.reflexive_load.item(), 0.0, places=5)
    
    def test_reflexivity_score(self):
        """测试反身性分数计算"""
        # 需要先进行一次反向传播
        X = torch.randn(5, 10, requires_grad=True)
        y = torch.randn(5, 1)
        
        output = self.model(X)
        loss = torch.mean((output - y) ** 2)
        loss.backward()
        
        score = self.model.compute_reflexivity_score()
        self.assertGreaterEqual(score, 0.0)


class TestReflexiveNeuron(unittest.TestCase):
    
    def setUp(self):
        self.neuron = ReflexiveNeuron(
            input_dim=10,
            output_dim=5,
            reflexive_dim=3
        )
    
    def test_forward_without_reflexive(self):
        """测试无反身性的前向传播"""
        X = torch.randn(3, 10)
        output = self.neuron(X, reflexive_load=0.0)
        
        self.assertEqual(output.shape, (3, 5))
    
    def test_forward_with_reflexive(self):
        """测试有反身性的前向传播"""
        X = torch.randn(3, 10)
        output = self.neuron(X, reflexive_load=0.5)
        
        self.assertEqual(output.shape, (3, 5))
        # 反身性应该增加一些变化
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
