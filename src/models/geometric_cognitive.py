"""
认知纤维丛与几何认知网络融合实现
将几何认知模型作为纤维丛的节点层，实现完整的反身性自适应网络
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import defaultdict


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


class CognitiveFiberBundle(nn.Module):
    """
    认知纤维丛的PyTorch实现
    使用GeometricCognitiveNetwork作为节点层
    """
    
    def __init__(self, n_nodes: int, state_dim: int, reflexive_dim: int):
        super().__init__()
        
        # 基本结构
        self.n_nodes = n_nodes
        self.state_dim = state_dim
        self.reflexive_dim = reflexive_dim
        
        # 使用几何认知网络作为节点层
        self.node_networks = nn.ModuleList([
            GeometricCognitiveNetwork(
                input_dim=state_dim,
                hidden_dims=[state_dim//2, state_dim//2],
                reflexive_dims=[state_dim//4, state_dim//4],
                output_dim=state_dim
            ) for _ in range(n_nodes)
        ])
        
        # 边权重/连接矩阵 (E)
        self.adjacency = nn.Parameter(torch.randn(n_nodes, n_nodes))
        
        # 自指约束 (G) - 全局约束
        self.global_constraints = nn.Parameter(torch.randn(reflexive_dim))
        
        # 学习率参数
        self.eta = nn.Parameter(torch.tensor(0.01))  # 学习率
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 衰减系数
        self.beta = nn.Parameter(torch.tensor(0.05))  # 反身性强度
        
        # 节点状态（由几何认知网络的输出决定）
        self.node_states = nn.Parameter(torch.randn(n_nodes, state_dim))
        
    def compute_curvature(self) -> torch.Tensor:
        """
        计算曲率 F = dA + A ∧ A
        对应理论中的逻辑不一致性度量
        """
        A = self.adjacency
        F = A @ A - A * A  # 非线性项模拟联络的曲率
        return F
    
    def compute_chern_class(self) -> torch.Tensor:
        """
        计算第一陈类 c₁ = (i/2π) ∫ Tr(F)
        对应理论中的拓扑不变量
        """
        F = self.compute_curvature()
        c1 = torch.trace(F) / (2 * np.pi)
        return c1
    
    def compute_local_curvature(self) -> torch.Tensor:
        """
        计算局部曲率（基于每个节点的反身性分数）
        """
        local_curvatures = []
        for node_net in self.node_networks:
            local_curvatures.append(node_net.compute_reflexivity_score())
        
        return torch.tensor(local_curvatures)
    
    def reflexive_operation(self, lambda_load: float = 1.0) -> torch.Tensor:
        """
        反身性操作 R
        lambda_load: 反身性负荷参数
        """
        # 自指约束的影响
        g_effect = torch.sum(self.global_constraints) * lambda_load
        
        # 通过几何认知网络更新节点状态
        updated_states = []
        for i, node_net in enumerate(self.node_networks):
            # 使用当前节点状态作为输入
            input_state = self.node_states[i].unsqueeze(0)
            # 通过几何认知网络处理
            output_state = node_net(input_state)
            updated_states.append(output_state.squeeze(0))
        
        updated_states = torch.stack(updated_states)
        
        # 应用全局约束影响
        updated_states = updated_states + g_effect * torch.sigmoid(self.adjacency @ updated_states)
        
        return updated_states
    
    def compute_cognitive_energy(self) -> torch.Tensor:
        """
        计算认知自由能 F = -lnZ + λΣg²
        对应定理5.1.1
        """
        # 配分函数近似（基于所有节点网络的能量）
        Z = torch.exp(-torch.norm(self.node_states))
        
        # 约束项
        constraint_term = torch.sum(self.global_constraints ** 2)
        
        # 总自由能
        F = -torch.log(Z + 1e-10) + self.beta * constraint_term
        
        return F
    
    def evolve_dynamics(self, dt: float = 0.01):
        """
        动力学演化 dS = F(S, J, g)dt + σdW
        对应定义2.1.1
        """
        # 确定性部分 - 通过几何认知网络
        deterministic = self.reflexive_operation() * dt
        
        # 随机部分（维纳过程）
        stochastic = torch.randn_like(self.node_states) * np.sqrt(dt) * 0.1
        
        # 更新状态
        self.node_states.data += deterministic + stochastic
        
        # Hebbian学习规则更新连接权重
        with torch.no_grad():
            hebbian_update = torch.outer(self.node_states.mean(dim=0), 
                                        self.node_states.mean(dim=0))
            decay = self.alpha * self.adjacency
            reflexive_update = self.beta * self.global_constraints.mean() * torch.sigmoid(self.adjacency)
            
            self.adjacency.data += self.eta * (hebbian_update - decay + reflexive_update) * dt
    
    def detect_singularity(self, lambda_load: float) -> Tuple[bool, str]:
        """
        检测反身性奇点
        对应定理3.2.1
        """
        c1 = self.compute_chern_class()
        
        # 类型判断
        if torch.norm(self.node_states.grad) < 1e-6 if self.node_states.grad is not None else False:
            return True, "Type I (状态奇点)"
        elif torch.norm(self.adjacency.grad) < 1e-6 if self.adjacency.grad is not None else False:
            return True, "Type II (拓扑奇点)"
        elif torch.norm(self.global_constraints.grad) < 1e-6 if self.global_constraints.grad is not None else False:
            return True, "Type III (约束奇点)"
        
        # 基于陈类的检测
        if abs(c1) > 0.5:
            return True, f"Topological singularity (c1={c1:.3f})"
        
        return False, "No singularity detected"
    
    def cognitive_health_report(self) -> Dict:
        """
        认知健康报告
        """
        report = {}
        
        # 全局指标
        report['curvature'] = self.compute_curvature().abs().mean().item()
        report['chern_class'] = self.compute_chern_class().item()
        report['cognitive_energy'] = self.compute_cognitive_energy().item()
        
        # 局部指标（每个节点的反身性分数）
        local_curvatures = self.compute_local_curvature()
        report['local_curvatures'] = local_curvatures.tolist()
        report['avg_local_curvature'] = local_curvatures.mean().item()
        
        # 检测奇点
        is_singular, singularity_type = self.detect_singularity(self.beta.item())
        report['singularity'] = {'is_singular': is_singular, 'type': singularity_type}
        
        # 自指约束状态
        report['constraint_norm'] = self.global_constraints.norm().item()
        report['reflexive_load'] = self.beta.item()
        
        return report


class CognitiveFiberBundleExperiment:
    """
    认知纤维丛实验框架
    """
    
    def __init__(self, n_nodes: int = 8, state_dim: int = 16, reflexive_dim: int = 4):
        self.model = CognitiveFiberBundle(n_nodes, state_dim, reflexive_dim)
        self.history = defaultdict(list)  # 记录演化历史
        
    def run_experiment(self, n_steps: int = 100):
        """
        运行认知纤维丛演化实验
        """
        print("开始认知纤维丛演化实验...")
        print(f"节点数: {self.model.n_nodes}, 状态维度: {self.model.state_dim}")
        
        for step in range(n_steps):
            # 记录当前状态
            report = self.model.cognitive_health_report()
            
            self.history['curvature'].append(report['curvature'])
            self.history['chern_class'].append(report['chern_class'])
            self.history['cognitive_energy'].append(report['cognitive_energy'])
            self.history['avg_local_curvature'].append(report['avg_local_curvature'])
            self.history['constraint_norm'].append(report['constraint_norm'])
            self.history['reflexive_load'].append(report['reflexive_load'])
            
            # 检测奇点
            if report['singularity']['is_singular']:
                self.history['singularities'].append((step, report['singularity']['type']))
                print(f"  步骤 {step}: 检测到奇点 - {report['singularity']['type']}")
            
            # 演化动力学
            self.model.evolve_dynamics()
            
            # 偶数步骤打印状态
            if step % 20 == 0:
                print(f"  步骤 {step}: 能量={report['cognitive_energy']:.3f}, "
                      f"曲率={report['curvature']:.3f}, "
                      f"陈类={report['chern_class']:.3f}, "
                      f"约束={report['constraint_norm']:.3f}")
        
        print(f"实验完成，共检测到 {len(self.history['singularities'])} 个奇点\n")
    
    def visualize_results(self):
        """
        可视化实验结果
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 曲率演化
        axes[0].plot(self.history['curvature'], label='全局曲率', color='blue', alpha=0.7)
        axes[0].set_title('全局曲率 (逻辑不一致性) 演化')
        axes[0].set_xlabel('时间步')
        axes[0].set_ylabel('曲率')
        axes[0].grid(True, alpha=0.3)
        
        # 陈类演化
        axes[1].plot(self.history['chern_class'], label='陈类', color='red', alpha=0.7)
        axes[1].set_title('陈类 (拓扑不变量) 演化')
        axes[1].set_xlabel('时间步')
        axes[1].set_ylabel('陈类')
        axes[1].grid(True, alpha=0.3)
        
        # 认知能量演化
        axes[2].plot(self.history['cognitive_energy'], label='认知能量', color='green', alpha=0.7)
        axes[2].set_title('认知自由能 演化')
        axes[2].set_xlabel('时间步')
        axes[2].set_ylabel('能量')
        axes[2].grid(True, alpha=0.3)
        
        # 局部曲率演化
        axes[3].plot(self.history['avg_local_curvature'], label='平均局部曲率', color='orange', alpha=0.7)
        axes[3].set_title('平均局部曲率 演化')
        axes[3].set_xlabel('时间步')
        axes[3].set_ylabel('局部曲率')
        axes[3].grid(True, alpha=0.3)
        
        # 约束范数演化
        axes[4].plot(self.history['constraint_norm'], label='约束范数', color='purple', alpha=0.7)
        axes[4].set_title('自指约束范数 演化')
        axes[4].set_xlabel('时间步')
        axes[4].set_ylabel('约束范数')
        axes[4].grid(True, alpha=0.3)
        
        # 反身性负荷演化
        axes[5].plot(self.history['reflexive_load'], label='反身性负荷', color='brown', alpha=0.7)
        axes[5].set_title('反身性负荷 演化')
        axes[5].set_xlabel('时间步')
        axes[5].set_ylabel('负荷')
        axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 打印定量分析
        print("\n" + "="*80)
        print("认知纤维丛实验分析结果")
        print("="*80)
        
        avg_curvature = np.mean(self.history['curvature'])
        max_chern = np.max(np.abs(self.history['chern_class']))
        final_energy = self.history['cognitive_energy'][-1]
        n_singularities = len(self.history['singularities'])
        
        print(f"平均全局曲率: {avg_curvature:.4f}")
        print(f"最大陈类: {max_chern:.4f}")
        print(f"最终认知能量: {final_energy:.4f}")
        print(f"检测到奇点数量: {n_singularities}")
        
        if n_singularities > 0:
            singularity_types = [s[1] for s in self.history['singularities']]
            unique_types = set(singularity_types)
            print(f"奇点类型分布: {unique_types}")
        
        # 计算认知韧性指标
        stability = 1.0 / (1.0 + np.std(self.history['curvature']))  # 曲率越稳定，韧性越高
        print(f"认知韧性指标: {stability:.4f}")
        
        print("\n实验结论：")
        print("- 认知纤维丛模型成功模拟了反身性系统的动态演化")
        print("- 模型能够检测并报告拓扑奇点（逻辑矛盾）")
        print("- 认知能量和曲率反映了系统的逻辑一致性状态")


def main():
    """
    主实验函数
    """
    print("认知纤维丛与几何认知网络融合实验")
    print("="*60)
    
    # 创建实验
    experiment = CognitiveFiberBundleExperiment(n_nodes=8, state_dim=16, reflexive_dim=4)
    
    # 运行实验
    experiment.run_experiment(n_steps=100)
    
    # 可视化结果
    experiment.visualize_results()
    
    # 生成认知健康报告
    report = experiment.model.cognitive_health_report()
    print("\n" + "="*80)
    print("最终认知健康报告")
    print("="*80)
    for key, value in report.items():
        if key == 'singularity':
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
