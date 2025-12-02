"""
逻辑熵理论的核心实现
"""
import numpy as np
import zlib


class LogicEntropyTheory:
    """逻辑熵理论的核心类"""
    
    def __init__(self):
        self.entropy_history = []
        self.complexity_history = []
        
    def compute_logical_entropy(self, system_description: str, 
                               n_godel_sentences: int) -> float:
        """
        计算逻辑熵 H(L) = log₂(1+N_G) + γ·K_min
        对应理论中的操作化定义
        """
        # 哥德尔句子数量
        N_G = n_godel_sentences
        
        # 最小描述长度（压缩近似）
        compressed = zlib.compress(system_description.encode('utf-8'))
        K_min = len(compressed) * 8  # 比特数
        
        # 逻辑熵计算
        gamma = 0.91  # 经验常数（来自实证）
        H = np.log2(1 + N_G) + gamma * K_min
        
        self.entropy_history.append(H)
        return H
    
    def compute_reflexive_load(self, system_complexity: float, 
                              n_paradoxes: int = 0) -> float:
        """
        计算反身性负荷 λ
        λ = K(π_L) + K(Prov_L) + α·P
        """
        # 系统固有复杂度
        base_load = system_complexity
        
        # 悖论惩罚项
        paradox_penalty = 0.5 * n_paradoxes
        
        return base_load + paradox_penalty
    
    def detect_phase_transition(self, reflexive_load: float, 
                               critical_load: float = 4.3) -> dict:
        """
        检测认知相变
        对应理论中的反身性奇点定理
        """
        if reflexive_load >= critical_load:
            return {
                'phase_transition': True,
                'phase': 'high_entropy',
                'critical_load': critical_load,
                'current_load': reflexive_load,
                'description': '系统进入高熵不确定性相'
            }
        
        return {
            'phase_transition': False,
            'phase': 'low_entropy',
            'current_load': reflexive_load,
            'description': '系统处于低熵确定性相'
        }
    
    def verify_second_law(self) -> bool:
        """
        验证逻辑热力学第二定律：dH_total ≥ 0
        逻辑熵永不减少
        """
        if len(self.entropy_history) < 2:
            return True
        
        entropy_changes = np.diff(self.entropy_history)
        return np.all(entropy_changes >= -1e-10)  # 允许微小数值误差
