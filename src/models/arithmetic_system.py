"""
算术系统模型 - 实现Q系统和PA系统
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class MinimalArithmeticExtension:
    """最小算术扩展模型 - 包含Q系统和PA系统公理"""
    def __init__(self):
        # Robinson Q 系统公理（无归纳法）
        self.axioms_Q = {
            "加法恒等元": "∀x(x + 0 = x)",
            "加法后继": "∀x∀y(x + S(y) = S(x + y))",
            "乘法恒等元": "∀x(x × 0 = 0)",
            "乘法后继": "∀x∀y(x × S(y) = (x × y) + x)",
            "后继注入": "∀x∀y(S(x) = S(y) → x = y)",
            "非零后继": "∀x ¬(S(x) = 0)"
        }
        
        # Peano 算术扩展（加入归纳法）
        self.axioms_PA = self.axioms_Q.copy()
        self.axioms_PA["数学归纳法"] = "∀P(P(0) ∧ ∀n(P(n) → P(S(n))) → ∀n P(n))"
        
        self.cognitive_energy = 1.0
        self.cognitive_cost = 0.0
        self.energy_history = []
        self.cost_history = []

    def prove_in_Q(self, theorem: str, steps: int = 3):
        """在Q系统中证明（无归纳法）"""
        success = np.random.random() < 0.95
        cost = steps * np.random.uniform(0.08, 0.15)
        return success, cost

    def prove_in_PA(self, theorem: str, steps: int = 2):
        """在PA系统中证明（有归纳法）"""
        success = np.random.random() < 0.98
        cost = steps * np.random.uniform(0.05, 0.12)
        return success, cost

    def compute_extension_cost(self, axioms_from: Dict, axioms_to: Dict):
        """计算从Q扩展到PA的认知成本"""
        added_axioms = len(axioms_to) - len(axioms_from)
        complexity = 1.5 ** added_axioms  # 指数级增长
        return complexity * np.random.uniform(0.8, 1.2)

    def compute_energy_dissipation(self, axioms: Dict, depth: int):
        """能量耗散：公理越多、推理越深，耗散越大"""
        base_dissipation = 0.05
        return base_dissipation * len(axioms) * np.log1p(depth)


class EnhancedArithmeticSimulation(MinimalArithmeticExtension):
    """增强的算术扩展模拟 - 添加统计检验和对比"""
   
    def __init__(self):
        super().__init__()
        self.trial_results = []
        self.statistical_significance = {}
       
    def run_multiple_trials(self, n_trials=100, random_seed=42):
        """运行多次试验并计算统计显著性"""
        np.random.seed(random_seed)
        print(f"运行 {n_trials} 次模拟试验...")
       
        for trial in range(n_trials):
            self.cognitive_energy = 1.0 + np.random.normal(0, 0.03)
            self.cognitive_cost = 0.0
            self.energy_history = []
            self.cost_history = []
           
            self.simulate_extension_process_single()
           
            trial_result = {
                'trial': trial,
                'final_energy': self.cognitive_energy,
                'total_cost': self.cognitive_cost,
                'q_phase_cost': self.cost_history[0] if len(self.cost_history) > 0 else 0,
                'pa_phase_cost': self.cognitive_cost - (self.cost_history[0] if len(self.cost_history) > 0 else 0),
                'extension_cost': self.compute_extension_cost(self.axioms_Q, self.axioms_PA),
                'energy_drop': 1.0 - self.cognitive_energy
            }
            self.trial_results.append(trial_result)
       
        self.compute_statistical_significance()
        return self.trial_results
   
    def simulate_extension_process_single(self):
        """单次扩展过程模拟"""
        # Q阶段
        theorems_Q = ["S(0) ≠ 0", "S(S(0)) ≠ S(0)", "∀x(x + 0 = x)"]
        q_cost_accum = 0
        for theorem in theorems_Q:
            success, cost = self.prove_in_Q(theorem, steps=3 + np.random.randint(0, 3))
            self.cognitive_cost += cost
            q_cost_accum += cost
            self.cognitive_energy -= self.compute_energy_dissipation(self.axioms_Q, 3)
       
        self.record_state_single("Q")
       
        # 扩展阶段（关键！）
        extension_cost = self.compute_extension_cost(self.axioms_Q, self.axioms_PA)
        self.cognitive_cost += extension_cost
        self.cognitive_energy -= extension_cost * 0.5
       
        # PA阶段
        theorems_PA = ["∀x(x = x)", "∀x∀y(x = y → y = x)", "∀n(n + 0 = n ∧ n × 0 = 0)"]
        for theorem in theorems_PA:
            success, cost = self.prove_in_PA(theorem, steps=2 + np.random.randint(0, 2))
            self.cognitive_cost += cost
            self.cognitive_energy -= self.compute_energy_dissipation(self.axioms_PA, 2)
       
        self.record_state_single("PA")
   
    def record_state_single(self, stage: str):
        """记录单个阶段的状态"""
        self.energy_history.append(self.cognitive_energy)
        self.cost_history.append(self.cognitive_cost)
   
    def compute_statistical_significance(self):
        """计算统计显著性"""
        energies = [r['final_energy'] for r in self.trial_results]
        costs = [r['total_cost'] for r in self.trial_results]
        q_costs = [r['q_phase_cost'] for r in self.trial_results]
        pa_costs = [r['pa_phase_cost'] for r in self.trial_results]
       
        self.statistical_significance = {
            'energy_mean': np.mean(energies),
            'energy_std': np.std(energies),
            'cost_mean': np.mean(costs),
            'cost_std': np.std(costs),
            'q_cost_mean': np.mean(q_costs),
            'pa_cost_mean': np.mean(pa_costs),
            'n_trials': len(self.trial_results)
        }
       
        if len(q_costs) > 1 and len(pa_costs) > 1:
            t_stat, p_value = stats.ttest_rel(q_costs, pa_costs)
            self.statistical_significance['t_statistic_q_vs_pa'] = t_stat
            self.statistical_significance['p_value_q_vs_pa'] = p_value
       
        energy_drops = [1.0 - e for e in energies]
        corr_coef, p_corr = stats.pearsonr(energy_drops, costs)
        self.statistical_significance['correlation_energy_cost'] = corr_coef
        self.statistical_significance['correlation_p_value'] = p_corr
