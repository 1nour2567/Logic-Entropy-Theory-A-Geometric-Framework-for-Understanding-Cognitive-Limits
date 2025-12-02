"""
集成Chaitin逻辑熵、随机证明与相变分析的完整实验
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import random

from src.core.chaitin_entropy import LogicalEntropyExtension, StochasticProofSystem
from src.core.phase_transition import CognitivePhaseTransitionAnalyzer
from src.models.arithmetic_system import MinimalArithmeticExtension


class IntegratedChaitinExperiment:
    """集成Chaitin逻辑熵的完整实验"""
    
    def __init__(self, n_trials=100):
        self.n_trials = n_trials
        self.entropy_calculator = LogicalEntropyExtension()
        self.proof_system = StochasticProofSystem(base_success_rate=0.75)
        self.phase_analyzer = CognitivePhaseTransitionAnalyzer()
        
        # 创建算术系统
        self.arithmetic_sim = MinimalArithmeticExtension()
        self.axioms_Q = self.arithmetic_sim.axioms_Q
        self.axioms_PA = self.arithmetic_sim.axioms_PA
    
    def run_comprehensive_analysis(self) -> Dict:
        """运行综合分析"""
        print("集成实验：Chaitin逻辑熵 + 随机证明 + 认知相变")
        print("="*70)
        
        results = {}
        
        # 1. 逻辑熵分析
        print("\n1. 逻辑熵与不可完备性分析")
        results['logical_entropy'] = self._analyze_logical_entropy()
        
        # 2. 随机证明实验
        print("\n2. 随机证明系统实验")
        results['stochastic_proofs'] = self._run_stochastic_proof_experiment()
        
        # 3. 相变检测
        print("\n3. 认知相变检测")
        results['phase_transitions'] = self._detect_phase_transitions(
            results['stochastic_proofs']['energy_history'],
            results['stochastic_proofs']['cost_history']
        )
        
        # 4. 理论验证
        print("\n4. 理论验证")
        results['theory_validation'] = self._validate_theoretical_predictions(results)
        
        return results
    
    def _analyze_logical_entropy(self) -> Dict:
        """分析逻辑熵"""
        # 计算两个系统的逻辑熵
        entropy_Q_chaitin = self.entropy_calculator.compute_logical_entropy(
            self.axioms_Q, "chaitin"
        )
        entropy_PA_chaitin = self.entropy_calculator.compute_logical_entropy(
            self.axioms_PA, "chaitin"
        )
        
        # 计算Chaitin Ω
        omega_Q = self.entropy_calculator.approximate_chaitin_constant(
            self.axioms_Q, "theoretical"
        )
        omega_PA = self.entropy_calculator.approximate_chaitin_constant(
            self.axioms_PA, "theoretical"
        )
        
        # 不可完备性估计
        incompleteness_Q = self.entropy_calculator.estimate_incompleteness_degree(self.axioms_Q)
        incompleteness_PA = self.entropy_calculator.estimate_incompleteness_degree(self.axioms_PA)
        
        print(f"Q系统 - Ω: {omega_Q:.6f}, 逻辑熵: {entropy_Q_chaitin:.4f}")
        print(f"PA系统 - Ω: {omega_PA:.6f}, 逻辑熵: {entropy_PA_chaitin:.4f}")
        print(f"熵增: {entropy_PA_chaitin - entropy_Q_chaitin:.4f}")
        print(f"Ω减少: {omega_Q - omega_PA:.6f} (更强的系统有更小的Ω)")
        print(f"Q系统哥德尔语句估计: {incompleteness_Q['estimated_godel_sentences']}")
        print(f"PA系统哥德尔语句估计: {incompleteness_PA['estimated_godel_sentences']}")
        
        return {
            'entropy_Q': entropy_Q_chaitin,
            'entropy_PA': entropy_PA_chaitin,
            'omega_Q': omega_Q,
            'omega_PA': omega_PA,
            'entropy_increase': entropy_PA_chaitin - entropy_Q_chaitin,
            'omega_decrease': omega_Q - omega_PA,
            'incompleteness_Q': incompleteness_Q,
            'incompleteness_PA': incompleteness_PA
        }
    
    def _run_stochastic_proof_experiment(self) -> Dict:
        """运行随机证明实验"""
        theorems = [
            "∀x(x = x)",                         # 自反性
            "∀x∀y(x = y → y = x)",               # 对称性
            "∀x∀y∀z((x = y ∧ y = z) → x = z)",   # 传递性
            "∀x(x + 0 = x)",                     # 加法单位元
            "∀x∀y(x + y = y + x)",               # 加法交换律
            "∀x∀y∀z((x + y) + z = x + (y + z))", # 加法结合律
            "∀x(x × 0 = 0)",                     # 乘法零元
            "∀x∀y(x × y = y × x)",               # 乘法交换律
            "∀n(P(n) → P(n+1)) → (P(0) → ∀n P(n))",  # 归纳法原理
        ]
        
        energy_history = [1.0]
        cost_history = [0.0]
        current_energy = 1.0
        current_cost = 0.0
        
        # 重置证明系统
        self.proof_system.reset_cognitive_state()
        
        print(f"运行 {self.n_trials} 次随机证明尝试...")
        
        for trial in range(self.n_trials):
            # 随机选择定理和系统
            theorem = random.choice(theorems)
            system = random.choice([self.axioms_Q, self.axioms_PA])
            system_name = "Q" if system is self.axioms_Q else "PA"
            
            # 随机步骤数（复杂度越高，步骤越多）
            steps = random.randint(3, 15)
            
            # 难度估计（基于定理长度和符号数）
            difficulty = len(theorem) * 0.02 + theorem.count('∀') * 0.1
            
            # 执行证明
            success, cost = self.proof_system.stochastic_prove(
                theorem, system, steps, difficulty
            )
            
            # 更新状态
            current_cost += cost
            current_energy = max(0.0, current_energy - cost * 0.25)
            
            energy_history.append(current_energy)
            cost_history.append(current_cost)
            
            # 进度显示
            if trial % 20 == 0 and trial > 0:
                success_rate = self.proof_system.analyze_proof_patterns()['overall_success_rate']
                print(f"  进度: {trial}/{self.n_trials}, 成功率: {success_rate:.1%}, 认知储备: {self.proof_system.cognitive_reserve:.2f}")
        
        # 分析证明模式
        proof_analysis = self.proof_system.analyze_proof_patterns()
        
        print(f"\n随机证明结果:")
        print(f"  总尝试: {proof_analysis['total_attempts']}")
        print(f"  成功: {proof_analysis['successful_proofs']}")
        print(f"  总体成功率: {proof_analysis['overall_success_rate']:.1%}")
        print(f"  最终认知储备: {proof_analysis['final_cognitive_reserve']:.2f}")
        print(f"  认知疲劳证据: {proof_analysis['fatigue_evidence']}")
        
        return {
            'energy_history': energy_history,
            'cost_history': cost_history,
            'proof_analysis': proof_analysis,
            'final_energy': current_energy,
            'total_cost': current_cost
        }
    
    def _detect_phase_transitions(self, energy_history: List[float], 
                                 cost_history: List[float]) -> Dict:
        """检测认知相变"""
        # 检测相变
        transitions = self.phase_analyzer.detect_phase_transitions(
            energy_history, cost_history
        )
        
        if transitions:
            print(f"检测到 {len(transitions)} 个认知相变")
            
            # 计算临界指数
            critical_exponents = self.phase_analyzer.compute_critical_exponents(
                energy_history, cost_history
            )
            
            # 生成报告
            report = self.phase_analyzer.generate_report()
            print("\n" + report)
            
            # 可视化
            self.phase_analyzer.visualize_phase_diagram(
                energy_history, cost_history,
                save_path="data/figures/cognitive_phase_diagram.png"
            )
            
            return {
                'transitions': transitions,
                'critical_exponents': critical_exponents,
                'n_transitions': len(transitions),
                'reflexive_singularities': sum(1 for t in transitions 
                                             if t.get('reflexive_singularity', False))
            }
        else:
            print("未检测到明显的认知相变")
            return {'transitions': [], 'n_transitions': 0}
    
    def _validate_theoretical_predictions(self, results: Dict) -> Dict:
        """验证理论预测"""
        predictions = {}
        
        # 1. 逻辑熵增定律
        entropy_increase = results['logical_entropy']['entropy_increase']
        predictions['entropy_increase_law'] = {
            'predicted': True,  # 理论预测熵增
            'observed': entropy_increase > 0,
            'magnitude': entropy_increase,
            'supported': entropy_increase > 0
        }
        
        # 2. 反身性奇点存在性
        n_singularities = results['phase_transitions'].get('reflexive_singularities', 0)
        predictions['reflexive_singularity'] = {
            'predicted': True,  # 理论预测存在
            'observed': n_singularities > 0,
            'count': n_singularities,
            'supported': n_singularities > 0
        }
        
        # 3. 认知疲劳效应
        fatigue_evidence = results['stochastic_proofs']['proof_analysis'].get('fatigue_evidence', False)
        predictions['cognitive_fatigue'] = {
            'predicted': True,  # 随机证明系统预测疲劳
            'observed': fatigue_evidence,
            'supported': fatigue_evidence
        }
        
        # 4. Ω与系统强度的关系
        omega_decrease = results['logical_entropy']['omega_decrease']
        predictions['omega_strength_relation'] = {
            'predicted': True,  # 更强的系统有更小的Ω
            'observed': omega_decrease > 0,
            'magnitude': omega_decrease,
            'supported': omega_decrease > 0
        }
        
        # 总体验证
        total_predictions = len(predictions)
        supported = sum(1 for p in predictions.values() if p['supported'])
        support_rate = supported / total_predictions if total_predictions > 0 else 0
        
        predictions['overall'] = {
            'total_predictions': total_predictions,
            'supported_predictions': supported,
            'support_rate': support_rate,
            'theory_supported': support_rate > 0.5
        }
        
        print("\n理论验证结果:")
        print("="*60)
        for name, pred in predictions.items():
            if name != 'overall':
                status = "✓ 支持" if pred['supported'] else "✗ 不支持"
                print(f"{name}: {status}")
        
        overall = predictions['overall']
        print(f"\n总体支持率: {overall['support_rate']:.1%} ({overall['supported_predictions']}/{overall['total_predictions']})")
        if overall['theory_supported']:
            print("✓ 理论整体上得到支持")
        else:
            print("✗ 理论需要进一步调整")
        
        return predictions
    
    def run_and_report(self):
        """运行实验并生成完整报告"""
        results = self.run_comprehensive_analysis()
        
        # 生成最终报告
        print("\n" + "="*70)
        print("集成实验最终报告")
        print("="*70)
        
        # 关键发现摘要
        print("\n关键发现:")
        print(f"1. 逻辑熵增加: {results['logical_entropy']['entropy_increase']:.4f}")
        print(f"2. Chaitin Ω减少: {results['logical_entropy']['omega_decrease']:.6f}")
        print(f"3. 随机证明成功率: {results['stochastic_proofs']['proof_analysis']['overall_success_rate']:.1%}")
        print(f"4. 检测到相变: {results['phase_transitions'].get('n_transitions', 0)}次")
        print(f"5. 反身性奇点: {results['phase_transitions'].get('reflexive_singularities', 0)}次")
        print(f"6. 理论支持率: {results['theory_validation']['overall']['support_rate']:.1%}")
        
        # 科学意义
        print("\n科学意义:")
        if results['theory_validation']['overall']['support_rate'] > 0.7:
            print("✓ 强有力地支持了逻辑熵理论的核心预测")
            print("✓ 验证了Chaitin不可完备性与认知热力学的联系")
            print("✓ 为反身性奇点提供了实证证据")
            print("✓ 展示了几何认知架构的理论优势")
        elif results['theory_validation']['overall']['support_rate'] > 0.5:
            print("○ 部分支持了理论预测")
            print("○ 需要更多数据来确认某些预测")
            print("○ 理论框架基本可行，但需细化")
        else:
            print("⚠ 理论需要重大调整")
            print("⚠ 某些核心预测未被验证")
            print("⚠ 考虑修改理论假设或实验方法")
        
        print("\n" + "="*70)
        
        return results


# 主函数
def main():
    """运行集成实验"""
    experiment = IntegratedChaitinExperiment(n_trials=200)
    results = experiment.run_and_report()
    
    # 保存结果
    import json
    with open('data/results/integrated_experiment_results.json', 'w') as f:
        # 简化结果以便JSON序列化
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    k: (v if not isinstance(v, (np.ndarray, np.generic)) else v.tolist())
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        json.dump(serializable_results, f, indent=2, default=str)
    
    print("\n结果已保存到: data/results/integrated_experiment_results.json")
    print("图表已保存到: data/figures/cognitive_phase_diagram.png")


if __name__ == "__main__":
    main()
