"""
基于Chaitin不可完备性的逻辑熵实现
这是理论的核心数学基础
"""
import math
import random
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter


class LogicalEntropyExtension:
    """逻辑熵扩展 - 基于Chaitin不可完备性的近似"""
    
    def __init__(self):
        self.algorithmic_complexity_cache = {}
        self.entropy_history = []
        self.omega_values = {}
        
    def approximate_kolmogorov_complexity(self, text: str) -> float:
        """近似Kolmogorov复杂度（使用压缩率）"""
        if text in self.algorithmic_complexity_cache:
            return self.algorithmic_complexity_cache[text]
        
        # 简化：使用字符串长度作为复杂度的代理
        # 实际应用中应该使用压缩算法
        unique_symbols = len(set(text))
        complexity = len(text) * math.log2(max(unique_symbols, 2))
        
        self.algorithmic_complexity_cache[text] = complexity
        return complexity
    
    def approximate_chaitin_constant(self, system_axioms: Dict, method: str = "symbolic") -> float:
        """
        近似计算Chaitin常数Ω（停机概率）
        
        参数:
            system_axioms: 公理系统字典
            method: 计算方法 ("symbolic", "compression", "theoretical")
        
        返回:
            Ω的近似值
        """
        if str(system_axioms) in self.omega_values:
            return self.omega_values[str(system_axioms)]
        
        omega = 0.0
        
        if method == "symbolic":
            # 基于公理符号的复杂度
            all_axioms = "".join(system_axioms.values())
            K = self.approximate_kolmogorov_complexity(all_axioms)
            omega = 2 ** (-K) if K < 50 else 0.0
            
        elif method == "compression":
            # 模拟压缩（简化）
            import zlib
            compressed = zlib.compress(str(system_axioms).encode())
            compression_ratio = len(compressed) / len(str(system_axioms).encode())
            omega = 1.0 - compression_ratio
            
        elif method == "theoretical":
            # 理论估计：更强的系统有更小的Ω
            # 因为能证明更多陈述 → 停机概率更低
            axiom_count = len(system_axioms)
            logical_symbols = sum(axiom.count(sym) for axiom in system_axioms.values() 
                                for sym in ['∀', '∃', '→', '¬'])
            omega = 0.5 ** (axiom_count * 0.1 + logical_symbols * 0.01)
        
        # 确保在合理范围内
        omega = max(0.0, min(1.0, omega))
        self.omega_values[str(system_axioms)] = omega
        return omega
    
    def compute_logical_entropy(self, system_axioms: Dict, 
                               method: str = "chaitin") -> float:
        """
        计算逻辑熵 H(L)
        
        参数:
            system_axioms: 公理系统
            method: 计算方法 ("chaitin", "shannon", "hybrid")
        
        返回:
            逻辑熵值
        """
        if method == "chaitin":
            # 基于Chaitin Ω的计算
            omega = self.approximate_chaitin_constant(system_axioms)
            p_provable = omega
            p_unprovable = max(0, 1 - omega)
            
            entropy = 0.0
            if p_provable > 0:
                entropy -= p_provable * math.log2(p_provable + 1e-10)
            if p_unprovable > 0:
                entropy -= p_unprovable * math.log2(p_unprovable + 1e-10)
                
        elif method == "shannon":
            # 基于公理复杂度的Shannon熵
            complexities = []
            for axiom in system_axioms.values():
                complexity = self.approximate_kolmogorov_complexity(axiom)
                complexities.append(complexity)
            
            total = sum(complexities)
            if total > 0:
                probs = [c/total for c in complexities]
                entropy = -sum(p * math.log2(p + 1e-10) for p in probs)
            else:
                entropy = 0.0
                
        elif method == "hybrid":
            # 混合方法
            omega = self.approximate_chaitin_constant(system_axioms)
            entropy_shannon = self.compute_logical_entropy(system_axioms, "shannon")
            entropy = 0.7 * entropy_shannon + 0.3 * (-math.log2(omega + 1e-10))
        
        self.entropy_history.append(entropy)
        return entropy
    
    def estimate_incompleteness_degree(self, system_axioms: Dict) -> Dict:
        """估计系统的不可完备性程度"""
        omega = self.approximate_chaitin_constant(system_axioms, "theoretical")
        entropy = self.compute_logical_entropy(system_axioms, "chaitin")
        
        # 基于Ω估计哥德尔语句密度
        # Ω的每一位编码了一个程序的停机问题
        godel_density = 1 - omega
        
        return {
            'chaitin_omega': omega,
            'logical_entropy': entropy,
            'provability_density': omega,
            'unprovability_density': godel_density,
            'estimated_godel_sentences': int(1000 * godel_density),  # 每1000个句子中的估计数量
            'incompleteness_level': self._classify_incompleteness(omega, entropy)
        }
    
    def _classify_incompleteness(self, omega: float, entropy: float) -> str:
        """分类不可完备性程度"""
        if omega > 0.8:
            return "低度不可完备（系统较弱）"
        elif omega > 0.5:
            return "中度不可完备（典型算术系统）"
        elif omega > 0.2:
            return "高度不可完备（强系统）"
        else:
            return "极度不可完备（如ZFC+大基数）"
    
    def verify_entropy_increase(self, system_evolution: List[Dict]) -> Dict:
        """验证逻辑熵增定律"""
        entropies = []
        for system in system_evolution:
            entropy = self.compute_logical_entropy(system)
            entropies.append(entropy)
        
        # 检查是否单调递增
        increasing = all(entropies[i] <= entropies[i+1] for i in range(len(entropies)-1))
        
        # 计算熵增率
        if len(entropies) > 1:
            entropy_increase = entropies[-1] - entropies[0]
            avg_rate = entropy_increase / len(entropies)
        else:
            entropy_increase = 0
            avg_rate = 0
        
        return {
            'entropy_sequence': entropies,
            'entropy_increase': entropy_increase,
            'average_rate': avg_rate,
            'monotonic_increase': increasing,
            'second_law_holds': increasing  # 逻辑热力学第二定律
        }


class StochasticProofSystem:
    """随机证明系统 - 引入认知疲劳和随机性"""
    
    def __init__(self, base_success_rate=0.8, cognitive_decay=0.1):
        self.base_success_rate = base_success_rate
        self.cognitive_decay = cognitive_decay  # 认知衰减系数
        self.fatigue_factor = 0.05  # 疲劳效应系数
        self.proof_history = []
        self.cognitive_reserve = 1.0  # 认知储备
        
    def stochastic_prove(self, theorem: str, system_axioms: Dict, 
                        steps: int, difficulty: float = 1.0) -> Tuple[bool, float]:
        """
        随机化证明过程，考虑认知限制
        
        参数:
            theorem: 要证明的定理
            system_axioms: 公理系统
            steps: 证明步骤数
            difficulty: 定理难度系数
        
        返回:
            (成功与否, 认知成本)
        """
        # 1. 计算定理复杂度
        theorem_complexity = self._compute_theorem_complexity(theorem)
        
        # 2. 调整基础成功率（考虑认知储备）
        adjusted_rate = self.base_success_rate * self.cognitive_reserve
        
        # 3. 复杂度惩罚
        complexity_penalty = math.exp(-self.cognitive_decay * theorem_complexity)
        
        # 4. 步骤疲劳效应
        fatigue_penalty = math.exp(-self.fatigue_factor * steps)
        
        # 5. 难度影响
        difficulty_penalty = 1.0 / (1.0 + difficulty)
        
        # 最终成功率
        success_rate = adjusted_rate * complexity_penalty * fatigue_penalty * difficulty_penalty
        success_rate = max(0.0, min(1.0, success_rate))
        
        # 随机决定
        success = random.random() < success_rate
        
        # 计算认知成本（失败的尝试成本更高）
        if success:
            # 成功成本：步骤数 × 复杂度 × 认知储备消耗
            cost = steps * theorem_complexity * 0.05 * (1.0 / self.cognitive_reserve)
            # 成功会略微恢复认知储备
            self.cognitive_reserve = min(1.0, self.cognitive_reserve + 0.01)
        else:
            # 失败成本：更高，且消耗更多认知储备
            cost = steps * theorem_complexity * 0.08 * (1.5 / self.cognitive_reserve)
            # 失败消耗认知储备
            self.cognitive_reserve = max(0.1, self.cognitive_reserve - 0.05)
        
        # 记录
        self.proof_history.append({
            'theorem': theorem,
            'complexity': theorem_complexity,
            'steps': steps,
            'difficulty': difficulty,
            'success': success,
            'success_rate': success_rate,
            'cost': cost,
            'cognitive_reserve': self.cognitive_reserve
        })
        
        return success, cost
    
    def _compute_theorem_complexity(self, theorem: str) -> float:
        """计算定理的逻辑复杂度"""
        # 逻辑符号权重
        symbol_weights = {
            '∀': 2.0, '∃': 2.0, '∧': 1.0, '∨': 1.0,
            '→': 1.5, '¬': 1.0, '=': 0.5, '≠': 0.5,
            '∈': 1.0, '⊆': 1.0, '≡': 1.5
        }
        
        complexity = 0.0
        for symbol, weight in symbol_weights.items():
            complexity += theorem.count(symbol) * weight
        
        # 变量数量
        variables = set([c for c in theorem if c.isalpha() and c.islower()])
        complexity += len(variables) * 0.3
        
        # 嵌套深度（括号匹配）
        depth = 0
        max_depth = 0
        for char in theorem:
            if char == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ')':
                depth -= 1
        
        complexity += max_depth * 0.5
        
        return max(0.1, complexity)  # 确保最小值
    
    def analyze_proof_patterns(self) -> Dict:
        """分析证明模式，揭示认知规律"""
        if not self.proof_history:
            return {'error': '没有证明历史'}
        
        n = len(self.proof_history)
        successes = sum(1 for p in self.proof_history if p['success'])
        overall_rate = successes / n if n > 0 else 0
        
        # 随时间的变化
        if n >= 2:
            success_rates = [p['success'] for p in self.proof_history]
            # 计算自相关性（认知状态的持续性）
            autocorrelation = np.corrcoef(success_rates[:-1], success_rates[1:])[0, 1] if len(success_rates) > 1 else 0
        else:
            autocorrelation = 0
        
        # 按复杂度分组
        complexity_bins = {}
        for p in self.proof_history:
            bin_key = int(p['complexity'])
            if bin_key not in complexity_bins:
                complexity_bins[bin_key] = {'attempts': 0, 'successes': 0, 'total_cost': 0}
            
            complexity_bins[bin_key]['attempts'] += 1
            complexity_bins[bin_key]['total_cost'] += p['cost']
            if p['success']:
                complexity_bins[bin_key]['successes'] += 1
        
        # 计算每个bin的成功率和平均成本
        for bin_key in complexity_bins:
            bin_data = complexity_bins[bin_key]
            bin_data['success_rate'] = bin_data['successes'] / bin_data['attempts'] if bin_data['attempts'] > 0 else 0
            bin_data['avg_cost'] = bin_data['total_cost'] / bin_data['attempts'] if bin_data['attempts'] > 0 else 0
        
        # 认知储备变化
        cognitive_reserves = [p['cognitive_reserve'] for p in self.proof_history]
        reserve_changes = np.diff(cognitive_reserves) if len(cognitive_reserves) > 1 else [0]
        
        return {
            'total_attempts': n,
            'successful_proofs': successes,
            'overall_success_rate': overall_rate,
            'success_by_complexity': complexity_bins,
            'autocorrelation': autocorrelation,
            'final_cognitive_reserve': cognitive_reserves[-1] if cognitive_reserves else 1.0,
            'reserve_depletion': 1.0 - cognitive_reserves[-1] if cognitive_reserves else 0.0,
            'average_reserve_change': np.mean(reserve_changes) if len(reserve_changes) > 0 else 0,
            'fatigue_evidence': np.mean(reserve_changes) < 0 if len(reserve_changes) > 0 else False
        }
    
    def reset_cognitive_state(self):
        """重置认知状态（开始新的认知任务）"""
        self.cognitive_reserve = 1.0
        self.proof_history = []
