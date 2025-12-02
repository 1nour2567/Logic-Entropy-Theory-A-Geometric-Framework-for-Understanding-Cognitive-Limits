"""
认知相变分析器
用于检测和分类认知系统的相变
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import Counter


class CognitivePhaseTransitionAnalyzer:
    """认知相变分析器 - 检测理论预测的临界行为"""
    
    def __init__(self, detection_window=5, sensitivity=0.1):
        self.detection_window = detection_window
        self.sensitivity = sensitivity
        self.transitions = []
        self.critical_points = []
        
    def detect_phase_transitions(self, energy_history: List[float], 
                               cost_history: List[float],
                               entropy_history: List[float] = None) -> List[Dict]:
        """
        检测认知相变
        
        参数:
            energy_history: 认知能量序列
            cost_history: 认知成本序列
            entropy_history: 逻辑熵序列（可选）
        
        返回:
            检测到的相变列表
        """
        if len(energy_history) < self.detection_window * 2:
            return []
        
        n = len(energy_history)
        transitions = []
        
        # 使用滑动窗口检测变化
        for i in range(self.detection_window, n - self.detection_window):
            # 前窗口
            prev_window_start = i - self.detection_window
            prev_window_end = i
            
            # 后窗口
            next_window_start = i
            next_window_end = i + self.detection_window
            
            # 提取窗口数据
            prev_energy = energy_history[prev_window_start:prev_window_end]
            next_energy = energy_history[next_window_start:next_window_end]
            
            prev_cost = cost_history[prev_window_start:prev_window_end]
            next_cost = cost_history[next_window_start:next_window_end]
            
            # 计算统计量
            energy_change = self._compute_phase_change(prev_energy, next_energy)
            cost_change = self._compute_phase_change(prev_cost, next_cost)
            
            # 如果有熵数据
            if entropy_history:
                prev_entropy = entropy_history[prev_window_start:prev_window_end]
                next_entropy = entropy_history[next_window_start:next_window_end]
                entropy_change = self._compute_phase_change(prev_entropy, next_entropy)
            else:
                entropy_change = 0
            
            # 检测相变：统计量显著变化
            if (abs(energy_change) > self.sensitivity or 
                abs(cost_change) > self.sensitivity):
                
                transition_type = self._classify_transition(
                    energy_change, cost_change, entropy_change
                )
                
                transition = {
                    'position': i,
                    'energy_change': energy_change,
                    'cost_change': cost_change,
                    'entropy_change': entropy_change,
                    'type': transition_type,
                    'subtype': self._determine_subtype(transition_type, energy_change, cost_change),
                    'prev_energy_mean': np.mean(prev_energy),
                    'next_energy_mean': np.mean(next_energy),
                    'prev_cost_mean': np.mean(prev_cost),
                    'next_cost_mean': np.mean(next_cost),
                    'magnitude': abs(energy_change) + abs(cost_change)  # 相变幅度
                }
                
                # 进一步分析：是否反身性奇点？
                if transition_type in ["认知重构", "反身性奇点"]:
                    transition['reflexive_singularity'] = True
                    transition['lambda_critical'] = self._estimate_critical_load(i, n)
                else:
                    transition['reflexive_singularity'] = False
                
                transitions.append(transition)
                self.critical_points.append(i)
        
        self.transitions = transitions
        return transitions
    
    def _compute_phase_change(self, prev_data: List[float], next_data: List[float]) -> float:
        """计算两个窗口间的相变强度"""
        if len(prev_data) == 0 or len(next_data) == 0:
            return 0.0
        
        # 使用均值变化 + 分布变化的组合
        mean_change = (np.mean(next_data) - np.mean(prev_data)) / (np.std(prev_data) + 1e-10)
        
        # 考虑方差变化（相变常伴随方差增大）
        var_change = (np.var(next_data) - np.var(prev_data)) / (np.var(prev_data) + 1e-10)
        
        # 考虑自相关变化（相变时时间相关性改变）
        if len(prev_data) > 1 and len(next_data) > 1:
            acf_prev = np.corrcoef(prev_data[:-1], prev_data[1:])[0, 1] if np.std(prev_data) > 0 else 0
            acf_next = np.corrcoef(next_data[:-1], next_data[1:])[0, 1] if np.std(next_data) > 0 else 0
            acf_change = acf_next - acf_prev
        else:
            acf_change = 0
        
        # 综合相变指标
        phase_change = 0.6 * mean_change + 0.3 * var_change + 0.1 * acf_change
        return phase_change
    
    def _classify_transition(self, energy_change: float, 
                           cost_change: float, 
                           entropy_change: float) -> str:
        """分类相变类型"""
        # 根据理论预测的分类
        if energy_change < -0.3 and cost_change > 0.3:
            if entropy_change > 0.2:
                return "反身性奇点"  # 理论预测的关键相变
            else:
                return "认知重构"
        elif energy_change > 0.2 and cost_change < -0.2:
            return "认知突破"
        elif abs(energy_change) < 0.1 and abs(cost_change) < 0.1:
            return "平稳过渡"
        elif entropy_change > 0.3:
            return "逻辑熵增相变"
        else:
            return "混合相变"
    
    def _determine_subtype(self, main_type: str, energy_change: float, cost_change: float) -> str:
        """确定相变子类型"""
        if main_type == "反身性奇点":
            if energy_change < -0.5:
                return "类型I: 状态奇点"
            elif cost_change > 0.5:
                return "类型II: 拓扑奇点"
            else:
                return "类型III: 约束奇点"
        elif main_type == "认知重构":
            if energy_change < -0.4:
                return "能量主导重构"
            elif cost_change > 0.4:
                return "成本主导重构"
            else:
                return "平衡重构"
        return "标准相变"
    
    def _estimate_critical_load(self, position: int, total_length: int) -> float:
        """估计临界反身性负荷λ_c"""
        # 假设λ随时间线性增加，临界点在position处
        return position / total_length
    
    def compute_critical_exponents(self, energy_history: List[float], 
                                 cost_history: List[float]) -> Dict:
        """计算临界指数（如理论预测的ν, β等）"""
        if not self.transitions:
            return {'error': '没有检测到相变'}
        
        # 找到最显著的相变
        main_transition = max(self.transitions, key=lambda x: x['magnitude'])
        critical_point = main_transition['position']
        
        # 在临界点附近拟合标度律
        # 这需要更多数据点，这里简化处理
        n = len(energy_history)
        
        if critical_point > 10 and n - critical_point > 10:
            # 临界点前的区域
            pre_region = energy_history[max(0, critical_point-10):critical_point]
            # 临界点后的区域
            post_region = energy_history[critical_point:min(n, critical_point+10)]
            
            # 计算关联长度指数ν（简化）
            # 实际上需要更复杂的有限尺寸标度分析
            energy_range_pre = max(pre_region) - min(pre_region) if pre_region else 1
            energy_range_post = max(post_region) - min(post_region) if post_region else 1
            
            # 简化的临界指数估计
            nu = 1.0 / (abs(energy_range_post - energy_range_pre) + 1e-10)
            beta = 0.5  # 平均场值
            gamma = 1.0  # 平均场值
            
            return {
                'critical_point': critical_point,
                'nu_estimated': min(10.0, nu),  # 限制范围
                'beta_estimated': beta,
                'gamma_estimated': gamma,
                'scaling_law_supported': abs(main_transition['energy_change']) > 0.3
            }
        
        return {'error': '无法计算临界指数'}
    
    def visualize_phase_diagram(self, energy_history: List[float], 
                              cost_history: List[float],
                              entropy_history: List[float] = None,
                              save_path: str = None):
        """可视化相图（认知状态演化）"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 能量演化
        ax = axes[0, 0]
        ax.plot(energy_history, 'b-', linewidth=2, alpha=0.7, label='认知能量 $D_e$')
        ax.set_xlabel('时间步 / 认知操作')
        ax.set_ylabel('认知能量')
        ax.set_title('(a) 认知能量演化')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 标记相变点
        for trans in self.transitions:
            ax.axvline(x=trans['position'], color='r', linestyle='--', 
                      alpha=0.5, linewidth=1)
        
        # 2. 成本演化
        ax = axes[0, 1]
        ax.plot(cost_history, 'r-', linewidth=2, alpha=0.7, label='累计成本 $\\sum\\Delta C$')
        ax.set_xlabel('时间步 / 认知操作')
        ax.set_ylabel('累计成本')
        ax.set_title('(b) 认知成本积累')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 3. 能量-成本相空间
        ax = axes[0, 2]
        scatter = ax.scatter(energy_history, cost_history, 
                           c=range(len(energy_history)), cmap='viridis',
                           s=20, alpha=0.6)
        ax.set_xlabel('认知能量 $D_e$')
        ax.set_ylabel('累计成本 $\\sum\\Delta C$')
        ax.set_title('(c) 认知相空间轨迹')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='时间步')
        
        # 标记相变点
        for trans in self.transitions:
            if (0 <= trans['position'] < len(energy_history) and 
                0 <= trans['position'] < len(cost_history)):
                ax.scatter(energy_history[trans['position']], 
                          cost_history[trans['position']],
                          color='red', s=100, marker='*', 
                          label=f"{trans['type']}")
        
        # 4. 相变幅度图
        ax = axes[1, 0]
        if self.transitions:
            positions = [t['position'] for t in self.transitions]
            magnitudes = [t['magnitude'] for t in self.transitions]
            types = [t['type'] for t in self.transitions]
            
            colors = {'反身性奇点': 'red', '认知重构': 'orange', 
                     '认知突破': 'green', '逻辑熵增相变': 'blue',
                     '混合相变': 'purple', '平稳过渡': 'gray'}
            
            for pos, mag, ttype in zip(positions, magnitudes, types):
                color = colors.get(ttype, 'black')
                ax.bar(pos, mag, color=color, alpha=0.7, width=5)
            
            ax.set_xlabel('时间步')
            ax.set_ylabel('相变幅度')
            ax.set_title('(d) 相变幅度与类型')
            ax.grid(True, alpha=0.3)
            
            # 创建图例
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=color, label=ttype) 
                             for ttype, color in colors.items()]
            ax.legend(handles=legend_elements, loc='upper right')
        
        # 5. 临界指数分析
        ax = axes[1, 1]
        if self.transitions and len(energy_history) > 20:
            # 计算滑动窗口的局部斜率（模拟关联长度）
            window = 5
            local_slopes = []
            for i in range(window, len(energy_history)-window):
                x = list(range(i-window, i+window))
                y = energy_history[i-window:i+window]
                if len(set(y)) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    local_slopes.append(abs(slope))
                else:
                    local_slopes.append(0)
            
            ax.plot(range(window, len(energy_history)-window), 
                   local_slopes, 'g-', alpha=0.7)
            ax.set_xlabel('时间步')
            ax.set_ylabel('局部变化率')
            ax.set_title('(e) 局部动力学变化（关联长度代理）')
            ax.grid(True, alpha=0.3)
            
            # 标记临界点
            for trans in self.transitions:
                if window <= trans['position'] < len(energy_history)-window:
                    ax.axvline(x=trans['position'], color='r', 
                              linestyle='--', alpha=0.5)
        
        # 6. 相变类型分布
        ax = axes[1, 2]
        if self.transitions:
            type_counts = Counter([t['type'] for t in self.transitions])
            types = list(type_counts.keys())
            counts = list(type_counts.values())
            
            ax.bar(types, counts, color=[colors.get(t, 'gray') for t in types])
            ax.set_xlabel('相变类型')
            ax.set_ylabel('出现次数')
            ax.set_title('(f) 相变类型分布')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"相图已保存到: {save_path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """生成相变分析报告"""
        if not self.transitions:
            return "未检测到明显的认知相变。"
        
        report = []
        report.append("="*60)
        report.append("认知相变分析报告")
        report.append("="*60)
        report.append(f"检测到 {len(self.transitions)} 个相变事件")
        
        # 主要相变
        main_transitions = sorted(self.transitions, key=lambda x: x['magnitude'], reverse=True)[:3]
        report.append("\n主要相变事件:")
        for i, trans in enumerate(main_transitions, 1):
            report.append(f"\n{i}. 位置: {trans['position']}")
            report.append(f"   类型: {trans['type']} ({trans['subtype']})")
            report.append(f"   能量变化: {trans['energy_change']:.3f}")
            report.append(f"   成本变化: {trans['cost_change']:.3f}")
            report.append(f"   幅度: {trans['magnitude']:.3f}")
            if trans.get('reflexive_singularity', False):
                report.append(f"   ⚠️ 反身性奇点 (λ ≈ {trans.get('lambda_critical', 0):.2f})")
        
        # 统计摘要
        report.append("\n统计摘要:")
        type_dist = Counter([t['type'] for t in self.transitions])
        for ttype, count in type_dist.items():
            report.append(f"  {ttype}: {count}次 ({count/len(self.transitions)*100:.1f}%)")
        
        # 理论验证
        reflexive_singularities = sum(1 for t in self.transitions 
                                     if t.get('reflexive_singularity', False))
        report.append(f"\n理论验证:")
        report.append(f"  反身性奇点: {reflexive_singularities}次")
        if reflexive_singularities > 0:
            report.append("  ✓ 支持反身性奇点定理的预测")
        
        avg_magnitude = np.mean([t['magnitude'] for t in self.transitions])
        report.append(f"  平均相变幅度: {avg_magnitude:.3f}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
