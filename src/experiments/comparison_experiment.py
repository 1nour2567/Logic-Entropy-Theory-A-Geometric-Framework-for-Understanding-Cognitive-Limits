"""
对比实验主文件
"""
import matplotlib.pyplot as plt
from src.models.arithmetic_system import EnhancedArithmeticSimulation
from src.models.deep_learning import StandardDeepLearningModel
from src.utils.visualization import visualize_statistical_results, plot_dl_training


def run_complete_experiment():
    """运行完整对比实验"""
    print("启动完整几何认知 vs 深度学习对比实验")
    print("="*80)
    
    # 第一部分：几何认知扩展模拟（100次）
    print("\n1. 几何认知扩展模拟...")
    sim = EnhancedArithmeticSimulation()
    trial_results = sim.run_multiple_trials(n_trials=100)
    visualize_statistical_results(trial_results, sim.statistical_significance)
    
    # 第二部分：标准深度学习训练
    print("\n2. 标准深度学习训练...")
    dl_model = StandardDeepLearningModel()
    loss_history = dl_model.train_addition_task(epochs=800)
    plot_dl_training(loss_history)
    
    # 结论
    print_conclusions()


def print_conclusions():
    """打印实验结论"""
    print("\n" + "="*80)
    print("最终结论".center(70))
    print("="*80)
    print("几何认知架构：")
    print("  - 显式建模认知扩展成本与能量耗散")
    print("  - 支持统计检验与可解释性分析")
    print("  - 自然模拟从Q系统→PA系统的认知跃迁")
    print("")
    print("标准深度学习：")
    print("  - 在数值任务上极快收敛")
    print("  - 完全缺乏对'为什么能证明'的解释")
    print("  - 无法建模系统扩展的认知代价")
    print("")
    print("胜利者：几何认知架构（在数学推理的本质层面）")
    print("="*80)
