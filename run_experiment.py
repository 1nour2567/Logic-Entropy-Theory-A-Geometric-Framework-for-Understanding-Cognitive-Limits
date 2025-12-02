#!/usr/bin/env python3
"""
主运行脚本 - 一键运行所有实验
"""
import sys
import os

# 添加src到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from experiments.comparison_experiment import run_complete_experiment


def main():
    """主函数"""
    print("="*80)
    print("逻辑熵理论 - 几何认知架构 vs 深度学习")
    print("完整对比实验")
    print("="*80)
    
    try:
        run_complete_experiment()
        print("\n✅ 实验完成！")
    except Exception as e:
        print(f"\n❌ 实验出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
