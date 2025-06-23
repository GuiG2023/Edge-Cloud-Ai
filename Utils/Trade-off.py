"""
准确率vs成本权衡分析
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict


class TradeOffAnalyzer:
    def __init__(self):
        self.cost_model = CostModel()

    def run_threshold_sweep(self, evaluator, dataset, thresholds):
        """在不同阈值下测试性能"""
        results = []

        for threshold in thresholds:
            print(f"🔬 测试阈值: {threshold}")

            # 更新分析器阈值
            evaluator.attention_analyzer.routing_threshold = threshold

            # 运行评估
            eval_result = evaluator.run_routing_evaluation(dataset)

            results.append({
                'threshold': threshold,
                'accuracy': eval_result['accuracy']['hybrid'],
                'routing_rate': eval_result['routing']['routing_rate'],
                'cost': eval_result['cost']['hybrid_cost'],
                'cost_savings': eval_result['cost']['cost_savings']
            })

        return results

    def plot_tradeoff_curves(self, sweep_results):
        """绘制权衡曲线"""
        thresholds = [r['threshold'] for r in sweep_results]
        accuracies = [r['accuracy'] for r in sweep_results]
        costs = [r['cost'] for r in sweep_results]
        routing_rates = [r['routing_rate'] for r in sweep_results]
        cost_savings = [r['cost_savings'] for r in sweep_results]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 准确率 vs 阈值
        ax1.plot(thresholds, accuracies, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Routing Threshold')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Routing Threshold')
        ax1.grid(True, alpha=0.3)

        # 2. 成本 vs 阈值
        ax2.plot(thresholds, costs, 'r-s', linewidth=2, markersize=6)
        ax2.set_xlabel('Routing Threshold')
        ax2.set_ylabel('Total Cost')
        ax2.set_title('Cost vs Routing Threshold')
        ax2.grid(True, alpha=0.3)

        # 3. 准确率 vs 成本 (最重要的权衡曲线)
        ax3.plot(costs, accuracies, 'g-^', linewidth=3, markersize=8)
        ax3.set_xlabel('Total Cost')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy vs Cost Trade-off')
        ax3.grid(True, alpha=0.3)

        # 添加基准点
        slm_cost = min(costs)
        llm_cost = max(costs) * 10  # 假设LLM成本
        slm_acc = min(accuracies)
        llm_acc = max(accuracies) * 1.1  # 假设LLM准确率

        ax3.scatter([slm_cost], [slm_acc], color='blue', s=100, label='SLM Only', zorder=5)
        ax3.scatter([llm_cost], [llm_acc], color='red', s=100, label='LLM Only', zorder=5)
        ax3.legend()

        # 4. 路由率 vs 成本节省
        ax4.plot(routing_rates, cost_savings, 'm-d', linewidth=2, markersize=6)
        ax4.set_xlabel('Routing Rate')
        ax4.set_ylabel('Cost Savings')
        ax4.set_title('Routing Rate vs Cost Savings')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig

    def find_optimal_threshold(self, sweep_results, weight_accuracy=0.7, weight_cost=0.3):
        """找到最优阈值"""
        best_score = -1
        best_threshold = None
        best_result = None

        # 归一化指标
        max_acc = max(r['accuracy'] for r in sweep_results)
        min_cost = min(r['cost'] for r in sweep_results)
        max_cost = max(r['cost'] for r in sweep_results)

        for result in sweep_results:
            # 归一化准确率 (越高越好)
            norm_acc = result['accuracy'] / max_acc

            # 归一化成本 (越低越好)
            norm_cost = 1 - (result['cost'] - min_cost) / (max_cost - min_cost)

            # 综合得分
            score = weight_accuracy * norm_acc + weight_cost * norm_cost

            if score > best_score:
                best_score = score
                best_threshold = result['threshold']
                best_result = result

        return best_threshold, best_result, best_score

    def run_complete_tradeoff_analysis(self, dataset):
        """运行完整的权衡分析"""
        print("🚀 开始完整的权衡分析...")

        # 1. 创建评估器
        evaluator = RoutingEffectivenessEvaluator()

        # 2. 定义阈值范围
        thresholds = np.linspace(0.1, 0.8, 8)  # 从0.1到0.8，8个点

        # 3. 扫描不同阈值
        sweep_results = self.run_threshold_sweep(evaluator, dataset, thresholds)

        # 4. 绘制权衡曲线
        self.plot_tradeoff_curves(sweep_results)

        # 5. 找到最优阈值
        optimal_threshold, optimal_result, optimal_score = self.find_optimal_threshold(sweep_results)

        # 6. 输出分析报告
        print("\n" + "=" * 60)
        print("📊 权衡分析报告")
        print("=" * 60)

        print(f"\n🎯 最优阈值: {optimal_threshold:.3f}")
        print(f"📈 最优准确率: {optimal_result['accuracy']:.3f}")
        print(f"💰 最优成本: {optimal_result['cost']:.1f}")
        print(f"🔄 最优路由率: {optimal_result['routing_rate']:.3f}")
        print(f"💎 综合得分: {optimal_score:.3f}")

        # 7. 与基准对比
        baseline_slm = sweep_results[0]  # 假设第一个是最低阈值(接近SLM only)
        baseline_llm_sim = sweep_results[-1]  # 假设最后一个是最高阈值(接近LLM)

        print(f"\n📊 与基准对比:")
        print(f"相比SLM Only:")
        print(f"  准确率提升: {(optimal_result['accuracy'] - baseline_slm['accuracy']) * 100:.1f}%")
        print(f"  成本增加: {(optimal_result['cost'] / baseline_slm['cost'] - 1) * 100:.1f}%")

        print(f"相比模拟LLM Only:")
        print(f"  成本节省: {optimal_result['cost_savings'] * 100:.1f}%")
        print(f"  准确率保持: {optimal_result['accuracy']:.3f}")

        return sweep_results, optimal_threshold


# 使用示例
def run_tradeoff_experiment():
    """运行权衡分析实验"""
    analyzer = TradeOffAnalyzer()

    # 加载数据集
    evaluator = RoutingEffectivenessEvaluator()
    dataset = evaluator.load_gsm8k_dataset(sample_size=30)  # 小样本快速测试

    # 运行分析
    results, optimal_threshold = analyzer.run_complete_tradeoff_analysis(dataset)

    return results, optimal_threshold


if __name__ == "__main__":
    results, optimal_threshold = run_tradeoff_experiment()