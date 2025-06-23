"""
优化后的阶段1实验：结合完整统计分析和路由决策
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List

# 注意力熵分析器
class OptimizedAttentionEntropyAnalyzer:
    def __init__(self, entropy_threshold=2.0):
        self.entropy_threshold = entropy_threshold # 熵的阈值，用于归一化
        self.complexity_history = []              # 存储历史记录

#entropy_threshold=2.0 的含义：这是一个经验值，用来将熵值转换为0-1的复杂度分数 熵值通常在0-4之间，2.0作为中点比较合理 （后续疑问）

    def calculate_attention_entropy(self, attention_weights):
        """
        计算注意力熵 - 支持多种计算方式
        输入: attention_weights [num_heads, seq_len] 或 [seq_len, seq_len]
        """
        # 支持两种输入格式：
        # 格式1: [num_heads, seq_len, seq_len] - 完整注意力矩阵
        # 格式2: [num_heads, seq_len] - 单个token的注意力分布

        if len(attention_weights.shape) == 3:  # [num_heads, seq_len, seq_len]
            # 对于完整attention矩阵，计算每个query的熵
            entropies = []
            for head in attention_weights: # 遍历每个注意力头
                head_entropies = []
                for i in range(head.shape[0]):
                    attn_dist = head[i] + 1e-9  # 避免log(0) # 避免数学错误：log(0) = -∞ , 1e-9 是一个极小的数，不会影响结果但能避免数值问题
                    # 熵公式：H = -Σ(p * log(p))
                    entropy = -torch.sum(attn_dist * torch.log(attn_dist))
                    head_entropies.append(entropy.item())
                entropies.append(np.mean(head_entropies)) # 该头的平均熵
            return entropies

        else:  # [num_heads, seq_len] - 单个token的注意力分布
            entropies = []
            for head_attn in attention_weights:
                head_attn = head_attn + 1e-9
                entropy = -torch.sum(head_attn * torch.log(head_attn))
                entropies.append(entropy.item())
            return entropies

    def predict_complexity(self, attention_weights):
        """基于注意力熵预测复杂度"""
        entropies = self.calculate_attention_entropy(attention_weights)
        avg_entropy = np.mean(entropies)   # 平均熵
        max_entropy = np.max(entropies)    # 最大熵
        entropy_std = np.std(entropies)    # 熵的标准差

        # 多维度复杂度评估
        complexity_score = min(avg_entropy / self.entropy_threshold, 1.0)

        return {
            'complexity_score': complexity_score,  # 主要指标
            'avg_entropy': avg_entropy,
            'max_entropy': max_entropy,
            'entropy_std': entropy_std,
            'head_entropies': entropies,
            'is_complex': complexity_score > 0.5  # 二分类结果
        }

    def should_route_to_cloud(self, attention_weights, threshold=0.5):
        """路由决策"""
        result = self.predict_complexity(attention_weights)
        return result['complexity_score'] > threshold, result

'''
————————————————————————————————实验框架————————————————————————————————
'''

class ComprehensiveBaselineExperiment:
    # 加载预训练模型
    def __init__(self, model_name="microsoft/DialoGPT-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True) # 关键：输出注意力权重
        self.analyzer = OptimizedAttentionEntropyAnalyzer()

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 扩展的测试数据集
        self.task_dataset = {
            "simple": [
                "What is 2+2?",
                "The capital of France is",
                "My name is",
                "What color is the sky?",
                "How many days in a week?",
                "What is water made of?",
                "The sun rises in the",
                "1+1 equals",
                "Cats are",
                "The alphabet starts with"
            ],
            "medium": [
                "Why do objects fall down?",
                "How does a bicycle work?",
                "What causes rain?",
                "Why is the ocean salty?",
                "How do plants make food?",
                "What makes ice float?",
                "Why do we have seasons?",
                "How do computers work?",
                "What is electricity?",
                "Why do we dream?"
            ],
            "complex": [
                "Explain the relationship between quantum mechanics and general relativity",
                "Analyze the economic impact of artificial intelligence on employment",
                "What would happen if gravity suddenly became twice as strong?",
                "Discuss the ethical implications of genetic engineering",
                "How might climate change affect global food security?",
                "Evaluate the societal effects of social media on democracy",
                "If I have 3 apples and give away 1.5 apples, then buy 2.7 more apples, what's the philosophical meaning?",
                "Compare the advantages and disadvantages of different renewable energy sources",
                "How do cultural differences affect international business negotiations?",
                "What are the long-term consequences of space exploration for humanity?"
            ]
        }

    def extract_attention_features(self, text, method="last_token"):
        """提取注意力特征"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # 获取最后一层的attention
        last_attention = outputs.attentions[-1][0]  # [num_heads, seq_len, seq_len]

        if method == "last_token":
            # 分析最后一个有效token的attention pattern
            # 对于生成式模型，最后一个token：
            # 1. 包含了对整个序列的理解
            # 2. 是模型进行下一步预测的基础
            # 3. 最能反映任务的整体复杂度
            seq_len = inputs['attention_mask'].sum().item()
            last_token_attn = last_attention[:, seq_len - 1, :]  # [num_heads, seq_len]
            return last_token_attn

        elif method == "average":
            # 平均所有token的attention
            return last_attention

        else:
            raise ValueError(f"Unknown method: {method}")

    def run_single_task(self, text, complexity_label):
        """运行单个任务"""
        attention_weights = self.extract_attention_features(text)
        result = self.analyzer.predict_complexity(attention_weights)

        # 添加真实标签
        result['true_complexity'] = complexity_label
        result['task'] = text

        return result

    def run_comprehensive_experiment(self):
        """运行完整实验"""
        print("🧪 运行优化后的阶段1实验...")

        all_results = []

        # 运行所有任务
        for complexity, tasks in self.task_dataset.items():
            print(f"\n📊 处理{complexity}任务...")

            for task in tasks:
                result = self.run_single_task(task, complexity)
                all_results.append(result)
                print(f"'{task[:50]}...' -> 复杂度={result['complexity_score']:.3f}")

        # 转换为DataFrame便于分析
        df = pd.DataFrame(all_results)

        return self.analyze_results(df)

    def analyze_results(self, df):
        """完整的结果分析"""
        print("\n" + "=" * 60)
        print("📈 实验结果分析")
        print("=" * 60)

        # 1. 描述性统计
        summary = df.groupby('true_complexity')['complexity_score'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(3)
        print("\n1. 描述性统计:")
        print(summary)

        # 2. 可视化
        self.plot_results(df)

        # 3. 统计检验
        self.statistical_tests(df)

        # 4. 相关性分析
        self.correlation_analysis(df)

        # 5. 路由决策分析
        self.routing_analysis(df)

        return df

    def plot_results(self, df):
        """结果可视化"""
        plt.figure(figsize=(15, 5))

        # 箱线图
        plt.subplot(1, 3, 1)
        df.boxplot(column='complexity_score', by='true_complexity', ax=plt.gca())
        plt.title('复杂度分数分布')
        plt.ylabel('复杂度分数')

        # 平均熵对比
        plt.subplot(1, 3, 2)
        df.boxplot(column='avg_entropy', by='true_complexity', ax=plt.gca())
        plt.title('平均注意力熵分布')
        plt.ylabel('平均熵')

        # 散点图
        plt.subplot(1, 3, 3)
        complexity_mapping = {'simple': 1, 'medium': 2, 'complex': 3}
        df['complexity_numeric'] = df['true_complexity'].map(complexity_mapping)
        plt.scatter(df['complexity_numeric'], df['complexity_score'], alpha=0.6)
        plt.xlabel('真实复杂度')
        plt.ylabel('预测复杂度分数')
        plt.title('真实 vs 预测复杂度')

        plt.tight_layout()
        plt.show()

    def statistical_tests(self, df):
        """统计显著性检验"""
        print("\n2. 统计显著性检验:")

        simple_scores = df[df['true_complexity'] == 'simple']['complexity_score']
        medium_scores = df[df['true_complexity'] == 'medium']['complexity_score']
        complex_scores = df[df['true_complexity'] == 'complex']['complexity_score']

        # ANOVA检验
        f_stat, p_value = stats.f_oneway(simple_scores, medium_scores, complex_scores)
        print(f"ANOVA F统计量: {f_stat:.4f}, p值: {p_value:.4f}")

        if p_value < 0.05:
            print("✅ 组间差异显著 (p < 0.05)")
        else:
            print("❌ 组间差异不显著 (p >= 0.05)")

        # 两两比较
        from scipy.stats import ttest_ind
        t1, p1 = ttest_ind(simple_scores, complex_scores)
        print(f"简单 vs 复杂任务 t检验: t={t1:.3f}, p={p1:.4f}")

    def correlation_analysis(self, df):
        """相关性分析"""
        print("\n3. 相关性分析:")

        complexity_mapping = {'simple': 1, 'medium': 2, 'complex': 3}
        df['complexity_numeric'] = df['true_complexity'].map(complexity_mapping)

        correlation = df['complexity_score'].corr(df['complexity_numeric'])
        print(f"复杂度分数与真实复杂度的相关系数: {correlation:.4f}")

        if correlation > 0.5:
            print("✅ 强正相关 - 假设验证成功!")
        elif correlation > 0.3:
            print("⚠️ 中等相关 - 有一定效果但需改进")
        else:
            print("❌ 弱相关 - 需要重新考虑方法")

    def routing_analysis(self, df):
        """路由决策分析"""
        print("\n4. 路由决策分析:")

        # 计算每个复杂度级别的路由决策
        routing_stats = df.groupby('true_complexity')['is_complex'].agg([
            'count', 'sum', lambda x: (x.sum() / len(x) * 100)
        ]).round(1)
        routing_stats.columns = ['总数', '路由到云端', '路由比例(%)']
        print(routing_stats)

        # 理想情况：simple都不路由，complex都路由
        simple_correct = (df[df['true_complexity'] == 'simple']['is_complex'] == False).sum()
        complex_correct = (df[df['true_complexity'] == 'complex']['is_complex'] == True).sum()

        simple_total = len(df[df['true_complexity'] == 'simple'])
        complex_total = len(df[df['true_complexity'] == 'complex'])

        print(f"\n路由准确性:")
        print(f"简单任务正确路由率: {simple_correct / simple_total * 100:.1f}%")
        print(f"复杂任务正确路由率: {complex_correct / complex_total * 100:.1f}%")

    def save_results(self, df, filename="stage1_results.csv"):
        """保存结果"""
        df.to_csv(filename, index=False)
        print(f"\n💾 结果已保存到 {filename}")


if __name__ == "__main__":
    # 运行实验
    experiment = ComprehensiveBaselineExperiment()
    results_df = experiment.run_comprehensive_experiment()

    # 保存结果
    experiment.save_results(results_df)

    print("\n🎉 阶段1实验完成!")