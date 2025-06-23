"""
Fixed Stage 1 Experiment: Multi-dimensional attention analysis
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List


class EnhancedAttentionAnalyzer:
    def __init__(self):
        self.feature_weights = {
            'entropy': 0.4,
            'variance': 0.3,
            'concentration': 0.2,
            'cross_layer': 0.1
        }

    def extract_multi_dimensional_features(self, text, model, tokenizer):
        """提取多维度注意力特征"""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # 获取所有层的注意力
        all_attentions = outputs.attentions  # [层数, batch, 头数, seq_len, seq_len]

        features = {}

        # 1. 多层熵特征
        features.update(self.calculate_entropy_features(all_attentions, inputs))

        # 2. 注意力方差特征
        features.update(self.calculate_variance_features(all_attentions, inputs))

        # 3. 注意力集中度特征
        features.update(self.calculate_concentration_features(all_attentions, inputs))

        # 4. 跨层一致性特征
        features.update(self.calculate_cross_layer_features(all_attentions, inputs))

        return features

    def calculate_entropy_features(self, all_attentions, inputs):
        """计算多种熵特征"""
        seq_len = inputs['attention_mask'].sum().item()

        # 最后一层的熵（原方法）
        last_layer = all_attentions[-1][0]  # [头数, seq_len, seq_len]
        last_token_entropy = self._calculate_token_entropy(
            last_layer[:, seq_len - 1, :]
        )

        # 所有token的平均熵
        all_token_entropies = []
        for i in range(seq_len):
            token_entropy = self._calculate_token_entropy(last_layer[:, i, :])
            all_token_entropies.append(token_entropy)
        avg_token_entropy = np.mean(all_token_entropies)

        # 关键token的熵（问号、动词等）
        key_token_entropy = self._calculate_key_token_entropy(
            last_layer, inputs, seq_len
        )

        return {
            'last_token_entropy': last_token_entropy,
            'avg_token_entropy': avg_token_entropy,
            'key_token_entropy': key_token_entropy,
            'entropy_variance': np.var(all_token_entropies)
        }

    def calculate_variance_features(self, all_attentions, inputs):
        """计算注意力方差特征"""
        seq_len = inputs['attention_mask'].sum().item()
        last_layer = all_attentions[-1][0]

        # 每个头的注意力方差
        head_variances = []
        for head in range(last_layer.shape[0]):
            # 计算该头最后token的注意力方差
            attn_weights = last_layer[head, seq_len - 1, :seq_len]
            variance = torch.var(attn_weights).item()
            head_variances.append(variance)

        return {
            'avg_attention_variance': np.mean(head_variances),
            'max_attention_variance': np.max(head_variances),
            'variance_std': np.std(head_variances)
        }

    def calculate_concentration_features(self, all_attentions, inputs):
        """计算注意力集中度特征"""
        seq_len = inputs['attention_mask'].sum().item()
        last_layer = all_attentions[-1][0]

        concentrations = []
        for head in range(last_layer.shape[0]):
            attn_weights = last_layer[head, seq_len - 1, :seq_len]
            # 集中度：最大注意力权重
            max_attention = torch.max(attn_weights).item()
            # Gini系数（不平等程度）
            gini = self._calculate_gini_coefficient(attn_weights)
            concentrations.extend([max_attention, gini])

        return {
            'avg_max_attention': np.mean(concentrations[::2]),  # 每隔一个取最大值
            'avg_gini_coefficient': np.mean(concentrations[1::2]),  # 每隔一个取Gini
        }

    def calculate_cross_layer_features(self, all_attentions, inputs):
        """计算跨层一致性特征"""
        seq_len = inputs['attention_mask'].sum().item()

        # 比较最后3层的注意力模式相似度
        if len(all_attentions) >= 3:
            last_layers = all_attentions[-3:]
            correlations = []

            for i in range(len(last_layers) - 1):
                layer1 = last_layers[i][0][:, seq_len - 1, :seq_len]  # [头数, seq_len]
                layer2 = last_layers[i + 1][0][:, seq_len - 1, :seq_len]

                # 计算每个头的相关性
                for head in range(layer1.shape[0]):
                    try:
                        corr = torch.corrcoef(torch.stack([layer1[head], layer2[head]]))[0, 1]
                        if not torch.isnan(corr):
                            correlations.append(corr.item())
                    except:
                        continue

            cross_layer_consistency = np.mean(correlations) if correlations else 0
        else:
            cross_layer_consistency = 0

        return {
            'cross_layer_consistency': cross_layer_consistency
        }

    def _calculate_token_entropy(self, attention_weights):
        """计算单个token的平均熵"""
        entropies = []
        for head_attn in attention_weights:
            head_attn = head_attn + 1e-9
            entropy = -torch.sum(head_attn * torch.log(head_attn))
            entropies.append(entropy.item())
        return np.mean(entropies)

    def _calculate_key_token_entropy(self, attention_matrix, inputs, seq_len):
        """计算关键token的熵"""
        # 简化版：计算中间token的平均熵（避免[CLS]和[SEP]）
        if seq_len > 3:
            middle_start = 1
            middle_end = seq_len - 1
            middle_entropies = []

            for i in range(middle_start, middle_end):
                token_entropy = self._calculate_token_entropy(attention_matrix[:, i, :])
                middle_entropies.append(token_entropy)

            return np.mean(middle_entropies)
        else:
            return self._calculate_token_entropy(attention_matrix[:, seq_len - 1, :])

    def _calculate_gini_coefficient(self, weights):
        """计算Gini系数（衡量不平等程度）"""
        weights = weights.detach().numpy()
        weights = np.sort(weights)
        n = len(weights)
        if n == 0 or weights.sum() == 0:
            return 0
        cumsum = np.cumsum(weights)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        return gini

    def predict_complexity_enhanced(self, features):
        """基于多维特征预测复杂度"""
        # 加权组合多个特征
        complexity_score = (
                self.feature_weights['entropy'] * self._normalize_feature(features['avg_token_entropy'], 0, 3) +
                self.feature_weights['variance'] * self._normalize_feature(features['avg_attention_variance'], 0, 0.1) +
                self.feature_weights['concentration'] * (
                            1 - self._normalize_feature(features['avg_max_attention'], 0, 1)) +
                self.feature_weights['cross_layer'] * self._normalize_feature(features['cross_layer_consistency'], 0, 1)
        )

        return {
            'complexity_score': min(complexity_score, 1.0),
            'features': features,
            'is_complex': complexity_score > 0.5,
            # 添加兼容字段
            'avg_entropy': features['avg_token_entropy'],
            'head_entropies': [features['avg_token_entropy']] * 8  # 假设8个头
        }

    # 🔧 添加兼容方法
    def predict_complexity(self, attention_weights):
        """兼容原始方法的简化版本"""
        # 简单计算熵值
        entropies = []
        for head_attn in attention_weights:
            head_attn = head_attn + 1e-9
            entropy = -torch.sum(head_attn * torch.log(head_attn))
            entropies.append(entropy.item())

        avg_entropy = np.mean(entropies)
        complexity_score = min(avg_entropy / 2.0, 1.0)

        return {
            'complexity_score': complexity_score,
            'avg_entropy': avg_entropy,
            'head_entropies': entropies,
            'is_complex': complexity_score > 0.5
        }

    def _normalize_feature(self, value, min_val, max_val):
        """将特征值归一化到[0,1]"""
        return max(0, min(1, (value - min_val) / (max_val - min_val)))


class ComprehensiveBaselineExperiment:
    def __init__(self, model_name="microsoft/DialoGPT-small", use_enhanced=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,
            attn_implementation="eager"  # 修复警告
        )
        self.analyzer = EnhancedAttentionAnalyzer()
        self.use_enhanced = use_enhanced

        # Set pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Extended test dataset
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
        """Extract attention features"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # Get last layer attention
        last_attention = outputs.attentions[-1][0]  # [num_heads, seq_len, seq_len]

        if method == "last_token":
            # Analyze attention pattern of the last valid token
            seq_len = inputs['attention_mask'].sum().item()
            last_token_attn = last_attention[:, seq_len - 1, :]  # [num_heads, seq_len]
            return last_token_attn

        elif method == "average":
            # Average attention across all tokens
            return last_attention

        else:
            raise ValueError(f"Unknown method: {method}")

    def run_single_task(self, text, complexity_label):
        """Run single task with option for enhanced or basic analysis"""
        if self.use_enhanced:
            # 使用增强版分析
            features = self.analyzer.extract_multi_dimensional_features(text, self.model, self.tokenizer)
            result = self.analyzer.predict_complexity_enhanced(features)
        else:
            # 使用基础版分析
            attention_weights = self.extract_attention_features(text)
            result = self.analyzer.predict_complexity(attention_weights)

        # Add true label
        result['true_complexity'] = complexity_label
        result['task'] = text

        return result

    def run_comprehensive_experiment(self):
        """Run comprehensive experiment"""
        method_name = "Enhanced Multi-dimensional" if self.use_enhanced else "Basic Entropy"
        print(f"🧪 Running {method_name} Stage 1 experiment...")

        all_results = []

        # Run all tasks
        for complexity, tasks in self.task_dataset.items():
            print(f"\n📊 Processing {complexity} tasks...")

            for task in tasks:
                result = self.run_single_task(task, complexity)
                all_results.append(result)
                print(f"'{task[:50]}...' -> complexity={result['complexity_score']:.3f}")

        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_results)

        return self.analyze_results(df)

    def analyze_results(self, df):
        """Comprehensive result analysis"""
        method_name = "Enhanced Multi-dimensional" if self.use_enhanced else "Basic Entropy"
        print("\n" + "=" * 60)
        print(f"📈 {method_name.upper()} RESULTS ANALYSIS")
        print("=" * 60)

        # 1. Descriptive statistics
        summary = df.groupby('true_complexity')['complexity_score'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(3)
        print("\n1. Descriptive Statistics:")
        print(summary)

        # 2. Visualization
        self.plot_results(df)

        # 3. Statistical tests
        self.statistical_tests(df)

        # 4. Correlation analysis
        self.correlation_analysis(df)

        # 5. Routing decision analysis
        self.routing_analysis(df)

        return df

    def plot_results(self, df):
        """Result visualization"""
        plt.figure(figsize=(15, 5))

        # Box plot for complexity scores
        plt.subplot(1, 3, 1)
        df.boxplot(column='complexity_score', by='true_complexity', ax=plt.gca())
        plt.title('Complexity Score Distribution')
        plt.ylabel('Complexity Score')

        # Box plot for average entropy
        plt.subplot(1, 3, 2)
        df.boxplot(column='avg_entropy', by='true_complexity', ax=plt.gca())
        plt.title('Average Attention Entropy Distribution')
        plt.ylabel('Average Entropy')

        # Scatter plot
        plt.subplot(1, 3, 3)
        complexity_mapping = {'simple': 1, 'medium': 2, 'complex': 3}
        df['complexity_numeric'] = df['true_complexity'].map(complexity_mapping)
        plt.scatter(df['complexity_numeric'], df['complexity_score'], alpha=0.6)
        plt.xlabel('True Complexity')
        plt.ylabel('Predicted Complexity Score')
        plt.title('True vs Predicted Complexity')

        plt.tight_layout()
        plt.show()

    def statistical_tests(self, df):
        """Statistical significance testing"""
        print("\n2. Statistical Significance Tests:")

        simple_scores = df[df['true_complexity'] == 'simple']['complexity_score']
        medium_scores = df[df['true_complexity'] == 'medium']['complexity_score']
        complex_scores = df[df['true_complexity'] == 'complex']['complexity_score']

        # ANOVA test
        f_stat, p_value = stats.f_oneway(simple_scores, medium_scores, complex_scores)
        print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")

        if p_value < 0.05:
            print("✅ Significant difference between groups (p < 0.05)")
        else:
            print("❌ No significant difference between groups (p >= 0.05)")

        # Pairwise comparison
        from scipy.stats import ttest_ind
        t1, p1 = ttest_ind(simple_scores, complex_scores)
        print(f"Simple vs Complex tasks t-test: t={t1:.3f}, p={p1:.4f}")

    def correlation_analysis(self, df):
        """Correlation analysis"""
        print("\n3. Correlation Analysis:")

        complexity_mapping = {'simple': 1, 'medium': 2, 'complex': 3}
        df['complexity_numeric'] = df['true_complexity'].map(complexity_mapping)

        correlation = df['complexity_score'].corr(df['complexity_numeric'])
        print(f"Correlation between complexity score and true complexity: {correlation:.4f}")

        if correlation > 0.5:
            print("✅ Strong positive correlation - Hypothesis validated!")
        elif correlation > 0.3:
            print("⚠️ Moderate correlation - Some effectiveness but needs improvement")
        else:
            print("❌ Weak correlation - Need to reconsider methodology")

    def routing_analysis(self, df):
        """Routing decision analysis"""
        print("\n4. Routing Decision Analysis:")

        # Calculate routing decisions for each complexity level
        routing_stats = df.groupby('true_complexity')['is_complex'].agg([
            'count', 'sum', lambda x: (x.sum() / len(x) * 100)
        ]).round(1)
        routing_stats.columns = ['Total', 'Routed to Cloud', 'Routing Rate (%)']
        print(routing_stats)

        # Ideal case: simple tasks not routed, complex tasks routed
        simple_correct = (df[df['true_complexity'] == 'simple']['is_complex'] == False).sum()
        complex_correct = (df[df['true_complexity'] == 'complex']['is_complex'] == True).sum()

        simple_total = len(df[df['true_complexity'] == 'simple'])
        complex_total = len(df[df['true_complexity'] == 'complex'])

        print(f"\nRouting Accuracy:")
        print(f"Simple task correct routing rate: {simple_correct / simple_total * 100:.1f}%")
        print(f"Complex task correct routing rate: {complex_correct / complex_total * 100:.1f}%")

    def save_results(self, df, filename="stage1_results.csv"):
        """Save results"""
        df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to {filename}")


def run_comparison_experiment():
    """运行对比实验：基础版 vs 增强版"""
    print("🔬 Running Comparison Experiment: Basic vs Enhanced")
    print("=" * 60)

    # 基础版实验
    print("\n🔵 Running Basic Entropy Method...")
    experiment_basic = ComprehensiveBaselineExperiment(use_enhanced=False)
    results_basic = experiment_basic.run_comprehensive_experiment()

    print("\n" + "=" * 60)

    # 增强版实验
    print("\n🟢 Running Enhanced Multi-dimensional Method...")
    experiment_enhanced = ComprehensiveBaselineExperiment(use_enhanced=True)
    results_enhanced = experiment_enhanced.run_comprehensive_experiment()

    # 保存结果
    experiment_basic.save_results(results_basic, "basic_method_results.csv")
    experiment_enhanced.save_results(results_enhanced, "enhanced_method_results.csv")

    return results_basic, results_enhanced


if __name__ == "__main__":
    # 选择运行模式
    run_mode = "enhanced"  # "basic", "enhanced", "comparison"

    if run_mode == "basic":
        experiment = ComprehensiveBaselineExperiment(use_enhanced=False)
        results_df = experiment.run_comprehensive_experiment()
        experiment.save_results(results_df, "basic_results.csv")

    elif run_mode == "enhanced":
        experiment = ComprehensiveBaselineExperiment(use_enhanced=True)
        results_df = experiment.run_comprehensive_experiment()
        experiment.save_results(results_df, "enhanced_results.csv")

    elif run_mode == "comparison":
        basic_results, enhanced_results = run_comparison_experiment()

    print("\n🎉 Stage 1 experiment completed!")