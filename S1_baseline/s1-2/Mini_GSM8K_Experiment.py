"""
Minimal GSM8K Experiment - 精简核心版本
专注核心功能：真实GSM8K + 客观分级 + 基础特征 + 统计验证
适合：本地调试 → GitHub → Colab运行
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from datasets import load_dataset
import warnings

warnings.filterwarnings('ignore')

# 设置
torch.manual_seed(42)
np.random.seed(42)


class MinimalGSM8KProcessor:
    """精简GSM8K处理器 - 专注核心功能"""

    def __init__(self):
        print("📚 Loading GSM8K dataset...")
        try:
            # 修复数据集加载方式
            dataset = load_dataset("openai/gsm8k", "main")
            # 取训练集的前100个样本
            train_data = dataset['train']
            self.samples = [train_data[i] for i in range(min(100, len(train_data)))]
            print(f"✅ Loaded {len(self.samples)} GSM8K samples")
        except Exception as e:
            print(f"⚠️ GSM8K loading failed: {e}")
            print("🔄 Using minimal fallback...")
            self.samples = self._create_minimal_fallback()

    def _create_minimal_fallback(self):
        """最小备用数据集"""
        return [
            # Simple (1-2 steps)
            {'question': "Tom has 5 apples. He eats 2. How many are left?",
             'answer': "Tom has 5 apples.\nHe eats 2.\n5 - 2 = 3.\n#### 3"},
            {'question': "Sarah buys 4 books for $3 each. How much does she spend?",
             'answer': "4 books at $3 each.\nTotal = 4 × 3 = $12.\n#### 12"},
            {'question': "A box has 15 pencils. 6 are removed. How many remain?",
             'answer': "15 pencils initially.\n6 removed.\n15 - 6 = 9.\n#### 9"},
            {'question': "Lisa walks 2 miles each day. How far in 5 days?",
             'answer': "2 miles per day.\n5 days total.\n2 × 5 = 10 miles.\n#### 10"},

            # Medium (3-4 steps)
            {'question': "John earns $12/hour for 8 hours, then spends $30. Money left?",
             'answer': "$12 per hour.\n8 hours worked.\nEarned = 12 × 8 = $96.\nSpent $30.\nLeft = 96 - 30 = $66.\n#### 66"},
            {'question': "A school has 240 students. 1/3 are girls. How many boys?",
             'answer': "240 students total.\n1/3 are girls.\nGirls = 240 ÷ 3 = 80.\nBoys = 240 - 80 = 160.\n#### 160"},
            {'question': "Rectangle: 12m long, 8m wide. What's the area and perimeter?",
             'answer': "Length = 12m, Width = 8m.\nArea = 12 × 8 = 96 m².\nPerimeter = 2(12 + 8) = 40m.\n#### 96"},
            {'question': "Store sells 25 items at $4 each, costs $60 total. Profit?",
             'answer': "25 items at $4 each.\nRevenue = 25 × 4 = $100.\nCosts = $60.\nProfit = 100 - 60 = $40.\n#### 40"},

            # Complex (5+ steps)
            {'question': "Investment: $1000 at 8% for 2 years compound. Final amount?",
             'answer': "Principal = $1000.\nRate = 8% = 0.08.\nTime = 2 years.\nYear 1: 1000 × 1.08 = $1080.\nYear 2: 1080 × 1.08 = $1166.40.\n#### 1166.40"},
            {'question': "Two cars 180 miles apart drive toward each other at 45mph and 60mph. Meeting time?",
             'answer': "Distance = 180 miles.\nCar A: 45 mph.\nCar B: 60 mph.\nCombined speed = 45 + 60 = 105 mph.\nTime = 180 ÷ 105 = 1.71 hours.\n#### 1.71"},
            {'question': "Company profit $80k, up 25% Q1, down 15% Q2. Final profit?",
             'answer': "Initial = $80,000.\nQ1: up 25%.\nQ1 profit = 80000 × 1.25 = $100,000.\nQ2: down 15%.\nQ2 profit = 100000 × 0.85 = $85,000.\n#### 85000"},
            {'question': "Pool 15ft×10ft×4ft. Gallons if 1 cubic foot = 7.5 gallons?",
             'answer': "Dimensions: 15ft × 10ft × 4ft.\nVolume = 15 × 10 × 4 = 600 cubic feet.\nGallons = 600 × 7.5 = 4500 gallons.\n#### 4500"}
        ]

    def count_solution_steps(self, answer: str) -> int:
        """计算解题步骤数 - 客观方法"""
        # 方法1: 数学表达式计数
        math_ops = len(re.findall(r'\d+\s*[+\-×÷*/]\s*\d+\s*=', answer))

        # 方法2: 等号计数
        equals_count = answer.count('=')

        # 方法3: 计算行数
        calc_lines = len([line for line in answer.split('\n')
                          if any(op in line for op in ['=', '+', '-', '×', '÷', '*', '/'])])

        # 取最大值作为步骤数
        steps = max(math_ops, equals_count, calc_lines, 1)
        return min(steps, 8)  # 限制最大步骤数

    def classify_difficulty(self, steps: int) -> str:
        """客观难度分级"""
        if steps <= 2:
            return "simple"
        elif steps <= 4:
            return "medium"
        else:
            return "complex"

    def get_balanced_dataset(self, n_per_class: int = 8) -> pd.DataFrame:
        """获取平衡数据集"""
        print(f"🎯 Creating balanced dataset: {n_per_class} per class...")

        # 处理所有样本
        processed = []
        for item in self.samples:
            steps = self.count_solution_steps(item['answer'])
            difficulty = self.classify_difficulty(steps)
            processed.append({
                'question': item['question'],
                'answer': item['answer'],
                'steps': steps,
                'difficulty': difficulty
            })

        # 按难度分组
        df = pd.DataFrame(processed)
        balanced_data = []

        for diff in ['simple', 'medium', 'complex']:
            subset = df[df['difficulty'] == diff].head(n_per_class)
            balanced_data.extend(subset.to_dict('records'))
            print(f"  {diff}: {len(subset)} samples")

        return pd.DataFrame(balanced_data)


class CoreAttentionAnalyzer:
    """核心注意力分析器 - 精简版"""

    def __init__(self):
        # 简化权重配置
        self.entropy_weight = 0.6
        self.variance_weight = 0.3
        self.concentration_weight = 0.1
        self.threshold = 0.4

    def extract_core_features(self, text: str, model, tokenizer) -> dict:
        """提取核心特征"""
        # 编码输入
        inputs = tokenizer(text, return_tensors="pt", max_length=200, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 获取注意力
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attentions = outputs.attentions[-1][0]  # 最后一层
        seq_len = inputs['attention_mask'].sum().item()

        # 计算三个核心特征
        entropy_features = self._compute_entropy(attentions, seq_len)
        variance_features = self._compute_variance(attentions, seq_len)
        concentration_features = self._compute_concentration(attentions, seq_len)

        return {**entropy_features, **variance_features, **concentration_features}

    def _compute_entropy(self, attentions, seq_len):
        """计算熵特征"""
        entropies = []

        # 计算所有位置的熵
        for head in range(attentions.shape[0]):
            for pos in range(seq_len):
                attn_dist = attentions[head, pos, :seq_len] + 1e-9
                entropy = -torch.sum(attn_dist * torch.log(attn_dist)).item()
                entropies.append(entropy)

        return {
            'avg_entropy': np.mean(entropies),
            'max_entropy': np.max(entropies),
            'entropy_std': np.std(entropies)
        }

    def _compute_variance(self, attentions, seq_len):
        """计算方差特征"""
        variances = []

        for head in range(attentions.shape[0]):
            for pos in range(seq_len):
                attn_weights = attentions[head, pos, :seq_len]
                variance = torch.var(attn_weights).item()
                variances.append(variance)

        return {
            'avg_variance': np.mean(variances),
            'max_variance': np.max(variances)
        }

    def _compute_concentration(self, attentions, seq_len):
        """计算集中度特征"""
        max_attentions = []

        for head in range(attentions.shape[0]):
            for pos in range(seq_len):
                attn_weights = attentions[head, pos, :seq_len]
                max_attn = torch.max(attn_weights).item()
                max_attentions.append(max_attn)

        return {
            'avg_max_attention': np.mean(max_attentions),
            'concentration_std': np.std(max_attentions)
        }

    def predict_complexity(self, features: dict) -> dict:
        """预测复杂度"""
        # 简单的线性组合
        entropy_score = self._normalize(features['avg_entropy'], 0.5, 3.0)
        variance_score = self._normalize(features['avg_variance'], 0.0, 0.3)
        concentration_score = self._normalize(features['avg_max_attention'], 0.2, 0.9)

        # 综合评分
        complexity_score = (
                self.entropy_weight * entropy_score +
                self.variance_weight * variance_score +
                self.concentration_weight * (1 - concentration_score)  # 反向
        )

        complexity_score = np.clip(complexity_score, 0, 1)

        return {
            'complexity_score': complexity_score,
            'is_complex': complexity_score > self.threshold,
            'entropy_score': entropy_score,
            'variance_score': variance_score,
            'concentration_score': concentration_score
        }

    def _normalize(self, value, min_val, max_val):
        """归一化"""
        if max_val <= min_val:
            return 0.0
        return np.clip((value - min_val) / (max_val - min_val), 0, 1)


class MinimalExperiment:
    """精简实验类"""

    def __init__(self, model_name="microsoft/DialoGPT-small"):
        print(f"🚀 Initializing minimal experiment with {model_name}")

        # 加载模型 (更robust的错误处理)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                output_attentions=True,
                torch_dtype=torch.float32,  # 确保兼容性
                device_map="auto" if torch.cuda.is_available() else None
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise

        # 初始化组件
        self.processor = MinimalGSM8KProcessor()
        self.analyzer = CoreAttentionAnalyzer()

    def run_experiment(self, n_per_class=8):
        """运行核心实验"""
        print(f"\n🧪 Running Minimal GSM8K Experiment")
        print(f"📊 Samples per class: {n_per_class}")
        print(f"🎯 Threshold: {self.analyzer.threshold}")

        # 准备数据
        try:
            data = self.processor.get_balanced_dataset(n_per_class)
            print(f"📋 Total samples: {len(data)}")
        except Exception as e:
            print(f"❌ Data preparation failed: {e}")
            print("🔄 Using fallback data...")
            # 如果数据准备失败，使用fallback数据
            self.processor.samples = self.processor._create_minimal_fallback()
            data = self.processor.get_balanced_dataset(n_per_class)

        # 运行实验
        results = []
        for i, row in data.iterrows():
            print(f"Processing {i + 1}/{len(data)}: {row['difficulty']}")

            try:
                # 提取特征
                features = self.analyzer.extract_core_features(
                    row['question'], self.model, self.tokenizer
                )

                # 预测复杂度
                prediction = self.analyzer.predict_complexity(features)

                # 记录结果
                result = {
                    'question': row['question'][:50] + "...",  # 截断显示
                    'true_difficulty': row['difficulty'],
                    'steps': row['steps'],
                    'complexity_score': prediction['complexity_score'],
                    'predicted_complex': prediction['is_complex'],
                    'entropy_score': prediction['entropy_score'],
                    'variance_score': prediction['variance_score'],
                    'concentration_score': prediction['concentration_score']
                }
                results.append(result)

                # 显示进度
                routing = "→☁️" if result['predicted_complex'] else "→💻"
                print(f"  Score: {result['complexity_score']:.3f} {routing}")

            except Exception as e:
                print(f"  ❌ Error processing sample: {e}")
                continue

        if not results:
            print("❌ No results generated. Check data and model setup.")
            return pd.DataFrame()

        # 分析结果
        results_df = pd.DataFrame(results)
        self.analyze_results(results_df)

        return results_df

    def analyze_results(self, df):
        """分析结果"""
        print("\n" + "=" * 60)
        print("📈 MINIMAL EXPERIMENT RESULTS")
        print("=" * 60)

        # 1. 基础统计
        print("\n1. 📊 Complexity Score Statistics:")
        stats_by_diff = df.groupby('true_difficulty')['complexity_score'].agg([
            'count', 'mean', 'std'
        ]).round(3)
        print(stats_by_diff)

        # 2. 模式检验
        means = stats_by_diff['mean']
        print(f"\n📈 Pattern Check:")
        print(f"Simple: {means['simple']:.3f} | Medium: {means['medium']:.3f} | Complex: {means['complex']:.3f}")

        pattern_ok = means['complex'] > means['medium'] > means['simple']
        print(f"✅ Ascending pattern: {'YES' if pattern_ok else 'NO'}")

        # 3. 统计显著性
        simple = df[df['true_difficulty'] == 'simple']['complexity_score']
        medium = df[df['true_difficulty'] == 'medium']['complexity_score']
        complex_scores = df[df['true_difficulty'] == 'complex']['complexity_score']

        try:
            f_stat, p_value = stats.f_oneway(simple, medium, complex_scores)
            print(f"\n📊 ANOVA Test:")
            print(f"F-statistic: {f_stat:.4f}")
            print(f"p-value: {p_value:.4f}")

            if p_value < 0.001:
                print("✅ Highly significant (p < 0.001)")
            elif p_value < 0.01:
                print("✅ Very significant (p < 0.01)")
            elif p_value < 0.05:
                print("✅ Significant (p < 0.05)")
            else:
                print("❌ Not significant (p >= 0.05)")
        except:
            print("⚠️ Statistical test failed (insufficient data)")
            p_value = 1.0

        # 4. 相关性分析
        diff_mapping = {'simple': 1, 'medium': 2, 'complex': 3}
        df['diff_numeric'] = df['true_difficulty'].map(diff_mapping)
        correlation = df['complexity_score'].corr(df['diff_numeric'])

        print(f"\n🔗 Correlation Analysis:")
        print(f"Correlation: {correlation:.4f}")
        print(f"R²: {correlation ** 2:.3f} ({correlation ** 2 * 100:.1f}%)")

        if correlation > 0.7:
            print("✅ Very strong correlation")
        elif correlation > 0.5:
            print("✅ Strong correlation")
        elif correlation > 0.3:
            print("⚠️ Moderate correlation")
        else:
            print("❌ Weak correlation")

        # 5. 路由性能
        print(f"\n🚦 Routing Analysis:")
        routing_stats = df.groupby('true_difficulty')['predicted_complex'].mean() * 100
        for diff in ['simple', 'medium', 'complex']:
            if diff in routing_stats.index:
                print(f"  {diff.capitalize()}: {routing_stats[diff]:.1f}% → LLM")

        # 6. 简单可视化
        self.create_simple_plots(df)

        # 7. 总结评估
        print(f"\n🎯 SUMMARY:")
        print(f"Pattern: {'✅ Good' if pattern_ok else '❌ Broken'}")
        print(f"Significance: {'✅ Yes' if p_value < 0.05 else '❌ No'}")
        print(f"Correlation: {'✅ Strong' if correlation > 0.5 else '⚠️ Moderate' if correlation > 0.3 else '❌ Weak'}")

        if pattern_ok and p_value < 0.05 and correlation > 0.4:
            print("\n🌟 EXCELLENT: Core validation successful!")
        elif p_value < 0.05 and correlation > 0.3:
            print("\n✅ GOOD: Solid foundation, ready for enhancement")
        else:
            print("\n⚠️ NEEDS WORK: Consider threshold/feature adjustment")

    def create_simple_plots(self, df):
        """创建简单可视化"""
        print("\n📈 Creating plots...")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. 分数分布
        df.boxplot(column='complexity_score', by='true_difficulty', ax=axes[0])
        axes[0].set_title('Complexity Score by Difficulty')

        # 2. 步骤数vs分数
        colors = {'simple': 'blue', 'medium': 'orange', 'complex': 'red'}
        for diff in colors:
            subset = df[df['true_difficulty'] == diff]
            axes[1].scatter(subset['steps'], subset['complexity_score'],
                            c=colors[diff], label=diff, alpha=0.7)
        axes[1].set_xlabel('Solution Steps')
        axes[1].set_ylabel('Complexity Score')
        axes[1].set_title('Steps vs Score')
        axes[1].legend()

        # 3. 路由决策
        routing_data = pd.crosstab(df['true_difficulty'], df['predicted_complex'], normalize='index') * 100
        routing_data.plot(kind='bar', ax=axes[2], color=['lightblue', 'orange'])
        axes[2].set_title('Routing Decisions (%)')
        axes[2].set_xlabel('True Difficulty')
        axes[2].legend(['SLM', 'LLM'])
        axes[2].tick_params(axis='x', rotation=0)

        plt.tight_layout()
        plt.show()


def run_minimal_gsm8k_experiment():
    """主运行函数"""
    print("🎯 Starting Minimal GSM8K Experiment")
    print("=" * 50)

    try:
        # 运行实验
        experiment = MinimalExperiment()
        results = experiment.run_experiment(n_per_class=8)

        # 保存结果
        results.to_csv("minimal_gsm8k_results.csv", index=False)
        print(f"\n💾 Results saved to minimal_gsm8k_results.csv")

        print("\n🎉 Minimal experiment completed successfully!")
        return results

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_minimal_gsm8k_experiment()