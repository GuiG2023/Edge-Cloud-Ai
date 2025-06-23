"""
Final Local GSM8K Experiment - 完整优化版
使用本地下载的GSM8K数据集，限制100个样本，完整的实验流程
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json
import os
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# GPU优化设置
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024 ** 3} GB")
else:
    device = torch.device('cpu')
    print("💻 Using CPU")


class LocalGSM8KProcessor:
    """本地GSM8K数据处理器"""

    def __init__(self, data_path="gsm8k_data/train.jsonl", max_samples=100):
        print(f"📚 Loading local GSM8K dataset from {data_path}...")
        print(f"📊 Max samples: {max_samples}")
        self.data_path = data_path
        self.max_samples = max_samples
        self.samples = []

        # 尝试加载本地数据
        if self._load_local_data():
            print(f"✅ Successfully loaded {len(self.samples)} real GSM8K samples")
        else:
            print("🔄 Using enhanced fallback data...")
            self.samples = self._create_enhanced_fallback()

    def _load_local_data(self):
        """加载本地GSM8K数据"""
        try:
            # 检查文件是否存在
            if not os.path.exists(self.data_path):
                print(f"❌ File not found: {self.data_path}")
                print("💡 Expected file location: gsm8k_data/train.jsonl")
                return False

            print(f"📂 Found file: {self.data_path}")

            # 读取JSONL文件
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        if line.strip():  # 跳过空行
                            data = json.loads(line)

                            # 验证必要字段
                            if 'question' in data and 'answer' in data:
                                self.samples.append({
                                    'question': data['question'],
                                    'answer': data['answer']
                                })
                            else:
                                print(f"⚠️ Missing required fields at line {line_num}")

                            # 限制样本数量
                            if len(self.samples) >= self.max_samples:
                                print(f"📊 Reached max samples limit: {self.max_samples}")
                                break

                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON decode error at line {line_num}: {e}")
                        continue
                    except Exception as e:
                        print(f"⚠️ Error at line {line_num}: {e}")
                        continue

            print(f"📈 Successfully parsed {len(self.samples)} valid samples")
            return len(self.samples) > 0

        except Exception as e:
            print(f"❌ Error loading local data: {e}")
            return False

    # def _create_enhanced_fallback(self):
    #     """增强版备用数据集"""
    #     print("🔧 Creating fallback dataset with 24 carefully designed samples...")
    #     return [
    #         # Simple (1-2 steps) - 8个样本
    #         {
    #             'question': "Janet has 8 ducks. Each duck lays 2 eggs per day. How many eggs does Janet get per day?",
    #             'answer': "Janet has 8 ducks.\nEach duck lays 2 eggs per day.\n8 × 2 = 16 eggs per day.\n#### 16"
    #         },
    #         {
    #             'question': "Tom has 25 marbles. He gives 7 marbles to his friend. How many marbles does Tom have left?",
    #             'answer': "Tom starts with 25 marbles.\nHe gives away 7 marbles.\n25 - 7 = 18 marbles.\n#### 18"
    #         },
    #         {
    #             'question': "Sarah buys 6 notebooks. Each notebook costs $4. How much does Sarah spend?",
    #             'answer': "Sarah buys 6 notebooks.\nEach costs $4.\n6 × 4 = $24.\n#### 24"
    #         },
    #         {
    #             'question': "A classroom has 30 students. 12 students are absent. How many students are present?",
    #             'answer': "Total students = 30.\nAbsent = 12.\nPresent = 30 - 12 = 18.\n#### 18"
    #         },
    #         {
    #             'question': "Mike reads 4 pages every day. How many pages will he read in 7 days?",
    #             'answer': "Mike reads 4 pages per day.\n7 days = 4 × 7 = 28 pages.\n#### 28"
    #         },
    #         {
    #             'question': "A box contains 24 pencils. If 9 pencils are used, how many remain?",
    #             'answer': "Box has 24 pencils.\n9 are used.\n24 - 9 = 15 pencils remain.\n#### 15"
    #         },
    #         {
    #             'question': "Lisa has $45. She spends $18 on lunch. How much money does she have left?",
    #             'answer': "Lisa starts with $45.\nSpends $18.\n$45 - $18 = $27.\n#### 27"
    #         },
    #         {
    #             'question': "A parking lot has 40 cars. 15 cars leave. How many cars remain?",
    #             'answer': "Parking lot has 40 cars.\n15 cars leave.\n40 - 15 = 25 cars remain.\n#### 25"
    #         },
    #
    #         # Medium (3-4 steps) - 8个样本
    #         {
    #             'question': "John works 7 hours per day and earns $16 per hour. After working 5 days, he spends $150. How much money does John have left?",
    #             'answer': "John earns $16 per hour.\nWorks 7 hours per day.\nDaily earnings = 16 × 7 = $112.\nFor 5 days = 112 × 5 = $560.\nSpends $150.\nMoney left = 560 - 150 = $410.\n#### 410"
    #         },
    #         {
    #             'question': "A school has 480 students. 3/4 of the students are girls. How many boys are there?",
    #             'answer': "Total students = 480.\n3/4 are girls.\nGirls = 480 × 3/4 = 360.\nBoys = 480 - 360 = 120.\n#### 120"
    #         },
    #         {
    #             'question': "A rectangular field is 18 meters long and 12 meters wide. What is the area and perimeter?",
    #             'answer': "Field: 18m × 12m.\nArea = 18 × 12 = 216 m².\nPerimeter = 2(18 + 12) = 2 × 30 = 60 meters.\n#### 216"
    #         },
    #         {
    #             'question': "Emma saves $30 every week for 8 weeks. Then she buys a bicycle that costs $180. How much money does Emma have left?",
    #             'answer': "Emma saves $30 per week.\nFor 8 weeks = 30 × 8 = $240.\nBuys bicycle for $180.\nMoney left = 240 - 180 = $60.\n#### 60"
    #         },
    #         {
    #             'question': "A store sells 60 items at $15 each. The cost to buy these items was $450. What is the profit?",
    #             'answer': "Store sells 60 items at $15 each.\nRevenue = 60 × 15 = $900.\nCost = $450.\nProfit = 900 - 450 = $450.\n#### 450"
    #         },
    #         {
    #             'question': "David has 4 times as many stickers as Anna. If Anna has 18 stickers, how many do they have together?",
    #             'answer': "Anna has 18 stickers.\nDavid has 4 times as many = 4 × 18 = 72 stickers.\nTogether = 18 + 72 = 90 stickers.\n#### 90"
    #         },
    #         {
    #             'question': "A pizza has 16 slices. Tom eats 1/4 and Jerry eats 3/8. How many slices are left?",
    #             'answer': "Pizza has 16 slices.\nTom eats 1/4 = 16 × 1/4 = 4 slices.\nJerry eats 3/8 = 16 × 3/8 = 6 slices.\nTotal eaten = 4 + 6 = 10 slices.\nLeft = 16 - 10 = 6 slices.\n#### 6"
    #         },
    #         {
    #             'question': "A car travels 120 kilometers in 2.5 hours. What is the average speed?",
    #             'answer': "Distance = 120 km.\nTime = 2.5 hours.\nSpeed = Distance ÷ Time = 120 ÷ 2.5 = 48 km/h.\n#### 48"
    #         },
    #
    #         # Complex (5+ steps) - 8个样本
    #         {
    #             'question': "An investment of $8000 grows at 6% compound interest annually. What is the amount after 3 years?",
    #             'answer': "Principal = $8000.\nRate = 6% = 0.06.\nYear 1: 8000 × 1.06 = $8480.\nYear 2: 8480 × 1.06 = $8988.80.\nYear 3: 8988.80 × 1.06 = $9527.73.\n#### 9527.73"
    #         },
    #         {
    #             'question': "Two trains start 350 km apart and travel toward each other. Train A goes 80 km/h, Train B goes 95 km/h. When do they meet?",
    #             'answer': "Distance = 350 km.\nTrain A speed = 80 km/h.\nTrain B speed = 95 km/h.\nCombined speed = 80 + 95 = 175 km/h.\nTime = 350 ÷ 175 = 2 hours.\n#### 2"
    #         },
    #         {
    #             'question': "A company's profit was $120,000. It increased 30% in Q1, then decreased 20% in Q2. What's the Q2 profit?",
    #             'answer': "Initial profit = $120,000.\nQ1 increase = 30%.\nQ1 profit = 120000 × 1.30 = $156,000.\nQ2 decrease = 20%.\nQ2 profit = 156000 × 0.80 = $124,800.\n#### 124800"
    #         },
    #         {
    #             'question': "A pool is 25ft long, 15ft wide, 8ft deep. Cost to fill at $0.04/gallon if 1 cubic foot = 7.5 gallons?",
    #             'answer': "Pool: 25ft × 15ft × 8ft.\nVolume = 25 × 15 × 8 = 3000 cubic feet.\nGallons = 3000 × 7.5 = 22,500 gallons.\nCost = 22,500 × $0.04 = $900.\n#### 900"
    #         },
    #         {
    #             'question': "Store marks up 50% then gives 25% discount. If cost is $80, what's the selling price?",
    #             'answer': "Cost = $80.\nMarkup 50%: 80 × 1.50 = $120.\nDiscount 25%: 120 × 0.75 = $90.\nFinal price = $90.\n#### 90"
    #         },
    #         {
    #             'question': "Worker earns $22/hour regular, $33/hour overtime. Works 40 regular + 6 overtime hours. Total pay?",
    #             'answer': "Regular: $22/hour × 40 hours = $880.\nOvertime: $33/hour × 6 hours = $198.\nTotal = 880 + 198 = $1078.\n#### 1078"
    #         },
    #         {
    #             'question': "Car value $28,000, depreciates 15% yearly. Value after 4 years?",
    #             'answer': "Initial = $28,000.\nYear 1: 28000 × 0.85 = $23,800.\nYear 2: 23800 × 0.85 = $20,230.\nYear 3: 20230 × 0.85 = $17,195.50.\nYear 4: 17195.50 × 0.85 = $14,616.18.\n#### 14616.18"
    #         },
    #         {
    #             'question': "Pipe A fills tank in 4 hours, Pipe B in 6 hours. How long together?",
    #             'answer': "Pipe A rate = 1/4 tank/hour.\nPipe B rate = 1/6 tank/hour.\nCombined = 1/4 + 1/6 = 3/12 + 2/12 = 5/12 tank/hour.\nTime = 1 ÷ (5/12) = 12/5 = 2.4 hours.\n#### 2.4"
    #         }
    #     ]

    def count_solution_steps(self, answer: str) -> int:
        """精确计算解题步骤数"""
        # 方法1: 数学运算符计数
        math_operations = len(re.findall(r'\d+\s*[+\-×÷*/]\s*\d+\s*=', answer))

        # 方法2: 等号计数
        equals_count = answer.count('=')

        # 方法3: 计算步骤行数
        lines = answer.split('\n')
        calc_lines = len([line for line in lines
                          if any(op in line for op in ['=', '+', '-', '×', '÷', '*', '/'])
                          and not line.strip().startswith('#')])

        # 方法4: 逻辑连接词
        logic_words = len(re.findall(r'\b(then|next|so|therefore|thus|after)\b', answer.lower()))

        # 取最大值
        steps = max(math_operations, equals_count, calc_lines, logic_words, 1)
        return min(steps, 10)

    def classify_difficulty(self, steps: int) -> str:
        """基于步骤数分级"""
        if steps <= 2:
            return "simple"
        elif steps <= 4:
            return "medium"
        else:
            return "complex"

    def get_balanced_dataset(self, n_per_class: int = 10) -> pd.DataFrame:
        """获取平衡数据集"""
        print(f"🎯 Creating balanced dataset: {n_per_class} per class...")

        # 处理所有样本
        processed = []
        for i, item in enumerate(self.samples):
            steps = self.count_solution_steps(item['answer'])
            difficulty = self.classify_difficulty(steps)
            processed.append({
                'question': item['question'],
                'answer': item['answer'],
                'steps': steps,
                'difficulty': difficulty,
                'sample_id': i
            })

        # 按难度分组
        df = pd.DataFrame(processed)
        balanced_data = []

        print("\n📊 Dataset composition:")
        for diff in ['simple', 'medium', 'complex']:
            subset = df[df['difficulty'] == diff]
            print(f"  Available {diff}: {len(subset)} samples")

            # 取指定数量的样本
            selected = subset.head(n_per_class)
            balanced_data.extend(selected.to_dict('records'))

            if len(selected) > 0:
                step_range = f"[{min(selected['steps'])}-{max(selected['steps'])}]"
                print(f"  Selected {diff}: {len(selected)} samples {step_range}")
            else:
                print(f"  Selected {diff}: 0 samples (none available)")

        result_df = pd.DataFrame(balanced_data)
        print(f"\n📋 Final balanced dataset: {len(result_df)} samples")
        return result_df


class OptimizedAttentionAnalyzer:
    """优化版注意力分析器"""

    def __init__(self, device):
        # 调整权重和阈值
        self.feature_weights = {
            'entropy': 0.5,
            'variance': 0.35,
            'concentration': 0.15
        }
        self.threshold = 0.3  # 设置合理阈值
        self.device = device
        print(f"🎯 Analyzer initialized - Threshold: {self.threshold}")

    def extract_core_features(self, text: str, model, tokenizer) -> dict:
        """提取核心特征 - 修复GPU设备问题"""

        # 获取模型设备
        model_device = next(model.parameters()).device

        # 编码输入
        inputs = tokenizer(text, return_tensors="pt", max_length=200, truncation=True, padding=True)

        # 🔧 关键修复：将输入移动到模型设备
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        # 获取注意力
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attentions = outputs.attentions[-1][0]  # 最后一层
        seq_len = inputs['attention_mask'].sum().item()

        # 移动到CPU计算，避免GPU内存问题
        attentions = attentions.cpu()

        # 计算特征
        entropy_features = self._compute_entropy(attentions, seq_len)
        variance_features = self._compute_variance(attentions, seq_len)
        concentration_features = self._compute_concentration(attentions, seq_len)

        return {**entropy_features, **variance_features, **concentration_features}

    def _compute_entropy(self, attentions, seq_len):
        """计算熵特征"""
        all_entropies = []
        last_token_entropies = []

        for head in range(attentions.shape[0]):
            for pos in range(seq_len):
                attn_dist = attentions[head, pos, :seq_len] + 1e-9
                entropy = -torch.sum(attn_dist * torch.log(attn_dist)).item()
                all_entropies.append(entropy)

                if pos == seq_len - 1:  # 最后一个token
                    last_token_entropies.append(entropy)

        return {
            'avg_entropy': np.mean(all_entropies),
            'last_token_entropy': np.mean(last_token_entropies),
            'entropy_std': np.std(all_entropies),
            'max_entropy': np.max(all_entropies)
        }

    def _compute_variance(self, attentions, seq_len):
        """计算方差特征"""
        variances = []
        spreads = []

        for head in range(attentions.shape[0]):
            for pos in range(seq_len):
                attn_weights = attentions[head, pos, :seq_len]

                variance = torch.var(attn_weights).item()
                variances.append(variance)

                max_attn = torch.max(attn_weights).item()
                spread = 1 - max_attn
                spreads.append(spread)

        return {
            'avg_variance': np.mean(variances),
            'variance_std': np.std(variances),
            'avg_spread': np.mean(spreads),
            'max_variance': np.max(variances)
        }

    def _compute_concentration(self, attentions, seq_len):
        """计算集中度特征"""
        max_attentions = []
        top3_concentrations = []

        for head in range(attentions.shape[0]):
            for pos in range(seq_len):
                attn_weights = attentions[head, pos, :seq_len]

                max_attn = torch.max(attn_weights).item()
                max_attentions.append(max_attn)

                k = min(3, seq_len)
                top_k = torch.topk(attn_weights, k).values
                top3_sum = torch.sum(top_k).item()
                top3_concentrations.append(top3_sum)

        return {
            'avg_max_attention': np.mean(max_attentions),
            'concentration_std': np.std(max_attentions),
            'avg_top3_concentration': np.mean(top3_concentrations)
        }

    def predict_complexity(self, features: dict) -> dict:
        """预测复杂度"""

        # 调整标准化范围
        entropy_score = self._normalize(features['avg_entropy'], 0.8, 3.5)
        variance_score = self._normalize(features['avg_variance'], 0.0, 0.35)
        concentration_score = self._normalize(features['avg_max_attention'], 0.1, 0.9)

        # 综合评分
        complexity_score = (
                self.feature_weights['entropy'] * entropy_score +
                self.feature_weights['variance'] * variance_score +
                self.feature_weights['concentration'] * (1 - concentration_score)
        )

        complexity_score = np.clip(complexity_score, 0, 1)

        return {
            'complexity_score': complexity_score,
            'is_complex': complexity_score > self.threshold,
            'entropy_score': entropy_score,
            'variance_score': variance_score,
            'concentration_score': concentration_score,
            'raw_features': features
        }

    def _normalize(self, value, min_val, max_val):
        """标准化特征"""
        if max_val <= min_val:
            return 0.0
        normalized = (value - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)


class FinalLocalExperiment:
    """最终本地数据实验类"""

    def __init__(self, model_name="microsoft/DialoGPT-small", data_path="gsm8k_data/train.jsonl", max_samples=100):
        print(f"🚀 Initializing Final Local GSM8K Experiment")
        print(f"📊 Model: {model_name}")
        print(f"📁 Data: {data_path}")
        print(f"📈 Max samples: {max_samples}")

        # 加载模型
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                output_attentions=True,
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                attn_implementation="eager"
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_device = next(self.model.parameters()).device
            print(f"✅ Model loaded on {model_device}")

        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise

        # 初始化组件
        self.processor = LocalGSM8KProcessor(data_path, max_samples)
        self.analyzer = OptimizedAttentionAnalyzer(device)

        # GPU内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 GPU memory cleared")

    def run_experiment(self, n_per_class=10):
        """运行实验"""
        print(f"\n🧪 Running Final Local GSM8K Experiment")
        print(f"📊 Samples per class: {n_per_class}")
        print(f"🎯 Routing threshold: {self.analyzer.threshold}")

        # 准备数据
        data = self.processor.get_balanced_dataset(n_per_class)

        if len(data) == 0:
            print("❌ No data available")
            return pd.DataFrame()

        # 运行实验
        results = []
        total_samples = len(data)

        print(f"\n🔄 Processing {total_samples} samples...")

        for i, row in data.iterrows():
            print(f"\nProcessing {i + 1}/{total_samples}: {row['difficulty']} (steps: {row['steps']})")
            print(f"Question: {row['question'][:80]}...")

            try:
                # 提取特征
                features = self.analyzer.extract_core_features(
                    row['question'], self.model, self.tokenizer
                )

                # 预测复杂度
                prediction = self.analyzer.predict_complexity(features)

                # 记录结果
                result = {
                    'sample_id': row.get('sample_id', i),
                    'question': row['question'][:100] + "...",
                    'true_difficulty': row['difficulty'],
                    'steps': row['steps'],
                    'complexity_score': prediction['complexity_score'],
                    'predicted_complex': prediction['is_complex'],
                    'entropy_score': prediction['entropy_score'],
                    'variance_score': prediction['variance_score'],
                    'concentration_score': prediction['concentration_score'],
                    'raw_entropy': features['avg_entropy'],
                    'raw_variance': features['avg_variance'],
                    'raw_concentration': features['avg_max_attention']
                }

                results.append(result)

                # 显示结果
                routing = "→☁️ LLM" if result['predicted_complex'] else "→💻 SLM"
                print(f"Score: {result['complexity_score']:.3f} {routing}")
                print(
                    f"Features - E:{features['avg_entropy']:.2f}, V:{features['avg_variance']:.3f}, C:{features['avg_max_attention']:.3f}")

                # 内存管理
                if (i + 1) % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"❌ Error processing sample: {e}")
                continue

        if not results:
            print("❌ No results generated")
            return pd.DataFrame()

        # 分析结果
        results_df = pd.DataFrame(results)
        self.analyze_final_results(results_df)

        return results_df

    def analyze_final_results(self, df):
        """分析最终实验结果"""
        print("\n" + "=" * 80)
        print("📈 FINAL LOCAL GSM8K EXPERIMENT RESULTS")
        print("=" * 80)

        # 数据源信息
        is_real_data = len(self.processor.samples) > 30  # 更准确的判断
        data_source = "Real GSM8K" if is_real_data else "Fallback Data"
        print(f"📊 Data Source: {data_source}")
        print(f"📋 Total Samples: {len(df)}")

        # 1. 复杂度分数统计
        print("\n1. 📊 Complexity Score Statistics:")
        stats_summary = df.groupby('true_difficulty')['complexity_score'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(3)
        print(stats_summary)

        # 2. 原始特征统计
        print("\n2. 🔍 Raw Feature Statistics:")
        feature_stats = df.groupby('true_difficulty')[
            ['raw_entropy', 'raw_variance', 'raw_concentration']].mean().round(3)
        print(feature_stats)

        # 3. 模式检验
        means = stats_summary['mean']
        print(f"\n📈 Difficulty Pattern:")
        print(f"Simple: {means['simple']:.3f} | Medium: {means['medium']:.3f} | Complex: {means['complex']:.3f}")

        pattern_check = means['complex'] > means['medium'] > means['simple']
        improvement = means['complex'] > means['simple']

        if pattern_check:
            print("✅ Perfect ascending pattern: Simple < Medium < Complex")
        elif improvement:
            print("⚠️ Partial pattern: Complex > Simple (Medium needs adjustment)")
        else:
            print("❌ Pattern broken")

        # 4. 统计显著性
        try:
            simple_scores = df[df['true_difficulty'] == 'simple']['complexity_score']
            medium_scores = df[df['true_difficulty'] == 'medium']['complexity_score']
            complex_scores = df[df['true_difficulty'] == 'complex']['complexity_score']

            if len(simple_scores) > 0 and len(medium_scores) > 0 and len(complex_scores) > 0:
                f_stat, p_value = stats.f_oneway(simple_scores, medium_scores, complex_scores)
                print(f"\n📊 ANOVA Results:")
                print(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")

                if p_value < 0.001:
                    print("✅ Highly significant (p < 0.001)")
                elif p_value < 0.01:
                    print("✅ Very significant (p < 0.01)")
                elif p_value < 0.05:
                    print("✅ Significant (p < 0.05)")
                else:
                    print("❌ Not significant (p >= 0.05)")
            else:
                print("\n⚠️ Insufficient data for ANOVA test")
                p_value = 1.0
        except Exception as e:
            print(f"\n⚠️ Statistical test error: {e}")
            p_value = 1.0

        # 5. 相关性分析
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

        # 6. 路由分析
        print(f"\n🚦 Routing Analysis:")
        routing_stats = df.groupby('true_difficulty')['predicted_complex'].agg(['count', 'sum', 'mean'])
        routing_stats['routing_rate'] = routing_stats['mean'] * 100

        for diff in ['simple', 'medium', 'complex']:
            if diff in routing_stats.index:
                rate = routing_stats.loc[diff, 'routing_rate']
                count = routing_stats.loc[diff, 'sum']
                total = routing_stats.loc[diff, 'count']
                print(f"  {diff.capitalize()}: {rate:.1f}% → LLM ({count}/{total})")

        # 计算理想路由准确率
        simple_correct = (df[df['true_difficulty'] == 'simple']['predicted_complex'] == False).sum()
        complex_correct = (df[df['true_difficulty'] == 'complex']['predicted_complex'] == True).sum()

        simple_total = len(df[df['true_difficulty'] == 'simple'])
        complex_total = len(df[df['true_difficulty'] == 'complex'])

        if simple_total > 0 and complex_total > 0:
            routing_accuracy = (simple_correct + complex_correct) / (simple_total + complex_total) * 100
            print(f"\n🎯 Routing Accuracy: {routing_accuracy:.1f}%")
            print(f"  Simple → SLM: {simple_correct}/{simple_total} ({simple_correct / simple_total * 100:.1f}%)")
            print(f"  Complex → LLM: {complex_correct}/{complex_total} ({complex_correct / complex_total * 100:.1f}%)")

        # 7. 可视化
        self.create_final_plots(df)

        # 8. 综合评估和建议
        print(f"\n🎯 COMPREHENSIVE EVALUATION:")
        print(f"📊 Data Quality: {data_source}")
        print(f"📈 Pattern: {'✅ Good' if pattern_check else '⚠️ Partial' if improvement else '❌ Poor'}")
        print(f"📊 Significance: {'✅ Yes' if p_value < 0.05 else '❌ No'} (p={p_value:.4f})")
        print(
            f"🔗 Correlation: {'✅ Strong' if correlation > 0.5 else '⚠️ Moderate' if correlation > 0.3 else '❌ Weak'} (r={correlation:.3f})")

        if simple_total > 0 and complex_total > 0:
            print(f"🚦 Routing: {routing_accuracy:.1f}%")

        # 实验成功性评估
        success_score = 0
        if pattern_check or improvement:
            success_score += 1
        if p_value < 0.05:
            success_score += 1
        if correlation > 0.3:
            success_score += 1
        if simple_total > 0 and complex_total > 0 and routing_accuracy > 60:
            success_score += 1

        print(f"\n🏆 EXPERIMENT SUCCESS SCORE: {success_score}/4")

        if success_score >= 3:
            print("🌟 EXCELLENT: Strong validation of attention-based complexity prediction!")
        elif success_score >= 2:
            print("✅ GOOD: Solid evidence supporting the approach")
        elif success_score >= 1:
            print("⚠️ MODERATE: Some validation, needs optimization")
        else:
            print("❌ NEEDS WORK: Significant methodology adjustment required")

        # 改进建议
        print(f"\n💡 NEXT STEPS:")
        if not is_real_data:
            print("📥 Priority: Use real GSM8K data for more reliable results")
        if correlation < 0.3:
            print("🔧 Optimize: Adjust feature weights and normalization ranges")
        if p_value >= 0.05:
            print("📊 Scale: Increase sample size for statistical power")
        if simple_total > 0 and complex_total > 0 and routing_accuracy < 70:
            print("🎯 Tune: Optimize threshold for better routing decisions")

        print("🚀 Consider: Test with larger models (GPT-2, Llama) for improved performance")

    def create_final_plots(self, df):
        """创建最终可视化"""
        print("\n📈 Creating final comprehensive visualizations...")

        fig, axes = plt.subplots(2, 4, figsize=(20, 12))

        # 1. 复杂度分数分布
        if len(df) > 0:
            df.boxplot(column='complexity_score', by='true_difficulty', ax=axes[0, 0])
            axes[0, 0].set_title('Complexity Score Distribution')
            axes[0, 0].set_xlabel('True Difficulty')
            axes[0, 0].set_ylabel('Complexity Score')

        # 2. 步骤数 vs 复杂度分数
        colors = {'simple': 'blue', 'medium': 'orange', 'complex': 'red'}
        for diff in colors:
            subset = df[df['true_difficulty'] == diff]
            if len(subset) > 0:
                axes[0, 1].scatter(subset['steps'], subset['complexity_score'],
                                   c=colors[diff], label=diff, alpha=0.7, s=50)
        axes[0, 1].set_xlabel('Solution Steps')
        axes[0, 1].set_ylabel('Complexity Score')
        axes[0, 1].set_title('Steps vs Complexity Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 路由决策
        if len(df) > 0:
            routing_data = pd.crosstab(df['true_difficulty'], df['predicted_complex'], normalize='index') * 100
            routing_data.plot(kind='bar', ax=axes[0, 2], color=['lightblue', 'orange'])
            axes[0, 2].set_title('Routing Decisions (%)')
            axes[0, 2].set_xlabel('True Difficulty')
            axes[0, 2].set_ylabel('Percentage')
            axes[0, 2].legend(['SLM', 'LLM'])
            axes[0, 2].tick_params(axis='x', rotation=0)

        # 4. 特征重要性
        feature_cols = ['entropy_score', 'variance_score', 'concentration_score', 'complexity_score']
        available_features = [col for col in feature_cols if col in df.columns and len(df[col].dropna()) > 1]

        if len(available_features) > 1:
            corr_matrix = df[available_features].corr()
            im = axes[0, 3].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[0, 3].set_xticks(range(len(corr_matrix.columns)))
            axes[0, 3].set_yticks(range(len(corr_matrix.columns)))
            axes[0, 3].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            axes[0, 3].set_yticklabels(corr_matrix.columns)
            axes[0, 3].set_title('Feature Correlation Matrix')

            # 添加数值标注
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    axes[0, 3].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                    ha="center", va="center", color="black", fontsize=8)

        # 5-7. 原始特征分布
        feature_names = ['raw_entropy', 'raw_variance', 'raw_concentration']
        feature_titles = ['Entropy Distribution', 'Variance Distribution', 'Concentration Distribution']

        for idx, (feature, title) in enumerate(zip(feature_names, feature_titles)):
            if feature in df.columns:
                for diff in colors:
                    subset = df[df['true_difficulty'] == diff][feature]
                    if len(subset) > 0:
                        axes[1, idx].hist(subset, bins=8, alpha=0.7, label=diff, color=colors[diff])
                axes[1, idx].set_title(title)
                axes[1, idx].set_xlabel(feature.replace('raw_', '').replace('_', ' ').title())
                axes[1, idx].set_ylabel('Frequency')
                axes[1, idx].legend()
                axes[1, idx].grid(True, alpha=0.3)

        # 8. 真实 vs 预测散点图
        if len(df) > 0:
            diff_mapping = {'simple': 1, 'medium': 2, 'complex': 3}
            df['diff_numeric'] = df['true_difficulty'].map(diff_mapping)

            for diff in colors:
                subset = df[df['true_difficulty'] == diff]
                if len(subset) > 0:
                    axes[1, 3].scatter(subset['diff_numeric'], subset['complexity_score'],
                                       c=colors[diff], label=diff, alpha=0.7, s=50)

            # 添加回归线
            if len(df) > 1:
                z = np.polyfit(df['diff_numeric'], df['complexity_score'], 1)
                p = np.poly1d(z)
                axes[1, 3].plot(df['diff_numeric'], p(df['diff_numeric']), "r--", alpha=0.8, linewidth=2)

            axes[1, 3].set_xlabel('True Difficulty (Numeric)')
            axes[1, 3].set_ylabel('Predicted Complexity Score')
            axes[1, 3].set_title('True vs Predicted Difficulty')
            axes[1, 3].legend()
            axes[1, 3].grid(True, alpha=0.3)

        # 设置整体标题
        if len(df) > 0:
            correlation = df['complexity_score'].corr(df['diff_numeric']) if 'diff_numeric' in df.columns else 0
            data_source = "Real GSM8K" if len(self.processor.samples) > 30 else "Fallback"
            fig.suptitle(f'Final Local GSM8K Experiment Results ({data_source}, r={correlation:.3f})', fontsize=16)

        plt.tight_layout()
        plt.show()

    def save_results(self, df, filename="final_local_gsm8k_results.csv"):
        """保存实验结果"""
        if len(df) > 0:
            df.to_csv(filename, index=False)
            print(f"\n💾 Results saved to {filename}")

            # 保存摘要
            summary = {
                'total_samples': len(df),
                'data_source': "Real GSM8K" if len(self.processor.samples) > 30 else "Fallback",
                'correlation': df['complexity_score'].corr(
                    df['true_difficulty'].map({'simple': 1, 'medium': 2, 'complex': 3})),
                'threshold': self.analyzer.threshold,
                'model': 'DialoGPT-small'
            }

            with open('experiment_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"📊 Summary saved to experiment_summary.json")
        else:
            print("❌ No results to save")


def run_final_gsm8k_experiment():
    """运行最终GSM8K实验"""
    print("🎯 Starting Final Local GSM8K Experiment")
    print("=" * 70)

    # 检查数据文件
    data_path = "gsm8k_data/train.jsonl"
    if os.path.exists(data_path):
        print(f"✅ Found GSM8K data: {data_path}")

        # 快速验证文件
        try:
            with open(data_path, 'r') as f:
                first_line = f.readline()
                sample = json.loads(first_line)
                print(f"📝 Sample question: {sample['question'][:60]}...")
        except Exception as e:
            print(f"⚠️ Data file validation warning: {e}")
    else:
        print(f"⚠️ GSM8K data not found at {data_path}")
        print("🔄 Will use fallback data")

    try:
        # 初始化实验
        experiment = FinalLocalExperiment(
            model_name="microsoft/DialoGPT-small",
            data_path=data_path,
            max_samples=100
        )

        # 运行实验
        print(f"\n{'=' * 50}")
        print("🚀 STARTING EXPERIMENT EXECUTION")
        print(f"{'=' * 50}")

        results = experiment.run_experiment(n_per_class=10)

        # 保存结果
        experiment.save_results(results)

        # 最终总结
        if len(results) > 0:
            print(f"\n{'=' * 70}")
            print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
            print(f"{'=' * 70}")

            print(f"\n📋 FINAL SUMMARY:")
            print(f"✅ Total samples processed: {len(results)}")

            data_source = "Real GSM8K" if len(experiment.processor.samples) > 30 else "Fallback Data"
            print(f"✅ Data source: {data_source}")

            correlation = results['complexity_score'].corr(
                results['true_difficulty'].map({'simple': 1, 'medium': 2, 'complex': 3})
            )
            print(f"✅ Correlation achieved: {correlation:.3f}")

            # 路由准确率
            simple_correct = (results[results['true_difficulty'] == 'simple']['predicted_complex'] == False).sum()
            complex_correct = (results[results['true_difficulty'] == 'complex']['predicted_complex'] == True).sum()
            simple_total = len(results[results['true_difficulty'] == 'simple'])
            complex_total = len(results[results['true_difficulty'] == 'complex'])

            if simple_total > 0 and complex_total > 0:
                routing_accuracy = (simple_correct + complex_correct) / (simple_total + complex_total) * 100
                print(f"✅ Routing accuracy: {routing_accuracy:.1f}%")

            print(f"✅ Files saved: final_local_gsm8k_results.csv, experiment_summary.json")

        else:
            print("\n❌ No results generated - check your setup")

        return results

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 运行最终实验
    print("🎯 Final Local GSM8K Experiment - Ready to Run!")
    print("=" * 70)

    results = run_final_gsm8k_experiment()

    if results is not None and len(results) > 0:
        print(f"\n🎊 SUCCESS! Experiment completed with {len(results)} samples")
        print("📂 Check the generated files for detailed results")
    else:
        print("\n⚠️ Experiment had issues - check the error messages above")

    print("\n👋 Experiment session ended")