"""
完整的GSM8K智能路由准确率评估系统
整合真实GSM8K数据集、注意力机制复杂度分析器和模型接口
核心目标：验证SLM/LLM的真实准确率，确保路由决策的可靠性
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import re
import json
import os
import warnings
import getpass
from typing import Dict, List, Tuple, Optional
import random

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# GPU设置
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("💻 Using CPU")


# ========================= 基础配置类 =========================
class ModelConfig:
    """模型配置类"""

    def __init__(self, name: str, model_path: str, cost_per_token: float, avg_latency_ms: int):
        self.name = name
        self.model_path = model_path
        self.cost_per_token = cost_per_token
        self.avg_latency_ms = avg_latency_ms


class ModelInterface:
    """模型接口基类"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def predict(self, question: str) -> str:
        raise NotImplementedError


# ========================= GSM8K数据处理器 =========================
class FixedGSM8KProcessor:
    """修复版GSM8K数据处理器"""

    def __init__(self, data_path="gsm8k_data/train.jsonl", max_samples=1000):
        print(f"📚 Loading GSM8K dataset...")
        self.data_path = data_path
        self.max_samples = max_samples
        self.samples = []

        # 尝试多种加载方式
        if self._load_from_datasets():
            print(f"✅ Loaded {len(self.samples)} samples from datasets library")
        elif self._load_from_local():
            print(f"✅ Loaded {len(self.samples)} samples from local file")
        else:
            print("🔄 Using enhanced fallback data...")
            self.samples = self._create_balanced_fallback()

    def _load_from_datasets(self):
        """从datasets库加载GSM8K"""
        try:
            from datasets import load_dataset
            print("🔄 Loading from HuggingFace datasets...")

            # 使用GSM8K的test集，因为它有标准答案
            dataset = load_dataset("gsm8k", "main")
            test_data = dataset['test']

            for i in range(min(self.max_samples, len(test_data))):
                self.samples.append({
                    'question': test_data[i]['question'],
                    'answer': test_data[i]['answer']
                })
            return len(self.samples) > 0
        except Exception as e:
            print(f"⚠️ Failed to load from datasets: {e}")
            return False

    def _load_from_local(self):
        """从本地文件加载"""
        try:
            if not os.path.exists(self.data_path):
                return False

            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            self.samples.append({
                                'question': data['question'],
                                'answer': data['answer']
                            })
                            if len(self.samples) >= self.max_samples:
                                break
                        except:
                            continue
            return len(self.samples) > 0
        except:
            return False

    def _create_balanced_fallback(self):
        """创建平衡的后备数据集"""
        fallback_data = [
            # 简单问题
            {
                'question': "Janet has 3 apples. She eats 1 apple. How many apples does she have left?",
                'answer': "Janet starts with 3 apples.\nShe eats 1 apple.\n3 - 1 = 2\n#### 2"
            },
            {
                'question': "Tom bought 5 books for $2 each. How much did he spend in total?",
                'answer': "Tom bought 5 books.\nEach book costs $2.\n5 × 2 = 10\n#### 10"
            },
            {
                'question': "There are 8 students in a class. If 3 students are absent, how many are present?",
                'answer': "Total students: 8\nAbsent students: 3\nPresent students: 8 - 3 = 5\n#### 5"
            },
            # 中等难度问题
            {
                'question': "Sarah has twice as many stickers as Tom. Tom has 12 stickers. How many stickers do they have together?",
                'answer': "Tom has 12 stickers.\nSarah has twice as many as Tom: 2 × 12 = 24 stickers.\nTogether they have: 12 + 24 = 36 stickers.\n#### 36"
            },
            {
                'question': "A store sells apples for $3 per kg. If John buys 2.5 kg and pays with a $10 bill, how much change does he get?",
                'answer': "Cost per kg: $3\nAmount bought: 2.5 kg\nTotal cost: 3 × 2.5 = $7.50\nPaid: $10\nChange: 10 - 7.50 = $2.50\n#### 2.5"
            },
            # 复杂问题
            {
                'question': "A company has 120 employees. 25% work in sales, 30% in engineering, and the rest in administration. If the sales team gets a 15% increase and engineering gets a 10% increase, how many total employees will there be after the increases?",
                'answer': "Initial employees: 120\nSales: 25% of 120 = 0.25 × 120 = 30 employees\nEngineering: 30% of 120 = 0.30 × 120 = 36 employees\nAdministration: 120 - 30 - 36 = 54 employees\n\nAfter increases:\nSales increase: 30 × 0.15 = 4.5 ≈ 5 new employees\nEngineering increase: 36 × 0.10 = 3.6 ≈ 4 new employees\n\nTotal after increases: 120 + 5 + 4 = 129 employees\n#### 129"
            }
        ]
        print(f"✅ Created {len(fallback_data)} fallback samples")
        return fallback_data

    def extract_answer(self, answer_text: str) -> str:
        """从GSM8K的答案文本中提取数值"""
        # GSM8K标准格式: "#### 42"
        match = re.search(r'####\s*([+-]?\d+(?:\.\d+)?)', answer_text)
        if match:
            return match.group(1)

        # 备用方案：提取最后一个数字
        numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', answer_text)
        if numbers:
            return numbers[-1]

        return "No answer found"

    def count_solution_steps(self, answer: str) -> int:
        """多维度步骤识别综合判断推理复杂度"""
        # 方法1: 计算行数（每行一个逻辑步骤）
        lines = [line.strip() for line in answer.split('\n') if line.strip()]
        meaningful_lines = [line for line in lines if not line.startswith('####')]

        # 方法2: 数学运算计数
        math_operations = len(re.findall(r'\d+\s*[+\-×÷*/]\s*\d+', answer))

        # 方法3: 等号计数
        equals_count = answer.count('=')

        # 方法4: 步骤标志词
        step_words = len(re.findall(r'\b(then|next|so|therefore|thus|after|now|finally)\b', answer.lower()))

        # 综合判断
        steps = max(len(meaningful_lines) - 1, math_operations, equals_count, step_words, 1)
        return min(steps, 12)

    def classify_difficulty(self, steps: int) -> str:
        """修复的难度分级 - 适应真实GSM8K分布"""
        if steps <= 4:
            return "simple"  # 放宽simple标准
        elif steps <= 8:
            return "medium"  # 调整medium范围
        else:
            return "complex"  # 8+步为complex

    def get_balanced_sample(self, n_total: int = 200, simple_ratio: float = 0.5) -> Tuple[List, List]:
        """获取平衡的样本数据，按步骤数分类"""
        print(f"🎯 准备采样 {n_total} 道题目 (简单题比例: {simple_ratio:.1%})")

        simple_problems = []
        complex_problems = []

        print("📋 正在分析问题复杂度...")
        for i, item in enumerate(self.samples):
            if i % 100 == 0 and i > 0:
                print(f"   处理进度: {i}/{len(self.samples)}")

            question = item['question']
            answer_text = item['answer']
            answer = self.extract_answer(answer_text)
            steps = self.count_solution_steps(answer_text)
            difficulty = self.classify_difficulty(steps)

            problem_data = {
                'question': question,
                'answer': answer,
                'original_answer': answer_text,
                'steps': steps,
                'difficulty': difficulty
            }

            if difficulty == "simple":
                simple_problems.append(problem_data)
            else:  # medium 和 complex 都归为复杂问题，简化为二分类
                complex_problems.append(problem_data)

        print(f"✅ 分类完成: {len(simple_problems)} 简单题, {len(complex_problems)} 复杂题")

        # 计算采样数量
        n_simple = int(n_total * simple_ratio)
        n_complex = n_total - n_simple

        # 检查数据充足性
        if len(simple_problems) < n_simple:
            print(f"⚠️ 简单题不足: 需要{n_simple}, 实际{len(simple_problems)}")
            n_simple = len(simple_problems)

        if len(complex_problems) < n_complex:
            print(f"⚠️ 复杂题不足: 需要{n_complex}, 实际{len(complex_problems)}")
            n_complex = len(complex_problems)

        # 随机采样
        if n_simple > 0:
            sampled_simple = random.sample(simple_problems, n_simple)
        else:
            sampled_simple = []

        if n_complex > 0:
            sampled_complex = random.sample(complex_problems, n_complex)
        else:
            sampled_complex = []

        print(f"🎲 最终采样: {len(sampled_simple)} 简单题, {len(sampled_complex)} 复杂题")

        return sampled_simple, sampled_complex


# ========================= 复杂度路由器（注意力机制） =========================
class RobustAttentionAnalyzer:
    """稳健的注意力分析器 - 作为复杂度路由器使用"""

    def __init__(self, device, threshold=0.160):
        self.feature_weights = {
            'entropy': 0.5,
            'variance': 0.35,
            'concentration': 0.15
        }
        self.threshold = threshold
        self.device = device
        print(f"🎯 Attention Analyzer initialized - Threshold: {self.threshold}")

    def route(self, question: str, model, tokenizer) -> str:
        """路由决策：返回'SLM'或'LLM'"""
        try:
            features = self.extract_core_features(question, model, tokenizer)
            prediction = self.predict_complexity(features)

            if prediction['is_complex']:
                return "LLM"
            else:
                return "SLM"
        except Exception as e:
            print(f"⚠️ Route decision failed: {e}, defaulting to LLM")
            return "LLM"  # 出错时默认用大模型

    def extract_core_features(self, text: str, model, tokenizer) -> dict:
        """稳健的特征提取"""
        # 获取模型设备
        model_device = next(model.parameters()).device

        # 编码输入
        inputs = tokenizer(text, return_tensors="pt", max_length=200, truncation=True, padding=True)

        # 确保设备一致性
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        # 获取注意力
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attentions = outputs.attentions[-1][0]
        seq_len = inputs['attention_mask'].sum().item()
        attentions = attentions.cpu()

        # 计算特征
        entropy_features = self._compute_entropy(attentions, seq_len)
        variance_features = self._compute_variance(attentions, seq_len)
        concentration_features = self._compute_concentration(attentions, seq_len)

        return {**entropy_features, **variance_features, **concentration_features}

    def _compute_entropy(self, attentions, seq_len):
        """计算熵特征"""
        all_entropies = []

        for head in range(attentions.shape[0]):
            for pos in range(seq_len):
                attn_dist = attentions[head, pos, :seq_len] + 1e-9
                entropy = -torch.sum(attn_dist * torch.log(attn_dist)).item()
                all_entropies.append(entropy)

        return {
            'avg_entropy': np.mean(all_entropies),
            'entropy_std': np.std(all_entropies),
            'max_entropy': np.max(all_entropies)
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
            'variance_std': np.std(variances),
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
        # 标准化特征
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


# ========================= SLM接口 =========================
class SLMInterface(ModelInterface):
    """小模型接口"""

    def load_model(self):
        """加载小模型"""
        print(f"🔄 Loading SLM: {self.config.name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                output_attentions=True  # 需要注意力用于路由器
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"✅ SLM loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load SLM: {e}")
            raise

    def predict(self, question: str) -> str:
        """SLM预测"""
        if self.model is None:
            self.load_model()

        prompt = f"Question: {question}\nAnswer: Let me solve this step by step.\n"

        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
        if torch.cuda.is_available():
            inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response[len(prompt):].strip()
        return answer


# ========================= LLM接口 =========================
class EnhancedLLMInterface(ModelInterface):
    """增强的大型语言模型接口"""

    def __init__(self, config: ModelConfig, hf_token: str = None):
        super().__init__(config)
        self.hf_token = hf_token
        self.max_memory_per_gpu = "35GB"

    def setup_authentication(self):
        """设置HuggingFace认证"""
        if self.hf_token:
            login(token=self.hf_token)
        elif os.getenv('HUGGINGFACE_TOKEN'):
            login(token=os.getenv('HUGGINGFACE_TOKEN'))
        else:
            print("⚠️ No HuggingFace token provided. Trying without authentication...")

    def load_model(self):
        """加载LLM模型"""
        print(f"🔄 Loading LLM: {self.config.name}")
        self.setup_authentication()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )

            model_kwargs = {
                "torch_dtype": torch.float16,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            if "70B" in self.config.model_path:
                model_kwargs.update({
                    "device_map": "auto",
                    "max_memory": {0: self.max_memory_per_gpu},
                    "load_in_8bit": True,
                })
            elif "34B" in self.config.model_path or "30B" in self.config.model_path:
                model_kwargs.update({
                    "device_map": "auto",
                    "max_memory": {0: self.max_memory_per_gpu},
                })
            else:
                model_kwargs.update({
                    "device_map": "auto" if torch.cuda.is_available() else None,
                })

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                **model_kwargs
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"✅ LLM loaded successfully on {next(self.model.parameters()).device}")

        except Exception as e:
            print(f"❌ Failed to load LLM: {e}")
            print("💡 Suggestions:")
            print("   1. Check if you have HuggingFace access to the model")
            print("   2. Verify your GPU memory is sufficient")
            print("   3. Try a smaller model variant")
            raise

    def predict(self, question: str) -> str:
        """LLM预测答案"""
        if self.model is None:
            self.load_model()

        prompt = self._build_math_prompt(question)
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=2048, truncation=True)
        inputs = inputs.to(next(self.model.parameters()).device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response[len(prompt):].strip()
        return answer

    def _build_math_prompt(self, question: str) -> str:
        """构建数学问题的提示"""
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful math tutor. Solve the following math problem step by step and provide the final numerical answer at the end.

<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I'll solve this step by step.

"""
        return prompt


# ========================= 准确率验证器 =========================
class AccuracyValidator:
    """准确率验证器"""

    @staticmethod
    def extract_final_answer(response: str) -> str:
        """从模型回答中提取最终数值答案"""
        patterns = [
            r'[Tt]he answer is\s*([+-]?\d+(?:\.\d+)?)',
            r'[Ff]inal answer:\s*([+-]?\d+(?:\.\d+)?)',
            r'[Aa]nswer:\s*([+-]?\d+(?:\.\d+)?)',
            r'=\s*([+-]?\d+(?:\.\d+)?)\s*$',
            r'####\s*([+-]?\d+(?:\.\d+)?)',
            r'([+-]?\d+(?:\.\d+)?)\s*$',
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip()

        numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', response)
        if numbers:
            return numbers[-1]

        return "No answer found"

    @staticmethod
    def is_correct(predicted: str, ground_truth: str, tolerance: float = 0.01) -> bool:
        """判断答案是否正确"""
        try:
            pred_num = float(predicted)
            true_num = float(ground_truth)

            if abs(pred_num - true_num) <= tolerance:
                return True

            if true_num != 0:
                relative_error = abs(pred_num - true_num) / abs(true_num)
                return relative_error <= 0.01

            return False
        except (ValueError, TypeError):
            return str(predicted).strip().lower() == str(ground_truth).strip().lower()


# ========================= 主评估器 =========================
class GSM8KAccuracyEvaluator:
    """基于GSM8K的准确率评估器"""

    def __init__(self, hf_token=None, max_samples=1000):
        self.hf_token = hf_token
        self.validator = AccuracyValidator()
        self.data_processor = FixedGSM8KProcessor(max_samples=max_samples)

        # 配置模型
        self.slm_config = ModelConfig(
            name="Llama-3.2-3B",
            model_path="meta-llama/Llama-3.2-3B",
            cost_per_token=0.001,
            avg_latency_ms=100
        )

        self.llm_config = ModelConfig(
            name="Llama-3.1-8B-Instruct",
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            cost_per_token=0.003,
            avg_latency_ms=800
        )

        # 初始化模型接口
        self.slm = SLMInterface(self.slm_config)
        self.llm = EnhancedLLMInterface(self.llm_config, hf_token)

        # 初始化路由器 - 等SLM加载后再初始化
        self.router = None

        print(f"🎯 SLM: {self.slm_config.name}")
        print(f"🎯 LLM: {self.llm_config.name}")
        print(f"📊 成本比例: 1:{self.llm_config.cost_per_token / self.slm_config.cost_per_token:.1f}")

    def _ensure_slm_loaded(self):
        """确保SLM已加载（路由器需要用到）"""
        if self.slm.model is None:
            self.slm.load_model()

        if self.router is None:
            self.router = RobustAttentionAnalyzer(device, threshold=0.16)
            print("✅ 注意力路由器已初始化")

    def evaluate_model_on_problems(self, model_interface, problems: List[Dict],
                                   model_name: str, max_problems: Optional[int] = None) -> Dict:
        """在指定问题集合上评估模型"""
        print(f"\n🔍 评估 {model_name}...")

        if max_problems and len(problems) > max_problems:
            problems = random.sample(problems, max_problems)
            print(f"   随机选择 {max_problems} 道题进行测试")

        correct_count = 0
        total_count = len(problems)
        detailed_results = []
        error_cases = []

        for i, problem in enumerate(problems):
            question = problem['question']
            ground_truth = problem['answer']

            print(f"   处理问题 {i + 1}/{total_count}...")

            try:
                # 获取模型预测
                response = model_interface.predict(question)
                predicted_answer = self.validator.extract_final_answer(response)

                # 验证准确性
                is_correct = self.validator.is_correct(predicted_answer, ground_truth)

                if is_correct:
                    correct_count += 1
                else:
                    # 记录错误案例
                    error_cases.append({
                        'question': question[:100] + "..." if len(question) > 100 else question,
                        'ground_truth': ground_truth,
                        'predicted': predicted_answer,
                        'full_response': response[:200] + "..." if len(response) > 200 else response
                    })

                detailed_results.append({
                    'question': question,
                    'ground_truth': ground_truth,
                    'model_response': response,
                    'extracted_answer': predicted_answer,
                    'is_correct': is_correct
                })

                # 实时显示错误（但不要太频繁）
                if not is_correct and len(error_cases) <= 3:
                    print(f"   ❌ 错误: 预测={predicted_answer}, 正确={ground_truth}")

            except Exception as e:
                print(f"   ⚠️ 处理错误: {e}")
                detailed_results.append({
                    'question': question,
                    'ground_truth': ground_truth,
                    'model_response': f"ERROR: {e}",
                    'extracted_answer': "ERROR",
                    'is_correct': False
                })

        accuracy = correct_count / total_count if total_count > 0 else 0

        result = {
            'model_name': model_name,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': total_count,
            'error_rate': 1 - accuracy,
            'error_cases': error_cases[:10],
            'detailed_results': detailed_results
        }

        print(f"✅ {model_name} 准确率: {accuracy:.2%} ({correct_count}/{total_count})")
        if len(error_cases) > 0:
            print(f"   错误案例数: {len(error_cases)}")

        return result

    def evaluate_routing_accuracy(self, simple_problems: List, complex_problems: List) -> Dict:
        """评估路由准确性"""
        print(f"\n🧭 评估路由准确性...")

        # 确保SLM已加载（路由器需要使用）
        self._ensure_slm_loaded()

        correct_routes = 0
        total_routes = 0
        routing_details = []

        # 测试简单问题的路由（应该路由到SLM）
        test_simple = simple_problems[:min(50, len(simple_problems))]
        for problem in test_simple:
            question = problem['question']
            try:
                predicted_route = self.router.route(question, self.slm.model, self.slm.tokenizer)
                expected_route = "SLM"

                is_correct = (predicted_route == expected_route)
                if is_correct:
                    correct_routes += 1

                routing_details.append({
                    'question': question[:100] + "..." if len(question) > 100 else question,
                    'true_complexity': 'simple',
                    'expected': expected_route,
                    'predicted': predicted_route,
                    'correct': is_correct
                })
                total_routes += 1
            except Exception as e:
                print(f"   ⚠️ 路由错误: {e}")
                continue

        # 测试复杂问题的路由（应该路由到LLM）
        test_complex = complex_problems[:min(50, len(complex_problems))]
        for problem in test_complex:
            question = problem['question']
            try:
                predicted_route = self.router.route(question, self.slm.model, self.slm.tokenizer)
                expected_route = "LLM"

                is_correct = (predicted_route == expected_route)
                if is_correct:
                    correct_routes += 1

                routing_details.append({
                    'question': question[:100] + "..." if len(question) > 100 else question,
                    'true_complexity': 'complex',
                    'expected': expected_route,
                    'predicted': predicted_route,
                    'correct': is_correct
                })
                total_routes += 1
            except Exception as e:
                print(f"   ⚠️ 路由错误: {e}")
                continue

        routing_accuracy = correct_routes / total_routes if total_routes > 0 else 0

        print(f"✅ 路由准确率: {routing_accuracy:.2%} ({correct_routes}/{total_routes})")

        return {
            'routing_accuracy': routing_accuracy,
            'correct_routes': correct_routes,
            'total_routes': total_routes,
            'routing_details': routing_details
        }

    def run_gsm8k_evaluation(self, n_samples: int = 200, simple_ratio: float = 0.5):
        """运行GSM8K评估"""
        print("🚀 开始基于GSM8K的准确率评估")
        print("=" * 60)

        # 1. 加载和采样数据
        print(f"📊 准备GSM8K数据 (样本数: {n_samples})...")
        try:
            simple_problems, complex_problems = self.data_processor.get_balanced_sample(
                n_total=n_samples, simple_ratio=simple_ratio
            )
        except Exception as e:
            print(f"❌ 数据准备失败: {e}")
            return None

        if len(simple_problems) == 0 and len(complex_problems) == 0:
            print("❌ 没有可用的数据")
            return None

        # 2. 评估SLM在简单问题上的表现
        if len(simple_problems) > 0:
            slm_simple_results = self.evaluate_model_on_problems(
                self.slm, simple_problems, "SLM on Simple Problems"
            )
        else:
            slm_simple_results = {
                'model_name': 'SLM on Simple Problems',
                'accuracy': 0, 'correct_count': 0, 'total_count': 0,
                'error_cases': [], 'detailed_results': []
            }

        # 3. 评估LLM在复杂问题上的表现
        if len(complex_problems) > 0:
            llm_complex_results = self.evaluate_model_on_problems(
                self.llm, complex_problems, "LLM on Complex Problems"
            )
        else:
            llm_complex_results = {
                'model_name': 'LLM on Complex Problems',
                'accuracy': 0, 'correct_count': 0, 'total_count': 0,
                'error_cases': [], 'detailed_results': []
            }

        # 4. 交叉验证：SLM在复杂问题上的表现（验证其局限性）
        if len(complex_problems) > 0:
            slm_complex_results = self.evaluate_model_on_problems(
                self.slm, complex_problems[:min(20, len(complex_problems))],
                "SLM on Complex Problems (验证局限性)"
            )
        else:
            slm_complex_results = {
                'model_name': 'SLM on Complex Problems',
                'accuracy': 0, 'correct_count': 0, 'total_count': 0,
                'error_cases': [], 'detailed_results': []
            }

        # 5. 交叉验证：LLM在简单问题上的表现（验证过度配置）
        if len(simple_problems) > 0:
            llm_simple_results = self.evaluate_model_on_problems(
                self.llm, simple_problems[:min(20, len(simple_problems))],
                "LLM on Simple Problems (验证过度配置)"
            )
        else:
            llm_simple_results = {
                'model_name': 'LLM on Simple Problems',
                'accuracy': 0, 'correct_count': 0, 'total_count': 0,
                'error_cases': [], 'detailed_results': []
            }

        # 6. 评估路由准确性
        if len(simple_problems) > 0 or len(complex_problems) > 0:
            routing_results = self.evaluate_routing_accuracy(simple_problems, complex_problems)
        else:
            routing_results = {
                'routing_accuracy': 0, 'correct_routes': 0, 'total_routes': 0,
                'routing_details': []
            }

        # 7. 计算智能路由系统性能
        smart_routing_results = self._calculate_smart_routing_performance(
            slm_simple_results, llm_complex_results, routing_results, n_samples
        )

        # 8. 生成综合报告
        final_report = self._generate_final_report(
            slm_simple_results, llm_complex_results, slm_complex_results,
            llm_simple_results, routing_results, smart_routing_results, n_samples
        )

        return final_report

    def _calculate_smart_routing_performance(self, slm_results, llm_results, routing_results, n_samples):
        """计算智能路由系统的整体性能"""
        routing_acc = routing_results['routing_accuracy']
        slm_acc = slm_results['accuracy']
        llm_acc = llm_results['accuracy']

        # 估算智能路由的准确率
        if slm_results['total_count'] > 0 and llm_results['total_count'] > 0:
            expected_accuracy = (slm_acc + llm_acc) / 2
        elif slm_results['total_count'] > 0:
            expected_accuracy = slm_acc
        elif llm_results['total_count'] > 0:
            expected_accuracy = llm_acc
        else:
            expected_accuracy = 0.5

        # 考虑路由错误的影响
        estimated_accuracy = expected_accuracy * routing_acc + (1 - routing_acc) * 0.5

        # 成本计算
        simple_ratio = slm_results['total_count'] / max(1, slm_results['total_count'] + llm_results['total_count'])

        smart_routing_cost = (simple_ratio * self.slm_config.cost_per_token +
                              (1 - simple_ratio) * self.llm_config.cost_per_token)
        pure_llm_cost = self.llm_config.cost_per_token
        pure_slm_cost = self.slm_config.cost_per_token

        cost_savings_vs_llm = (pure_llm_cost - smart_routing_cost) / pure_llm_cost
        cost_increase_vs_slm = (smart_routing_cost - pure_slm_cost) / pure_slm_cost

        return {
            'estimated_accuracy': estimated_accuracy,
            'cost_per_problem': smart_routing_cost,
            'cost_savings_vs_llm': cost_savings_vs_llm,
            'cost_increase_vs_slm': cost_increase_vs_slm,
            'baseline_accuracy': expected_accuracy
        }

    def _generate_final_report(self, slm_simple, llm_complex, slm_complex,
                               llm_simple, routing, smart_routing, n_samples):
        """生成最终评估报告"""

        print("\n" + "=" * 70)
        print("📋 GSM8K数据集智能路由准确率评估报告")
        print("=" * 70)
        print(f"📊 评估规模: {n_samples} 道GSM8K真题")

        print(f"\n🎯 核心性能指标:")
        print(
            f"├── SLM在简单问题准确率: {slm_simple['accuracy']:.2%} ({slm_simple['correct_count']}/{slm_simple['total_count']})")
        print(
            f"├── LLM在复杂问题准确率: {llm_complex['accuracy']:.2%} ({llm_complex['correct_count']}/{llm_complex['total_count']})")
        print(
            f"├── 路由判断准确率: {routing['routing_accuracy']:.2%} ({routing['correct_routes']}/{routing['total_routes']})")
        print(f"└── 智能路由预估准确率: {smart_routing['estimated_accuracy']:.2%}")

        print(f"\n📊 交叉验证结果:")
        print(f"├── SLM在复杂问题准确率: {slm_complex['accuracy']:.2%} (证明局限性)")
        print(f"├── LLM在简单问题准确率: {llm_simple['accuracy']:.2%} (证明过度配置)")

        if slm_simple['total_count'] > 0 and llm_simple['total_count'] > 0:
            capability_gap = llm_simple['accuracy'] - slm_simple['accuracy']
            print(f"└── 能力差距: {capability_gap:.2%}")

        print(f"\n💰 成本效益分析:")
        print(f"├── 纯SLM成本: ${self.slm_config.cost_per_token:.4f}/问题")
        print(f"├── 纯LLM成本: ${self.llm_config.cost_per_token:.4f}/问题")
        print(f"├── 智能路由成本: ${smart_routing['cost_per_problem']:.4f}/问题")
        print(f"├── vs LLM节省: {smart_routing['cost_savings_vs_llm']:.1%}")
        print(f"└── vs SLM增加: {smart_routing['cost_increase_vs_slm']:.1%}")

        # 关键判断
        slm_reliable = slm_simple['accuracy'] >= 0.80
        routing_reliable = routing['routing_accuracy'] >= 0.85
        cost_effective = smart_routing['cost_savings_vs_llm'] >= 0.30

        print(f"\n💡 关键判断:")
        print(
            f"├── SLM可靠性: {'✅ 可靠' if slm_reliable else '❌ 不可靠'} ({slm_simple['accuracy']:.1%} {'≥' if slm_reliable else '<'} 80%)")
        print(
            f"├── 路由可靠性: {'✅ 可靠' if routing_reliable else '❌ 不可靠'} ({routing['routing_accuracy']:.1%} {'≥' if routing_reliable else '<'} 85%)")
        print(
            f"└── 成本效益: {'✅ 显著' if cost_effective else '❌ 有限'} ({smart_routing['cost_savings_vs_llm']:.1%} {'≥' if cost_effective else '<'} 30%)")

        # 最终建议
        if slm_reliable and routing_reliable and cost_effective:
            recommendation = "✅ 强烈推荐使用智能路由系统"
            reason = "SLM可靠、路由准确、成本效益显著"
        elif not slm_reliable:
            recommendation = "❌ 不推荐使用 - SLM不可靠"
            reason = f"SLM在简单问题上准确率仅{slm_simple['accuracy']:.1%}，存在过度自信风险"
        elif not routing_reliable:
            recommendation = "❌ 不推荐使用 - 路由不准确"
            reason = f"路由判断准确率仅{routing['routing_accuracy']:.1%}，会导致错误分配"
        else:
            recommendation = "⚠️ 谨慎考虑 - 成本效益有限"
            reason = f"虽然系统可靠，但成本节省仅{smart_routing['cost_savings_vs_llm']:.1%}"

        print(f"\n🎯 最终建议:")
        print(f"└── {recommendation}")
        print(f"    理由: {reason}")

        # 显示关键错误案例
        if len(slm_simple['error_cases']) > 0:
            print(f"\n❌ SLM错误案例分析 (前3个):")
            for i, case in enumerate(slm_simple['error_cases'][:3]):
                print(f"   {i + 1}. 问题: {case['question']}")
                print(f"      预测: {case['predicted']}, 正确: {case['ground_truth']}")

        # 保存结果
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gsm8k_routing_evaluation_{timestamp}.csv"

        # 创建详细结果数据框
        all_results = []
        for result in slm_simple['detailed_results']:
            all_results.append({
                'model': 'SLM',
                'problem_type': 'simple',
                **result
            })
        for result in llm_complex['detailed_results']:
            all_results.append({
                'model': 'LLM',
                'problem_type': 'complex',
                **result
            })

        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(filename, index=False)
            print(f"\n💾 详细结果已保存到: {filename}")

        return {
            'dataset_info': {'name': 'GSM8K', 'sample_size': n_samples},
            'slm_simple_performance': slm_simple,
            'llm_complex_performance': llm_complex,
            'slm_complex_performance': slm_complex,
            'llm_simple_performance': llm_simple,
            'routing_performance': routing,
            'smart_routing_performance': smart_routing,
            'evaluation_summary': {
                'slm_reliable': slm_reliable,
                'routing_reliable': routing_reliable,
                'cost_effective': cost_effective,
                'recommendation': recommendation,
                'reason': reason
            }
        }


# ========================= 主函数和工具函数 =========================
def get_secure_token():
    """安全获取token"""
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if hf_token:
        print("✅ 从环境变量获取token")
        return hf_token

    print("🔑 请输入你的HuggingFace Token:")
    return getpass.getpass("Token: ")


def test_model_access(hf_token):
    """测试token是否可以访问指定的模型"""
    try:
        login(token=hf_token)
        print("✅ HuggingFace token valid")

        # 测试Llama-3.2-3B访问
        print("🔄 Testing Llama-3.2-3B access...")
        llama32_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
        print("✅ Llama-3.2-3B access successful")

        # 测试Llama-3.1-8B访问
        print("🔄 Testing Llama-3.1-8B access...")
        llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        print("✅ Llama-3.1-8B access successful")

        return True

    except Exception as e:
        print(f"❌ Model access test failed: {e}")
        return False


def main():
    """主函数"""
    print("🔬 GSM8K智能路由准确率评估系统")
    print("🎯 核心目标: 验证SLM/LLM的真实可靠性")
    print("=" * 50)

    # 获取token
    hf_token = get_secure_token()
    if not hf_token:
        print("❌ 无法获取token")
        return None

    try:
        # 可选：测试模型访问
        print("\n🔍 测试模型访问权限...")
        if not test_model_access(hf_token):
            print("⚠️ 模型访问测试失败，但将继续尝试实验...")

        # 创建评估器
        print("\n🚀 初始化评估器...")
        evaluator = GSM8KAccuracyEvaluator(hf_token=hf_token, max_samples=1000)

        # 运行准确率评估 - 从小样本开始
        print("\n🧪 开始准确率验证实验...")
        print("📋 实验参数:")
        print("   • 样本数量: 200道题")
        print("   • 简单题比例: 50%")
        print("   • 评估重点: SLM可靠性验证")

        results = evaluator.run_gsm8k_evaluation(n_samples=200, simple_ratio=0.5)

        if results:
            # 输出最终评估结论
            print("\n🎯 最终评估结论:")
            summary = results['evaluation_summary']
            print(f"建议: {summary['recommendation']}")
            print(f"理由: {summary['reason']}")

            # 如果结果良好，提示可以扩大规模
            if summary['slm_reliable'] and summary['routing_reliable']:
                print("\n🚀 系统表现良好！建议:")
                print("   1. 增加样本数量到500-1000进行更全面验证")
                print("   2. 调整路由阈值以优化性能")
                print("   3. 考虑在生产环境中部署")

        return results

    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        hf_token = None  # 清理token


if __name__ == "__main__":
    results = main()