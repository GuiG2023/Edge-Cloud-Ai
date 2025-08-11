# coding=utf-8
"""
完整的GSM8K智能路由准确率评估系统
整合了真实GSM8K数据集、一个可学习的复杂度分析器和模型接口。
本文件包含两种模式:
1. 'train' 模式: 生成训练数据并训练路由器模型。
2. 'eval' 模式: 加载训练好的路由器模型，并进行端到端的性能评估。
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import re
import json
import os
import warnings
import getpass
from typing import Dict, List, Tuple, Optional
import random
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

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

    # def _create_balanced_fallback(self):
    #     """创建平衡的后备数据集"""
    #     fallback_data = [
    #         {'question': "Janet has 3 apples. She eats 1 apple. How many apples does she have left?",
    #          'answer': "Janet starts with 3 apples.\nShe eats 1 apple.\n3 - 1 = 2\n#### 2"},
    #         {'question': "Tom bought 5 books for $2 each. How much did he spend in total?",
    #          'answer': "Tom bought 5 books.\nEach book costs $2.\n5 × 2 = 10\n#### 10"},
    #         {
    #             'question': "A store sells apples for $3 per kg. If John buys 2.5 kg and pays with a $10 bill, how much change does he get?",
    #             'answer': "Cost per kg: $3\nAmount bought: 2.5 kg\nTotal cost: 3 × 2.5 = $7.50\nPaid: $10\nChange: 10 - 7.50 = $2.50\n#### 2.5"},
    #         {
    #             'question': "A company has 120 employees. 25% work in sales, 30% in engineering, and the rest in administration. If the sales team gets a 15% increase and engineering gets a 10% increase, how many total employees will there be after the increases?",
    #             'answer': "Initial employees: 120\nSales: 25% of 120 = 0.25 × 120 = 30 employees\nEngineering: 30% of 120 = 0.30 × 120 = 36 employees\nAdministration: 120 - 30 - 36 = 54 employees\n\nAfter increases:\nSales increase: 30 × 0.15 = 4.5 ≈ 5 new employees\nEngineering increase: 36 × 0.10 = 3.6 ≈ 4 new employees\n\nTotal after increases: 120 + 5 + 4 = 129 employees\n#### 129"}
    #     ]
    #     print(f"✅ Created {len(fallback_data)} fallback samples")
    #     return fallback_data

    def extract_answer(self, answer_text: str) -> str:
        """从GSM8K的答案文本中提取数值"""
        match = re.search(r'####\s*([+-]?\d+(?:\.\d+)?)', answer_text)
        if match: return match.group(1)
        numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', answer_text)
        return numbers[-1] if numbers else "No answer found"

    def count_solution_steps(self, answer: str) -> int:
        """多维度步骤识别综合判断推理复杂度"""
        lines = [line.strip() for line in answer.split('\n') if line.strip()]
        meaningful_lines = [line for line in lines if not line.startswith('####')]
        math_operations = len(re.findall(r'\d+\s*[+\-×÷*/]\s*\d+', answer))
        equals_count = answer.count('=')
        step_words = len(re.findall(r'\b(then|next|so|therefore|thus|after|now|finally)\b', answer.lower()))
        steps = max(len(meaningful_lines) - 1, math_operations, equals_count, step_words, 1)
        return min(steps, 12)

    def classify_difficulty(self, steps: int) -> str:
        """修复的难度分级 - 适应真实GSM8K分布"""
        if steps <= 4:
            return "simple"
        elif steps <= 8:
            return "medium"
        else:
            return "complex"

    def get_balanced_sample(self, n_total: int = 200, simple_ratio: float = 0.5) -> Tuple[List, List]:
        """获取平衡的样本数据，按步骤数分类"""
        print(f"🎯 准备采样 {n_total} 道题目 (简单题比例: {simple_ratio:.1%})")
        simple_problems, complex_problems = [], []
        print("📋 正在分析问题复杂度...")
        for i, item in enumerate(self.samples):
            if i % 100 == 0 and i > 0: print(f"   处理进度: {i}/{len(self.samples)}")
            problem_data = {'question': item['question'], 'answer': self.extract_answer(item['answer']),
                            'original_answer': item['answer'], 'steps': self.count_solution_steps(item['answer']),
                            'difficulty': self.classify_difficulty(self.count_solution_steps(item['answer']))}
            if problem_data['difficulty'] == "simple":
                simple_problems.append(problem_data)
            else:
                complex_problems.append(problem_data)
        print(f"✅ 分类完成: {len(simple_problems)} 简单题, {len(complex_problems)} 复杂题")
        n_simple, n_complex = int(n_total * simple_ratio), n_total - int(n_total * simple_ratio)
        if len(simple_problems) < n_simple: n_simple = len(simple_problems)
        if len(complex_problems) < n_complex: n_complex = len(complex_problems)
        sampled_simple = random.sample(simple_problems, n_simple) if n_simple > 0 else []
        sampled_complex = random.sample(complex_problems, n_complex) if n_complex > 0 else []
        print(f"🎲 最终采样: {len(sampled_simple)} 简单题, {len(sampled_complex)} 复杂题")
        return sampled_simple, sampled_complex


# ========================= 新增: 可训练的复杂度预测网络 =========================
class ComplexityPredictorNet(nn.Module):
    """
    一个简单的神经网络，用于从注意力特征中学习并预测任务复杂度。
    取代了原来硬编码的规则和权重。
    """

    def __init__(self, input_features: int = 8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出一个原始的logit值，用于二分类
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: 一个包含所有注意力特征的向量
        输出: 一个logit值，>0倾向于认为是复杂问题
        """
        return self.network(x)


# ========================= 升级: 可学习的智能路由器 =========================
class LearnedAttentionRouter:
    """
    使用一个预训练好的神经网络来代替固定规则，进行路由决策。
    """

    def __init__(self, model_path: str, device, threshold=0.5):
        self.device = device
        self.threshold = threshold  # 这里的阈值作用于模型的sigmoid输出概率

        # 加载我们训练好的复杂度预测网络
        print(f"🧠 Loading learned complexity predictor from: {model_path}")
        self.predictor_net = ComplexityPredictorNet().to(self.device)
        try:
            self.predictor_net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.predictor_net.eval()  # 切换到评估模式
            print("✅ Learned predictor loaded successfully.")
        except FileNotFoundError:
            print(f"⚠️ 模型文件 {model_path} 未找到! 路由器将使用未训练的网络，结果将是随机的。")
            print("💡 请先运行 'train' 模式来生成模型文件。")
        except Exception as e:
            print(f"❌ 加载预测网络失败: {e}")
            raise

    def route(self, question: str, slm_model, slm_tokenizer) -> Tuple[str, float]:
        """路由决策: 返回 ('SLM'/'LLM', 复杂度分数)"""
        try:
            features = self.extract_core_features(question, slm_model, slm_tokenizer)
            prediction = self.predict_complexity(features)
            route_decision = "LLM" if prediction['is_complex'] else "SLM"
            return route_decision, prediction['complexity_score']
        except Exception as e:
            print(f"⚠️ Route decision failed: {e}, defaulting to LLM")
            return "LLM", 1.0

    def predict_complexity(self, features: dict) -> dict:
        """【核心修改】使用神经网络进行预测"""
        feature_vector = torch.tensor([
            features['avg_entropy'], features['entropy_std'], features['max_entropy'],
            features['avg_variance'], features['variance_std'], features['max_variance'],
            features['avg_max_attention'], features['concentration_std']
        ], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logit = self.predictor_net(feature_vector.unsqueeze(0))
            probability = torch.sigmoid(logit).item()

        return {'complexity_score': probability, 'is_complex': probability > self.threshold}

    def extract_core_features(self, text: str, model, tokenizer) -> dict:
        """稳健的特征提取"""
        model_device = next(model.parameters()).device
        inputs = tokenizer(text, return_tensors="pt", max_length=200, truncation=True, padding=True)
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions[-1][0]
        seq_len = inputs['attention_mask'].sum().item()
        attentions = attentions.cpu()
        entropy_features = self._compute_entropy(attentions, seq_len)
        variance_features = self._compute_variance(attentions, seq_len)
        concentration_features = self._compute_concentration(attentions, seq_len)
        return {**entropy_features, **variance_features, **concentration_features}

    def _compute_entropy(self, attentions, seq_len):
        all_entropies = [-torch.sum(
            (attentions[head, pos, :seq_len] + 1e-9) * torch.log(attentions[head, pos, :seq_len] + 1e-9)).item() for
                         head in range(attentions.shape[0]) for pos in range(seq_len)]
        return {'avg_entropy': np.mean(all_entropies), 'entropy_std': np.std(all_entropies),
                'max_entropy': np.max(all_entropies)}

    def _compute_variance(self, attentions, seq_len):
        variances = [torch.var(attentions[head, pos, :seq_len]).item() for head in range(attentions.shape[0]) for pos in
                     range(seq_len)]
        return {'avg_variance': np.mean(variances), 'variance_std': np.std(variances),
                'max_variance': np.max(variances)}

    def _compute_concentration(self, attentions, seq_len):
        max_attentions = [torch.max(attentions[head, pos, :seq_len]).item() for head in range(attentions.shape[0]) for
                          pos in range(seq_len)]
        return {'avg_max_attention': np.mean(max_attentions), 'concentration_std': np.std(max_attentions)}


# ========================= SLM/LLM接口 和 准确率验证器 (保持不变) =========================
class SLMInterface(ModelInterface):
    """小模型接口"""

    def load_model(self):
        print(f"🔄 Loading SLM: {self.config.name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path, torch_dtype=torch.float16,
                                                              device_map="auto" if torch.cuda.is_available() else None,
                                                              trust_remote_code=True, output_attentions=True)
            if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"✅ SLM loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load SLM: {e}");
            raise

    def predict(self, question: str, num_tokens_to_generate=0) -> tuple:
        """
        【最终升级版 predict 方法】
        - 使用思维链提示，并规定了输出格式。
        - 使用贪婪搜索 (do_sample=False) 保证结果可复现。
        - 在需要时，可以只生成少量token并返回注意力序列。
        """
        if self.model is None: self.load_model()

        # --- 1. 使用新的、更优化的“思维链”提示 ---
        prompt = f"""Solve the following math problem. Think step by step and then write the final answer in the format #### <answer>.

        Question: {question}
        Answer: Let's think step by step.
        """

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        max_len = num_tokens_to_generate if num_tokens_to_generate > 0 else 200
        should_return_attentions = num_tokens_to_generate > 0

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_len,
                do_sample=False,  # <--- 2. 改为贪婪搜索，保证结果确定性
                pad_token_id=self.tokenizer.eos_token_id,
                output_attentions=should_return_attentions,
                return_dict_in_generate=should_return_attentions
            )

        sequence = outputs.sequences[0] if hasattr(outputs, 'sequences') else outputs[0]
        full_response = self.tokenizer.decode(sequence, skip_special_tokens=True)

        # 3. 调整答案切分逻辑以匹配新提示
        assistant_response_start = "Answer: Let's think step by step."
        start_index = full_response.rfind(assistant_response_start)
        answer_text = full_response[
                      start_index + len(assistant_response_start):].strip() if start_index != -1 else full_response

        attentions = outputs.attentions if should_return_attentions and hasattr(outputs, 'attentsions') else None

        return answer_text, attentions


class EnhancedLLMInterface(ModelInterface):
    """增强的大型语言模型接口"""

    def __init__(self, config: ModelConfig, hf_token: str = None):
        super().__init__(config)
        self.hf_token = hf_token
        self.max_memory_per_gpu = "35GB"

    def setup_authentication(self):
        if self.hf_token:
            login(token=self.hf_token)
        elif os.getenv('HUGGINGFACE_TOKEN'):
            login(token=os.getenv('HUGGINGFACE_TOKEN'))
        else:
            print("⚠️ No HuggingFace token provided. Trying without authentication...")

    def load_model(self):
        print(f"🔄 Loading LLM: {self.config.name}");
        self.setup_authentication()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
            model_kwargs = {"torch_dtype": torch.float16, "trust_remote_code": True, "low_cpu_mem_usage": True}
            if "70B" in self.config.model_path:
                model_kwargs.update(
                    {"device_map": "auto", "max_memory": {0: self.max_memory_per_gpu}, "load_in_8bit": True})
            elif "34B" in self.config.model_path or "30B" in self.config.model_path:
                model_kwargs.update({"device_map": "auto", "max_memory": {0: self.max_memory_per_gpu}})
            else:
                model_kwargs.update({"device_map": "auto" if torch.cuda.is_available() else None})
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path, **model_kwargs)
            if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"✅ LLM loaded successfully on {next(self.model.parameters()).device}")
        except Exception as e:
            print(f"❌ Failed to load LLM: {e}"); raise

    def predict(self, question: str) -> str:
        if self.model is None: self.load_model()
        prompt = self._build_math_prompt(question)
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=2048, truncation=True)
        inputs = inputs.to(next(self.model.parameters()).device)
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9,
                                          pad_token_id=self.tokenizer.eos_token_id, repetition_penalty=1.1)
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_response[len(prompt):].strip()

    def _build_math_prompt(self, question: str) -> str:
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful math tutor. Solve the following math problem step by step and provide the final numerical answer at the end.\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI'll solve this step by step.\n\n"""


class AccuracyValidator:
    """准确率验证器"""

    @staticmethod
    def extract_final_answer(response: str) -> str:
        patterns = [r'[Tt]he answer is\s*([+-]?\d+(?:\.\d+)?)', r'[Ff]inal answer:\s*([+-]?\d+(?:\.\d+)?)',
                    r'[Aa]nswer:\s*([+-]?\d+(?:\.\d+)?)', r'=\s*([+-]?\d+(?:\.\d+)?)\s*$',
                    r'####\s*([+-]?\d+(?:\.\d+)?)', r'([+-]?\d+(?:\.\d+)?)\s*$']
        for pattern in patterns:
            match = re.search(pattern, response)
            if match: return match.group(1).strip()
        numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', response)
        return numbers[-1] if numbers else "No answer found"

    @staticmethod
    def is_correct(predicted: str, ground_truth: str, tolerance: float = 0.01) -> bool:
        try:
            pred_num, true_num = float(predicted), float(ground_truth)
            if abs(pred_num - true_num) <= tolerance: return True
            if true_num != 0: return abs(pred_num - true_num) / abs(true_num) <= 0.01
            return False
        except (ValueError, TypeError):
            return str(predicted).strip().lower() == str(ground_truth).strip().lower()


# ========================= 主评估器 (修改后) =========================
class GSM8KAccuracyEvaluator:
    """基于GSM8K的准确率评估器"""

    def __init__(self, hf_token=None, max_samples=1000):
        self.hf_token = hf_token
        self.validator = AccuracyValidator()
        self.data_processor = FixedGSM8KProcessor(max_samples=max_samples)
        self.slm_config = ModelConfig(name="Llama-3.2-3B", model_path="meta-llama/Llama-3.2-3B", cost_per_token=0.001,
                                      avg_latency_ms=100)
        self.llm_config = ModelConfig(name="Llama-3.1-8B-Instruct", model_path="meta-llama/Llama-3.1-8B-Instruct",
                                      cost_per_token=0.003, avg_latency_ms=800)
        self.slm = SLMInterface(self.slm_config)
        self.llm = EnhancedLLMInterface(self.llm_config, hf_token)
        self.router = None  # 等待SLM加载后初始化
        print(
            f"🎯 SLM: {self.slm_config.name}\n🎯 LLM: {self.llm_config.name}\n📊 成本比例: 1:{self.llm_config.cost_per_token / self.slm_config.cost_per_token:.1f}")

    def _ensure_slm_loaded(self):
        """确保SLM已加载（路由器需要用到）"""
        if self.slm.model is None: self.slm.load_model()
        if self.router is None:
            # <--- 修改: 使用新的LearnedAttentionRouter
            self.router = LearnedAttentionRouter(model_path="router_model.pth", device=device, threshold=0.5)
            print("✅ 可学习的注意力路由器已初始化")

    def evaluate_model_on_problems(self, model_interface, problems: List[Dict], model_name: str,
                                   max_problems: Optional[int] = None) -> Dict:
        """在指定问题集合上评估模型"""
        print(f"\n🔍 评估 {model_name}...")
        if max_problems and len(problems) > max_problems:
            problems = random.sample(problems, max_problems)
            print(f"   随机选择 {max_problems} 道题进行测试")
        correct_count, total_count, detailed_results, error_cases = 0, len(problems), [], []
        for i, problem in enumerate(problems):
            question, ground_truth = problem['question'], problem['answer']
            print(f"   处理问题 {i + 1}/{total_count}...", end="\r")
            try:
                response = model_interface.predict(question)
                predicted_answer = self.validator.extract_final_answer(response)
                is_correct = self.validator.is_correct(predicted_answer, ground_truth)
                if is_correct:
                    correct_count += 1
                else:
                    error_cases.append(
                        {'question': question[:100], 'ground_truth': ground_truth, 'predicted': predicted_answer})
                detailed_results.append({'question': question, 'ground_truth': ground_truth, 'model_response': response,
                                         'extracted_answer': predicted_answer, 'is_correct': is_correct})
            except Exception as e:
                print(f"   ⚠️ 处理错误: {e}")
                detailed_results.append(
                    {'question': question, 'ground_truth': ground_truth, 'model_response': f"ERROR: {e}",
                     'extracted_answer': "ERROR", 'is_correct': False})
        print()  # 换行
        accuracy = correct_count / total_count if total_count > 0 else 0
        result = {'model_name': model_name, 'accuracy': accuracy, 'correct_count': correct_count,
                  'total_count': total_count, 'error_rate': 1 - accuracy, 'error_cases': error_cases[:10],
                  'detailed_results': detailed_results}
        print(f"✅ {model_name} 准确率: {accuracy:.2%} ({correct_count}/{total_count})")
        return result

    def evaluate_routing_accuracy(self, simple_problems: List, complex_problems: List) -> Dict:
        """评估路由准确性"""
        print(f"\n🧭 评估路由准确性...")
        self._ensure_slm_loaded()
        correct_routes, total_routes, routing_details = 0, 0, []
        test_simple = simple_problems[:min(50, len(simple_problems))]
        for problem in test_simple:
            predicted_route, _ = self.router.route(problem['question'], self.slm.model, self.slm.tokenizer)
            is_correct = (predicted_route == "SLM")
            if is_correct: correct_routes += 1
            routing_details.append(
                {'question': problem['question'][:100], 'true_complexity': 'simple', 'expected': "SLM",
                 'predicted': predicted_route, 'correct': is_correct})
            total_routes += 1
        test_complex = complex_problems[:min(50, len(complex_problems))]
        for problem in test_complex:
            predicted_route, _ = self.router.route(problem['question'], self.slm.model, self.slm.tokenizer)
            is_correct = (predicted_route == "LLM")
            if is_correct: correct_routes += 1
            routing_details.append(
                {'question': problem['question'][:100], 'true_complexity': 'complex', 'expected': "LLM",
                 'predicted': predicted_route, 'correct': is_correct})
            total_routes += 1
        routing_accuracy = correct_routes / total_routes if total_routes > 0 else 0
        print(f"✅ 路由准确率: {routing_accuracy:.2%} ({correct_routes}/{total_routes})")
        return {'routing_accuracy': routing_accuracy, 'correct_routes': correct_routes, 'total_routes': total_routes,
                'routing_details': routing_details}

    def run_gsm8k_evaluation(self, n_samples: int = 200, simple_ratio: float = 0.5):
        """运行GSM8K评估"""
        print("=" * 60 + "\n🚀 开始基于GSM8K的准确率评估\n" + "=" * 60)
        print(f"📊 准备GSM8K数据 (样本数: {n_samples})...")
        simple_problems, complex_problems = self.data_processor.get_balanced_sample(n_total=n_samples,
                                                                                    simple_ratio=simple_ratio)
        if not simple_problems and not complex_problems: print("❌ 没有可用的数据"); return None

        slm_simple_results = self.evaluate_model_on_problems(self.slm, simple_problems, "SLM on Simple")
        llm_complex_results = self.evaluate_model_on_problems(self.llm, complex_problems, "LLM on Complex")
        slm_complex_results = self.evaluate_model_on_problems(self.slm, complex_problems, "SLM on Complex (X-Eval)")
        llm_simple_results = self.evaluate_model_on_problems(self.llm, simple_problems, "LLM on Simple (X-Eval)")
        routing_results = self.evaluate_routing_accuracy(simple_problems, complex_problems)

        smart_routing_results = self._calculate_smart_routing_performance(slm_simple_results, llm_complex_results,
                                                                          routing_results)
        self._generate_final_report(slm_simple_results, llm_complex_results, slm_complex_results, llm_simple_results,
                                    routing_results, smart_routing_results, n_samples)

    def _calculate_smart_routing_performance(self, slm_res, llm_res, rout_res):
        """计算智能路由系统的整体性能"""
        rout_acc, slm_acc, llm_acc = rout_res['routing_accuracy'], slm_res['accuracy'], llm_res['accuracy']
        simple_ratio = slm_res['total_count'] / max(1, slm_res['total_count'] + llm_res['total_count'])
        # 预估准确率 = (正确路由的部分 * 其对应的模型准确率) + (错误路由的部分 * 其对应的模型准确率)
        # 简单问题被正确路由(SLM)的准确率 + 简单问题被错误路由(LLM)的准确率 + 复杂问题被正确路由(LLM)的准确率 + 复杂问题被错误路由(SLM)的准确率
        # 这里简化为：(理想情况下的准确率 * 路由准确率) + (随机情况下的准确率 * 路由错误率)
        est_acc = ((slm_acc * simple_ratio) + (llm_acc * (1 - simple_ratio))) * rout_acc + 0.5 * (1 - rout_acc)

        smart_cost = simple_ratio * self.slm_config.cost_per_token + (1 - simple_ratio) * self.llm_config.cost_per_token
        llm_cost, slm_cost = self.llm_config.cost_per_token, self.slm_config.cost_per_token
        return {'estimated_accuracy': est_acc, 'cost_per_problem': smart_cost,
                'cost_savings_vs_llm': (llm_cost - smart_cost) / llm_cost,
                'cost_increase_vs_slm': (smart_cost - slm_cost) / slm_cost}

    def _generate_final_report(self, s_s, l_c, s_c, l_s, r, s_r, n):
        print("\n" + "=" * 70 + "\n📋 GSM8K数据集智能路由准确率评估报告\n" + "=" * 70)
        print(f"📊 评估规模: {n} 道GSM8K真题")
        print(f"\n🎯 核心性能指标:")
        print(
            f"├── SLM在简单问题准确率: {s_s['accuracy']:.2%}\n├── LLM在复杂问题准确率: {l_c['accuracy']:.2%}\n├── 路由判断准确率: {r['routing_accuracy']:.2%}\n└── 智能路由预估准确率: {s_r['estimated_accuracy']:.2%}")
        print(
            f"\n📊 交叉验证结果:\n├── SLM在复杂问题准确率: {s_c['accuracy']:.2%} (证明局限性)\n├── LLM在简单问题准确率: {l_s['accuracy']:.2%} (证明过度配置)")
        print(
            f"\n💰 成本效益分析:\n├── 纯SLM成本: ${self.slm_config.cost_per_token:.4f}/题\n├── 纯LLM成本: ${self.llm_config.cost_per_token:.4f}/题\n├── 智能路由成本: ${s_r['cost_per_problem']:.4f}/题\n├── vs LLM节省: {s_r['cost_savings_vs_llm']:.1%}")
        slm_ok, rout_ok, cost_ok = s_s['accuracy'] >= 0.8, r['routing_accuracy'] >= 0.85, s_r[
            'cost_savings_vs_llm'] >= 0.3
        print(
            f"\n💡 关键判断:\n├── SLM可靠性: {'✅' if slm_ok else '❌'} ({s_s['accuracy']:.1%})\n├── 路由可靠性: {'✅' if rout_ok else '❌'} ({r['routing_accuracy']:.1%})\n└── 成本效益: {'✅' if cost_ok else '❌'} ({s_r['cost_savings_vs_llm']:.1%})")
        if slm_ok and rout_ok and cost_ok:
            rec = "✅ 强烈推荐使用智能路由"
        elif not slm_ok:
            rec = "❌ 不推荐 - SLM不可靠"
        elif not rout_ok:
            rec = "❌ 不推荐 - 路由不准确"
        else:
            rec = "⚠️ 谨慎考虑 - 成本效益有限"
        print(f"\n🎯 最终建议: {rec}")


# ========================= 新增: 训练模块 =========================
def generate_router_training_data(evaluator: GSM8KAccuracyEvaluator, output_file="router_training_data.jsonl"):
    """生成用于训练路由器的数据集。"""
    print("\n" + "=" * 50 + "\n🧠 开始生成路由器训练数据...\n" + "=" * 50)
    evaluator._ensure_slm_loaded()
    slm_interface = evaluator.slm
    all_problems = evaluator.data_processor.samples
    print(f"📊 将处理 {len(all_problems)} 个问题来生成特征和标签。")
    training_samples = []
    # 使用临时的、基于规则的分析器来提取特征
    temp_feature_extractor = LearnedAttentionRouter("dummy_path.pth", device)  # 创建实例以使用其特征提取方法

    for i, problem in enumerate(all_problems):
        question = problem['question']
        ground_truth_answer = evaluator.data_processor.extract_answer(problem['answer'])
        if i % 20 == 0: print(f"   进度: {i}/{len(all_problems)}")
        try:
            slm_response = slm_interface.predict(question)
            slm_extracted_answer = evaluator.validator.extract_final_answer(slm_response)
            is_slm_correct = evaluator.validator.is_correct(slm_extracted_answer, ground_truth_answer)
            is_complex_label = 1.0 if not is_slm_correct else 0.0
            attention_features = temp_feature_extractor.extract_core_features(question, slm_interface.model,
                                                                              slm_interface.tokenizer)
            training_samples.append({"features": attention_features, "label": is_complex_label})
        except Exception as e:
            print(f"   ⚠️ 跳过问题 {i}，处理错误: {e}");
            continue
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in training_samples: f.write(json.dumps(sample) + '\n')
    print(f"\n✅ 训练数据生成完毕! 共 {len(training_samples)} 条样本已保存至 {output_file}")


class RouterDataset(Dataset):
    def __init__(self, training_data):
        self.samples = []
        for sample in training_data:
            feature_vector = [sample['features']['avg_entropy'], sample['features']['entropy_std'],
                              sample['features']['max_entropy'], sample['features']['avg_variance'],
                              sample['features']['variance_std'], sample['features']['max_variance'],
                              sample['features']['avg_max_attention'], sample['features']['concentration_std']]
            self.samples.append({"features": torch.tensor(feature_vector, dtype=torch.float32),
                                 "label": torch.tensor([sample['label']], dtype=torch.float32)})

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx): return self.samples[idx]


def train_router(training_data_path="router_training_data.jsonl", epochs=20, lr=1e-4, batch_size=32):
    """训练复杂度预测网络"""
    print("\n" + "=" * 50 + "\n🚀 开始训练智能路由器...\n" + "=" * 50)
    training_data = []
    with open(training_data_path, 'r', encoding='utf-8') as f:
        for line in f: training_data.append(json.loads(line))
    dataset = RouterDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ComplexityPredictorNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss, correct_preds, total_samples = 0, 0, 0
        for batch in dataloader:
            features, labels = batch['features'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct_preds += (preds == labels.bool()).sum().item()
            total_samples += labels.size(0)
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_preds / total_samples
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")
    model_save_path = "router_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ 训练完成! 模型已保存至 {model_save_path}")


# ========================= 主函数和工具函数 =========================
def get_secure_token():
    """安全获取token"""
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if hf_token: print("✅ 从环境变量获取token"); return hf_token
    print("🔑 请输入你的HuggingFace Token:")
    return getpass.getpass("Token: ")


def test_model_access(hf_token):
    """测试token是否可以访问指定的模型"""
    try:
        login(token=hf_token);
        print("✅ HuggingFace token valid")
        print("🔄 Testing Llama-3.2-3B access...");
        AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B");
        print("✅ Llama-3.2-3B access successful")
        print("🔄 Testing Llama-3.1-8B access...");
        AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct");
        print("✅ Llama-3.1-8B access successful")
        return True
    except Exception as e:
        print(f"❌ Model access test failed: {e}"); return False


def main():
    """主函数 - 支持训练和评估模式"""
    print("🔬 GSM8K智能路由准确率评估系统\n" + "=" * 50)
    hf_token = get_secure_token()
    if not hf_token: print("❌ 无法获取token"); return None

    # 模式选择
    mode = input("请选择运行模式 (train/eval): ").strip().lower()

    try:
        if mode == 'train':
            evaluator = GSM8KAccuracyEvaluator(hf_token=hf_token, max_samples=2000)
            generate_router_training_data(evaluator)
            train_router()
        elif mode == 'eval':
            evaluator = GSM8KAccuracyEvaluator(hf_token=hf_token, max_samples=1000)
            evaluator.run_gsm8k_evaluation(n_samples=200, simple_ratio=0.5)
        else:
            print("无效的模式。请输入 'train' 或 'eval'。")
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        hf_token = None


if __name__ == "__main__":
    main()