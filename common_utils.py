# common_utils.py

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json
import os
import warnings
from typing import Dict, List, Tuple, Optional
import random
from torch import nn

# GPU设置
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("💻 Using CPU")

# ========================= 此处粘贴所有共享的类定义 =========================
# ModelConfig, ModelInterface, FixedGSM8KProcessor, ComplexityPredictorNet,
# LearnedAttentionRouter, SLMInterface, EnhancedLLMInterface,
# AccuracyValidator, GSM8KAccuracyEvaluator
#
# 注意：您需要将我们之前讨论过的所有类的【完整代码】粘贴到这里。
# 为保持本回答的简洁性，这里只展示一个框架。
# ========================================================================

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


class ComplexityPredictorNet(nn.Module):
    """一个简单的神经网络，用于从注意力特征中学习并预测任务复杂度。"""
    def __init__(self, input_features: int = 8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class LearnedAttentionRouter:
    # ... (粘贴完整的类定义)
    pass

class GSM8KAccuracyEvaluator:
    # ... (粘贴完整的类定义)
    pass