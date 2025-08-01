# common_utils.py

import torch
import numpy as np
import pandas as pd
from huggingface_hub import login
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
        if self._load_from_datasets():
            print(f"✅ Loaded {len(self.samples)} samples from datasets library")
        elif self._load_from_local():
            print(f"✅ Loaded {len(self.samples)} samples from local file")
        else:
            print("❌ Critical Error: Could not load GSM8K data from any source.")

    def _load_from_datasets(self):
        try:
            from datasets import load_dataset
            print("🔄 Loading from HuggingFace datasets...")
            dataset = load_dataset("gsm8k", "main")
            # 使用训练集生成训练数据，测试集用于最终评估
            data_split = dataset['train']
            for i in range(min(self.max_samples, len(data_split))):
                self.samples.append({'question': data_split[i]['question'], 'answer': data_split[i]['answer']})
            return len(self.samples) > 0
        except Exception as e:
            print(f"⚠️ Failed to load from datasets: {e}"); return False

    def _load_from_local(self):
        try:
            if not os.path.exists(self.data_path): return False
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            self.samples.append(json.loads(line))
                            if len(self.samples) >= self.max_samples: break
                        except: continue
            return len(self.samples) > 0
        except: return False

    def extract_answer(self, answer_text: str) -> str:
        match = re.search(r'####\s*([+-]?[\d,]+(?:\.\d+)?)', answer_text)
        if match: return match.group(1).replace(',', '')
        numbers = re.findall(r'([+-]?[\d,]+(?:\.\d+)?)', answer_text)
        return numbers[-1].replace(',', '') if numbers else "No answer found"

    def count_solution_steps(self, answer: str) -> int:
        lines = [line.strip() for line in answer.split('\n') if line.strip()]
        meaningful_lines = [line for line in lines if not line.startswith('####')]
        math_operations = len(re.findall(r'\d+\s*[+\-×÷*/]\s*\d+', answer))
        equals_count = answer.count('=')
        return max(len(meaningful_lines) - 1, math_operations, equals_count, 1)

    def classify_difficulty(self, steps: int) -> str:
        if steps <= 4: return "simple"
        elif steps <= 8: return "medium"
        else: return "complex"

    def get_balanced_sample(self, n_total: int = 200, simple_ratio: float = 0.5) -> Tuple[List, List]:
        print(f"🎯 准备采样 {n_total} 道题目 (简单题比例: {simple_ratio:.1%})")
        simple_problems, complex_problems = [], []
        print("📋 正在分析问题复杂度...")
        for i, item in enumerate(self.samples):
            steps = self.count_solution_steps(item['answer'])
            problem_data = {'question': item['question'], 'answer': self.extract_answer(item['answer']), 'difficulty': self.classify_difficulty(steps)}
            if problem_data['difficulty'] == "simple": simple_problems.append(problem_data)
            else: complex_problems.append(problem_data)
        print(f"✅ 分类完成: {len(simple_problems)} 简单题, {len(complex_problems)} 复杂题")
        n_simple = int(n_total * simple_ratio)
        n_complex = n_total - n_simple
        sampled_simple = random.sample(simple_problems, min(n_simple, len(simple_problems)))
        sampled_complex = random.sample(complex_problems, min(n_complex, len(complex_problems)))
        print(f"🎲 最终采样: {len(sampled_simple)} 简单题, {len(sampled_complex)} 复杂题")
        return sampled_simple, sampled_complex

# ========================= 可训练的复杂度预测网络 =========================
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

# ========================= 可学习的智能路由器 =========================
class LearnedAttentionRouter:
    """使用一个预训练好的神经网络来代替固定规则，进行路由决策。"""
    def __init__(self, model_path: str, device, threshold=0.5):
        self.device = device
        self.threshold = threshold
        self.model_path = model_path
        print(f"🧠 Initializing LearnedAttentionRouter...")
        self.predictor_net = ComplexityPredictorNet(input_features=8).to(self.device)
        if os.path.exists(self.model_path):
            print(f"   Loading learned predictor from: {self.model_path}")
            self.predictor_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.predictor_net.eval()
            print("   ✅ Learned predictor loaded successfully.")
        else:
            print(f"   ⚠️ Model file {self.model_path} not found! Router will be untrained.")
            print("   💡 Run 'train_router.py' first to create this file.")

    def route(self, question: str, slm_model, slm_tokenizer) -> Tuple[str, float]:
        try:
            features = self.extract_core_features(question, slm_model, slm_tokenizer)
            prediction = self.predict_complexity(features)
            route_decision = "LLM" if prediction['is_complex'] else "SLM"
            return route_decision, prediction['complexity_score']
        except Exception as e:
            print(f"⚠️ Route decision failed: {e}, defaulting to LLM")
            return "LLM", 1.0

    def predict_complexity(self, features: dict) -> dict:
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
        model_device = next(model.parameters()).device
        inputs = tokenizer(text, return_tensors="pt", max_length=200, truncation=True, padding=True)
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions[-1][0].cpu()
        seq_len = inputs['attention_mask'].sum().item()
        entropy_features = self._compute_entropy(attentions, seq_len)
        variance_features = self._compute_variance(attentions, seq_len)
        concentration_features = self._compute_concentration(attentions, seq_len)
        return {**entropy_features, **variance_features, **concentration_features}

    def _compute_entropy(self, attentions, seq_len):
        """
        【【【数值稳定版】】】的熵计算函数
        """
        all_entropies = []
        for head in range(attentions.shape[0]):
            for pos in range(seq_len):
                # 选取当前位置有效的注意力分布
                attn_dist = attentions[head, pos, :seq_len]

                # --- 核心修复开始 ---
                # 1. 为防止除以0，在分母上增加一个极小值epsilon
                #    这一步确保attn_dist是一个合法的概率分布（总和为1）
                attn_dist_normalized = attn_dist / (torch.sum(attn_dist) + 1e-9)

                # 2. 在取对数前，也增加一个极小值epsilon，防止log(0)导致NaN
                entropy = -torch.sum(attn_dist_normalized * torch.log(attn_dist_normalized + 1e-9)).item()
                # --- 核心修复结束 ---

                all_entropies.append(entropy)

        # 作为最后的保险，检查最终列表中是否仍有NaN值
        valid_entropies = [e for e in all_entropies if not np.isnan(e)]
        if not valid_entropies:
            # 如果因未知原因所有计算都失败了，返回默认的0值
            return {'avg_entropy': 0.0, 'entropy_std': 0.0, 'max_entropy': 0.0}

        return {
            'avg_entropy': np.mean(valid_entropies),
            'entropy_std': np.std(valid_entropies),
            'max_entropy': np.max(valid_entropies)
        }
    def _compute_variance(self, attentions, seq_len):
        variances = [torch.var(attentions[h, p, :seq_len]).item() for h in range(attentions.shape[0]) for p in range(seq_len)]
        return {'avg_variance': np.mean(variances), 'variance_std': np.std(variances), 'max_variance': np.max(variances)}
    def _compute_concentration(self, attentions, seq_len):
        max_attentions = [torch.max(attentions[h, p, :seq_len]).item() for h in range(attentions.shape[0]) for p in range(seq_len)]
        return {'avg_max_attention': np.mean(max_attentions), 'concentration_std': np.std(max_attentions)}

# ========================= SLM/LLM接口 和 准确率验证器 =========================
class SLMInterface(ModelInterface):
    def load_model(self):
        print(f"🔄 Loading SLM: {self.config.name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path, torch_dtype=torch.float16, device_map="auto" if torch.cuda.is_available() else None, trust_remote_code=True, output_attentions=True,attn_implementation="eager" )
            if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✅ SLM loaded successfully")
        except Exception as e: print(f"❌ Failed to load SLM: {e}"); raise

    def predict(self, question: str) -> str:
        if self.model is None:
            self.load_model()

        # --- 【【【UPGRADED PROMPT FOR SLM】】】---
        # This prompt guides the model to reason step-by-step and format the final answer.
        prompt = f"""Solve the following math problem. Think step by step and then write the final answer in the format #### <answer>.

    Question: {question}
    Answer: Let's think step by step.
    """
        # --- END NEW PROMPT ---

        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
        if torch.cuda.is_available():
            inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=200,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Adjust the slicing logic to match the new prompt
        assistant_response_start = "Answer: Let's think step by step."
        start_index = full_response.rfind(assistant_response_start)  # Use rfind for robustness
        if start_index != -1:
            answer = full_response[start_index + len(assistant_response_start):].strip()
        else:  # Fallback if the prompt isn't found in the output
            answer = full_response

        return answer


# ==========================================================
# ===== 在 common_utils.py 中，用这个版本替换旧的 =====
# ==========================================================

class EnhancedLLMInterface(ModelInterface):
    def __init__(self, config: ModelConfig, hf_token: str = None):
        super().__init__(config)
        self.hf_token = hf_token

    def setup_authentication(self):
        # ... (此方法保持不变)
        if self.hf_token:
            login(token=self.hf_token)
        elif os.getenv('HUGGINGFACE_TOKEN'):
            print("Found token in environment.")
        else:
            print("⚠️ No HuggingFace token provided.")

    def load_model(self):
        # ... (此方法保持不变)
        print(f"🔄 Loading LLM: {self.config.name}");
        self.setup_authentication()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path, torch_dtype=torch.bfloat16,
                                                              device_map="auto", trust_remote_code=True)
            if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"✅ LLM loaded successfully on {next(self.model.parameters()).device}")
        except Exception as e:
            print(f"❌ Failed to load LLM: {e}"); raise

    def predict(self, question: str) -> str:
        """
        【【【已更新为包含优化版Prompt的版本】】】
        """
        if self.model is None:
            self.load_model()

        # --- 使用新的、更优化的“思维链”提示 ---
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Solve the following math problem step-by-step. At the end, provide the final numerical answer inside <|answer|> and <|end-of-answer|>.

Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Let's think step by step.
"""
        # --- 新提示结束 ---

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(next(self.model.parameters()).device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=300,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 调整答案切分逻辑以匹配新提示
        assistant_response_start = "Let's think step by step."
        start_index = full_response.rfind(assistant_response_start)  # 使用rfind确保从最后的assistant部分开始
        if start_index != -1:
            answer = full_response[start_index + len(assistant_response_start):].strip()
        else:
            # 如果找不到提示，使用正则表达式作为备用方案
            answer = re.sub(r"(?s).*\<\|start_header_id\|\>assistant\<\|end_header_id\|\>\n", "", full_response,
                            1).strip()

        return answer


class AccuracyValidator:
    @staticmethod
    def extract_final_answer(response: str) -> str:
        """
        【【【升级版答案提取器】】】
        1. 优先匹配我们自定义的、最可靠的答案标签。
        2. 其次匹配GSM8K的标准格式。
        3. 增加更多常见句式的匹配。
        4. 将提取最后一个数字作为最终的备用方案。
        5. 自动处理数字中的逗号。
        """
        # 优先匹配我们为LLM设计的<|answer|>标签
        match = re.search(r'<\|answer\|>(.*?)<\|end-of-answer\|>', response, re.DOTALL)
        if match:
            return match.group(1).strip().replace(',', '')

        # 其次匹配GSM8K的标准####格式
        match = re.search(r'####\s*([+-]?[\d,]+(?:\.\d+)?)', response)
        if match:
            return match.group(1).replace(',', '')

        # 匹配 "the answer is X" 等常见句式
        patterns = [
            r'[Tt]he final answer is.*?([+-]?[\d,]+(?:\.\d+)?)',
            r'[Tt]he answer is.*?([+-]?[\d,]+(?:\.\d+)?)',
            r'is therefore.*?([+-]?[\d,]+(?:\.\d+)?)'
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip().replace(',', '')

        # 如果都没有，提取回答中的最后一个数字作为备用
        numbers = re.findall(r'([+-]?[\d,]+(?:\.\d+)?)', response)
        if numbers:
            return numbers[-1].replace(',', '')

        return "No answer found"

# --- 【【【请确保这个方法存在且位于类定义内部】】】 ---
    @staticmethod
    def is_correct(predicted: str, ground_truth: str, tolerance: float = 1e-9) -> bool:
        """判断答案是否正确"""
        try:
            pred_num = float(predicted)
            true_num = float(ground_truth)
            return abs(pred_num - true_num) <= tolerance
        except (ValueError, TypeError):
            # 如果无法转换为数字，则进行字符串比较
            return str(predicted).strip().lower() == str(ground_truth).strip().lower()
# ========================= 主评估器 =========================
class GSM8KAccuracyEvaluator:
    def __init__(self, hf_token=None, max_samples=1000, project_path="."):
        self.project_path = project_path
        self.hf_token = hf_token
        self.validator = AccuracyValidator()
        self.data_processor = FixedGSM8KProcessor(max_samples=max_samples)
        self.slm_config = ModelConfig(name="Llama-3.2-3B", model_path="meta-llama/Llama-3.2-3B", cost_per_token=0.001, avg_latency_ms=100)
        self.llm_config = ModelConfig(name="Llama-3.1-8B-Instruct", model_path="meta-llama/Llama-3.1-8B-Instruct", cost_per_token=0.003, avg_latency_ms=800)
        self.slm = SLMInterface(self.slm_config)
        self.llm = EnhancedLLMInterface(self.llm_config, hf_token)
        self.router = None

    def _ensure_slm_loaded(self):
        if self.slm.model is None: self.slm.load_model()
        if self.router is None:
            router_model_path = os.path.join(self.project_path, "router_model.pth")
            self.router = LearnedAttentionRouter(model_path=router_model_path, device=device, threshold=0.5)

    def evaluate_model_on_problems(self, model_interface, problems, model_name, max_problems=None):
        print(f"\n🔍 评估 {model_name}...")
        if max_problems and len(problems) > max_problems:
            problems = random.sample(problems, max_problems)
        correct_count, total_count, detailed_results, error_cases = 0, len(problems), [], []
        for i, problem in enumerate(problems):
            print(f"   处理问题 {i + 1}/{total_count}...", end="\r")
            try:
                response = model_interface.predict(problem['question'])
                predicted_answer = self.validator.extract_final_answer(response)
                is_correct = self.validator.is_correct(predicted_answer, problem['answer'])
                if is_correct: correct_count += 1
                else: error_cases.append({'q': problem['question'][:100], 'gt': problem['answer'], 'pred': predicted_answer})
                detailed_results.append({'is_correct': is_correct})
            except Exception as e:
                print(f"   ⚠️ 处理错误: {e}"); detailed_results.append({'is_correct': False})
        print()
        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"✅ {model_name} 准确率: {accuracy:.2%} ({correct_count}/{total_count})")
        return {'accuracy': accuracy, 'correct_count': correct_count, 'total_count': total_count, 'detailed_results': detailed_results}

    def evaluate_routing_accuracy(self, simple_problems, complex_problems):
        print("\n🧭 评估路由准确性...")
        self._ensure_slm_loaded()
        correct_routes, total_routes, routing_details = 0, 0, []
        for p_type, problems, expected in [("simple", simple_problems, "SLM"), ("complex", complex_problems, "LLM")]:
            for problem in problems:
                route, _ = self.router.route(problem['question'], self.slm.model, self.slm.tokenizer)
                if route == expected: correct_routes += 1
                total_routes += 1
        accuracy = correct_routes / total_routes if total_routes > 0 else 0
        print(f"✅ 路由准确率: {accuracy:.2%} ({correct_routes}/{total_routes})")
        return {'routing_accuracy': accuracy, 'correct_routes': correct_routes, 'total_routes': total_routes}

    def run_gsm8k_evaluation(self, n_samples=200, simple_ratio=0.5):
        print("="*60 + "\n🚀 开始评估\n" + "="*60)
        simple, complex = self.data_processor.get_balanced_sample(n_total=n_samples, simple_ratio=simple_ratio)
        slm_simple = self.evaluate_model_on_problems(self.slm, simple, "SLM on Simple")
        llm_complex = self.evaluate_model_on_problems(self.llm, complex, "LLM on Complex")
        slm_complex = self.evaluate_model_on_problems(self.slm, complex, "SLM on Complex (X-Eval)", max_problems=50)
        llm_simple = self.evaluate_model_on_problems(self.llm, simple, "LLM on Simple (X-Eval)", max_problems=50)
        routing = self.evaluate_routing_accuracy(simple, complex)
        smart_routing = self._calculate_smart_routing_performance(slm_simple, llm_complex, routing)
        self._generate_final_report(slm_simple, llm_complex, slm_complex, llm_simple, routing, smart_routing, n_samples)

    def _calculate_smart_routing_performance(self, slm_res, llm_res, rout_res):
        rout_acc, slm_acc, llm_acc = rout_res['routing_accuracy'], slm_res['accuracy'], llm_res['accuracy']
        simple_ratio = slm_res['total_count'] / max(1, slm_res['total_count'] + llm_res['total_count'])
        est_acc = ((slm_acc * simple_ratio) + (llm_acc * (1 - simple_ratio))) * rout_acc + ((slm_res.get('accuracy_on_complex', 0) * (1-simple_ratio)) + (llm_res.get('accuracy_on_simple', 1) * simple_ratio)) * (1-rout_acc)
        smart_cost = simple_ratio * self.slm_config.cost_per_token + (1-simple_ratio) * self.llm_config.cost_per_token
        return {'estimated_accuracy': est_acc, 'cost_per_problem': smart_cost, 'cost_savings_vs_llm': (self.llm_config.cost_per_token - smart_cost) / self.llm_config.cost_per_token}

    def _generate_final_report(self, s_s, l_c, s_c, l_s, r, s_r, n):
        print("\n" + "="*70 + "\n📋 评估报告\n" + "="*70)
        print(f"📊 评估规模: {n} 道题")
        print(f"\n🎯 核心性能:\n├── SLM on Simple: {s_s['accuracy']:.2%}\n├── LLM on Complex: {l_c['accuracy']:.2%}\n├── Router Accuracy: {r['routing_accuracy']:.2%}\n└── Smart System Est. Accuracy: {s_r['estimated_accuracy']:.2%}")
        print(f"\n📊 交叉验证:\n├── SLM on Complex: {s_c['accuracy']:.2%}\n├── LLM on Simple: {l_s['accuracy']:.2%}")
        print(f"\n💰 成本效益:\n├── SLM Cost: ${self.slm_config.cost_per_token:.4f}\n├── LLM Cost: ${self.llm_config.cost_per_token:.4f}\n├── Smart Route Cost: ${s_r['cost_per_problem']:.4f}\n├── Savings vs LLM: {s_r['cost_savings_vs_llm']:.1%}")
        slm_ok, rout_ok, cost_ok = s_s['accuracy'] >= 0.8, r['routing_accuracy'] >= 0.85, s_r['cost_savings_vs_llm'] >= 0.3
        print(f"\n💡 关键判断:\n├── SLM Reliability: {'✅' if slm_ok else '❌'}\n├── Router Reliability: {'✅' if rout_ok else '❌'}\n└── Cost-Benefit: {'✅' if cost_ok else '❌'}")
        if slm_ok and rout_ok and cost_ok: rec = "✅ 强烈推荐"
        elif not slm_ok: rec = "❌ 不推荐 - SLM不可靠"
        elif not rout_ok: rec = "❌ 不推荐 - 路由不准确"
        else: rec = "⚠️ 谨慎考虑 - 成本效益有限"
        print(f"\n🎯 最终建议: {rec}")