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
# 在 common_utils.py 文件顶部
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # <--- 增加 BitsAndBytesConfig

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
    # 在 common_utils.py 中

class ComplexityPredictorNet(nn.Module):
    def __init__(self, input_features: int = 4):  # <--- 改为4
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_features, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)

    # class ComplexityPredictorNet(nn.Module):
#     def __init__(self, input_features: int = 18):  # <--- 从8修改为18
#             super().__init__()
#             self.network = nn.Sequential(
#                 nn.Linear(input_features, 128),  # <--- 变宽
#                 nn.ReLU(),
#                 nn.Dropout(0.3),
#                 nn.Linear(128, 64),
#                 nn.ReLU(),
#                 nn.Linear(64, 1)
#             )
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.network(x)

# ========================= 可学习的智能路由器 =========================
class LearnedAttentionRouter:
    def __init__(self, model_path: str, device, threshold: float = 0.5):
        self.device = device
        self.threshold = threshold
        self.model_path = model_path

        # 您需要确保 ComplexityPredictorNet 的定义在此之前
        # 并且 input_features 的数量是正确的
        self.predictor_net = ComplexityPredictorNet(input_features=4).to(self.device)

        if os.path.exists(self.model_path):
            print(f"   Loading learned predictor from: {self.model_path}")
            self.predictor_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.predictor_net.eval()
            print("   ✅ Learned predictor loaded successfully.")
        else:
            print(f"   ⚠️ Model file {self.model_path} not found! Router will be untrained.")
    def extract_core_features(self, text: str, model, tokenizer, slm_interface) -> dict:
        """【【【全新动态特征提取逻辑】】】"""
        # 1. 调用SLM生成前15个token，并获取每一步的注意力序列
        _, attentions_sequence = slm_interface.predict(text, num_tokens_to_generate=15)

        if not attentions_sequence:
            print("   ⚠️ Warning: Attentions not found in generation output.")
            return {}  # 如果出错，返回空字典

        entropies = []
        # 2. 遍历生成过程中的每一步
        for step_attentions_in_tuple in attentions_sequence:
            # 我们只关心最后一层的注意力
            last_layer_attentions = step_attentions_in_tuple[-1]  # 形状: [batch, heads, seq, seq]
            # 只看最新生成的那个token，它对前面所有token的注意力分布
            last_token_attentions_dist = last_layer_attentions[0, :, -1, :].flatten().cpu()

            # 3. 为每一步计算熵
            dist_norm = last_token_attentions_dist / (last_token_attentions_dist.sum() + 1e-9)
            step_entropy = -torch.sum(dist_norm * torch.log(dist_norm + 1e-9)).item()
            entropies.append(step_entropy)

        if not entropies or len(entropies) < 2:
            return {}

        # 4. 从这个熵的时间序列中，提取最终的统计特征
        final_features = {
            'entropy_mean': np.mean(entropies),  # 过程平均熵
            'entropy_std': np.std(entropies),  # 熵波动性 (关键特征!)
            'entropy_max': np.max(entropies),  # 过程中的困惑峰值 (关键特征!)
            'entropy_trend': np.polyfit(range(len(entropies)), entropies, 1)[0]  # 熵的变化趋势
        }
        return final_features

    def route(self, question: str, slm_model, slm_tokenizer, slm_interface) -> Tuple[str, float]:
        try:
            features = self.extract_core_features(question, slm_model, slm_tokenizer, slm_interface)
            if not features: return "LLM", 1.0
            prediction = self.predict_complexity(features)
            return ("LLM" if prediction['is_complex'] else "SLM"), prediction['complexity_score']
        except Exception as e:
            print(f"⚠️ Route decision failed: {e}, defaulting to LLM")
            return "LLM", 1.0

    def predict_complexity(self, features: dict) -> dict:
        feature_keys = ['entropy_mean', 'entropy_std', 'entropy_max', 'entropy_trend']
        feature_vector = torch.tensor([features.get(key, 0.0) for key in feature_keys], dtype=torch.float32).to(
            self.device)
        with torch.no_grad():
            logit = self.predictor_net(feature_vector.unsqueeze(0))
            probability = torch.sigmoid(logit).item()
        return {'complexity_score': probability, 'is_complex': probability > self.threshold}

    # class LearnedAttentionRouter:
#     """使用一个预训练好的神经网络来代替固定规则，进行路由决策。"""
#     def __init__(self, model_path: str, device, threshold=0.5):
#         self.device = device
#         self.threshold = threshold
#         self.model_path = model_path
#         print(f"🧠 Initializing LearnedAttentionRouter...")
#         self.predictor_net = ComplexityPredictorNet().to(self.device)
#         if os.path.exists(self.model_path):
#             print(f"   Loading learned predictor from: {self.model_path}")
#             self.predictor_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
#             self.predictor_net.eval()
#             print("   ✅ Learned predictor loaded successfully.")
#         else:
#             print(f"   ⚠️ Model file {self.model_path} not found! Router will be untrained.")
#             print("   💡 Run 'train_router.py' first to create this file.")
#
#     def route(self, question: str, slm_model, slm_tokenizer) -> Tuple[str, float]:
#         try:
#             features = self.extract_core_features(question, slm_model, slm_tokenizer)
#             prediction = self.predict_complexity(features)
#             route_decision = "LLM" if prediction['is_complex'] else "SLM"
#             return route_decision, prediction['complexity_score']
#         except Exception as e:
#             print(f"⚠️ Route decision failed: {e}, defaulting to LLM")
#             return "LLM", 1.0
#
#     def predict_complexity(self, features: dict) -> dict:
#         feature_vector = torch.tensor([
#             features['avg_entropy'], features['entropy_std'], features['max_entropy'],
#             features['avg_variance'], features['variance_std'], features['max_variance'],
#             features['avg_max_attention'], features['concentration_std']
#         ], dtype=torch.float32).to(self.device)
#         with torch.no_grad():
#             logit = self.predictor_net(feature_vector.unsqueeze(0))
#             probability = torch.sigmoid(logit).item()
#         return {'complexity_score': probability, 'is_complex': probability > self.threshold}

    # 在 LearnedAttentionRouter 类中
    def extract_core_features(self, text: str, model, tokenizer) -> dict:
        """【【【升级版：提取分层特征】】】"""
        model_device = next(model.parameters()).device
        inputs = tokenizer(text, return_tensors="pt", max_length=200, truncation=True, padding=True)
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        all_layer_attentions = outputs.attentions
        seq_len = inputs['attention_mask'].sum().item()
        num_layers = len(all_layer_attentions)
        mid_layer_index, last_layer_index = num_layers // 2, -1
        mid_layer_metrics = self._calculate_metrics_for_layer(all_layer_attentions[mid_layer_index][0], seq_len)
        last_layer_metrics = self._calculate_metrics_for_layer(all_layer_attentions[last_layer_index][0], seq_len)
        final_features = {}
        for key, value in mid_layer_metrics.items(): final_features[f"mid_{key}"] = value
        for key, value in last_layer_metrics.items(): final_features[f"last_{key}"] = value
        final_features['entropy_diff'] = last_layer_metrics['avg_entropy'] - mid_layer_metrics['avg_entropy']
        final_features['variance_diff'] = last_layer_metrics['avg_variance'] - mid_layer_metrics['avg_variance']
        return final_features

    # 在 LearnedAttentionRouter 类中
    def _calculate_metrics_for_layer(self, attentions_for_layer: torch.Tensor, seq_len: int) -> dict:
        """一个辅助函数，为单层的注意力权重计算所有指标"""
        attentions_for_layer = attentions_for_layer.cpu()
        all_entropies, all_variances, all_max_attentions = [], [], []
        for head in range(attentions_for_layer.shape[0]):
            for pos in range(seq_len):
                attn_dist = attentions_for_layer[head, pos, :seq_len]
                attn_dist_normalized = attn_dist / (torch.sum(attn_dist) + 1e-9)
                entropy = -torch.sum(attn_dist_normalized * torch.log(attn_dist_normalized + 1e-9)).item()
                all_entropies.append(entropy)
                all_variances.append(torch.var(attn_dist).item())
                all_max_attentions.append(torch.max(attn_dist).item())
        valid_entropies = [e for e in all_entropies if not np.isnan(e)]
        if not valid_entropies: valid_entropies = [0.0]
        return {
            'avg_entropy': np.mean(valid_entropies), 'entropy_std': np.std(valid_entropies),
            'max_entropy': np.max(valid_entropies), 'avg_variance': np.mean(all_variances),
            'variance_std': np.std(all_variances), 'max_variance': np.max(all_variances),
            'avg_max_attention': np.mean(all_max_attentions), 'concentration_std': np.std(all_max_attentions)
        }

# ========================= SLM/LLM接口 和 准确率验证器 =========================
class SLMInterface(ModelInterface):
    def load_model(self):
        print(f"🔄 Loading SLM: {self.config.name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path, torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True, output_attentions=True, attn_implementation="eager"
            )
            if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✅ SLM loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load SLM: {e}"); raise

    def predict(self, question: str, num_tokens_to_generate=0) -> tuple:
        """
        【升级版】: num_tokens_to_generate > 0 时, 返回(文本, 注意力序列)
        """
        if self.model is None: self.load_model()
        prompt = f"Solve the following math problem step by step...\n\nQuestion: {question}\nAnswer: Let's think step by step."
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        max_len = num_tokens_to_generate if num_tokens_to_generate > 0 else 200
        should_return_attentions = num_tokens_to_generate > 0

        with torch.no_grad():
            outputs = self.model.generate(
                inputs, max_new_tokens=max_len, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                output_attentions=should_return_attentions,
                return_dict_in_generate=should_return_attentions
            )

        sequence = outputs.sequences[0] if hasattr(outputs, 'sequences') else outputs[0]
        answer_text = self.tokenizer.decode(sequence[inputs.shape[-1]:], skip_special_tokens=True)
        attentions = outputs.attentions if should_return_attentions and hasattr(outputs, 'attentions') else None
        return answer_text, attentions


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
        print(f"🔄 Loading LLM: {self.config.name} in bfloat16...")
        self.setup_authentication()
        try:
            # 移除所有和BitsAndBytesConfig相关的代码
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.bfloat16,  # <--- 直接使用bfloat16全精度
                device_map="auto",
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"✅ LLM ({self.config.name}) loaded successfully on {next(self.model.parameters()).device}")
        except Exception as e:
            print(f"❌ Failed to load LLM: {e}")
            raise

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
        # 升级SLM为8B模型
        # --- 【【【修改点 1：将SLM升级为8B模型】】】---
        self.slm_config = ModelConfig(
            name="Llama-3.1-8B-Instruct (New SLM)",
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            cost_per_token=0.003,
            avg_latency_ms=300
        )

        # --- 【【【修改点 2：将LLM更换为Mixtral模型】】】---
        self.llm_config = ModelConfig(
            name="Mixtral-8x7B-Instruct (Sweet Spot LLM)",
            model_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
            cost_per_token=0.008,
            avg_latency_ms=1200
        )
        self.slm = SLMInterface(self.slm_config)
        self.llm = EnhancedLLMInterface(self.llm_config, hf_token)
        self.router = None

    def _ensure_slm_loaded(self):
        if self.slm.model is None: self.slm.load_model()
        if self.router is None:
            router_model_path = os.path.join(self.project_path, "router_model.pth")
            self.router = LearnedAttentionRouter(model_path=router_model_path, device=device, threshold=0.5)

    # ========================================================================
    # ===== 在 common_utils.py 中，使用这个【正确版本】的函数替换旧的 =====
    # ========================================================================

    # 在 GSM8KAccuracyEvaluator 类中
    # 在 common_utils.py 的 GSM8KAccuracyEvaluator 类中

    def evaluate_model_on_problems(self, model_interface, problems: List[Dict],
                                   model_name: str, max_problems: Optional[int] = None) -> Dict:
        import time

        print(f"\n🔍 开始评估 {model_name}... @ {time.ctime()}")

        # 模型加载现在会自动触发
        if model_interface.model is None:
            model_interface.load_model()

        if max_problems and len(problems) > max_problems:
            problems = random.sample(problems, max_problems)

        correct_count, total_count, detailed_results, error_cases = 0, len(problems), [], []

        for i, problem in enumerate(problems):
            print(f"   ➡️  正在处理 {model_name} 的问题 #{i + 1}/{total_count}... @ {time.ctime()}")
            try:
                response = model_interface.predict(problem['question'])
                print(f"   ...问题 #{i + 1} 推理完成，正在验证。 @ {time.ctime()}")

                predicted_answer = self.validator.extract_final_answer(response)
                is_correct = self.validator.is_correct(predicted_answer, problem['answer'])

                if is_correct:
                    correct_count += 1
                else:
                    error_cases.append(
                        {'q': problem['question'][:100], 'gt': problem['answer'], 'pred': predicted_answer})

                detailed_results.append({'is_correct': is_correct})
            except Exception as e:
                print(f"\n   ⚠️ 处理问题 #{i + 1} ('{problem['question'][:30]}...') 时发生错误: {e}")
                detailed_results.append({'is_correct': False})
                continue

        print()  # 换行
        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"✅ {model_name} 准确率: {accuracy:.2%} ({correct_count}/{total_count})")
        return {'accuracy': accuracy, 'correct_count': correct_count, 'total_count': total_count,
                'detailed_results': detailed_results}

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

    # 在 common_utils.py 的 GSM8KAccuracyEvaluator 类中

    def run_gsm8k_evaluation(self, n_samples=200, simple_ratio=0.5):
        import torch
        import time

        print("=" * 60 + "\n🚀 开始优化版评估流程\n" + "=" * 60)
        print(f"📊 准备GSM8K数据 (样本数: {n_samples})...")
        simple_problems, complex_problems = self.data_processor.get_balanced_sample(n_total=n_samples,
                                                                                    simple_ratio=simple_ratio)

        if not simple_problems and not complex_problems:
            print("❌ 错误：没有采样到任何数据。")
            return None

        # --- 阶段一：执行所有SLM相关测试 ---
        print(f"\n--- [阶段一] 开始执行所有SLM相关测试 --- @ {time.ctime()}")
        slm_simple_results = self.evaluate_model_on_problems(self.slm, simple_problems, "SLM on Simple")
        slm_complex_results = self.evaluate_model_on_problems(self.slm, complex_problems, "SLM on Complex (X-Eval)")

        # 评估路由器也需要SLM，所以在这里一并完成
        routing_results = self.evaluate_routing_accuracy(simple_problems, complex_problems)

        # --- 中场：清理SLM，为LLM腾出空间 ---
        print(f"\n--- [中场休息] 清理SLM显存，为LLM做准备 --- @ {time.ctime()}")
        if self.slm.model is not None:
            del self.slm.model
            self.slm.model = None
            torch.cuda.empty_cache()
            print("✅ SLM已从显存移除。")

        # --- 阶段二：执行所有LLM相关测试 ---
        print(f"\n--- [阶段二] 开始执行所有LLM相关测试 --- @ {time.ctime()}")
        llm_complex_results = self.evaluate_model_on_problems(self.llm, complex_problems, "LLM on Complex")
        llm_simple_results = self.evaluate_model_on_problems(self.llm, simple_problems, "LLM on Simple (X-Eval)")

        # --- 生成报告 ---
        print("\n--- [评估完成] 所有测试已结束，正在生成报告 ---")
        smart_routing_results = self._calculate_smart_routing_performance(slm_simple_results, llm_complex_results,
                                                                          routing_results)
        self._generate_final_report(slm_simple_results, llm_complex_results, slm_complex_results, llm_simple_results,
                                    routing_results, smart_routing_results, n_samples)

        return {
            'slm_simple': slm_simple_results,
            'llm_complex': llm_complex_results,
            'routing': routing_results
        }
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