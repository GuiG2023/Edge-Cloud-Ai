"""
预测式 vs 验证式路由对比实验
"""
import torch
import time
from typing import Dict, List, Tuple


class PredictiveRouter:
    """预测式路由器 - 你的方法"""

    def __init__(self, slm_model, slm_tokenizer, attention_analyzer):
        self.slm = slm_model
        self.tokenizer = slm_tokenizer
        self.attention_analyzer = attention_analyzer

    def route_decision(self, task_text):
        """在生成前预测是否需要路由"""
        start_time = time.time()

        # 提取注意力特征
        features = self.attention_analyzer.extract_multi_dimensional_features(
            task_text, self.slm, self.tokenizer
        )

        # 预测复杂度
        result = self.attention_analyzer.predict_complexity_enhanced(features)

        decision_time = time.time() - start_time

        return {
            'should_route': result['is_complex'],
            'complexity_score': result['complexity_score'],
            'decision_time': decision_time,
            'method': 'predictive'
        }


class VerificationRouter:
    """验证式路由器 - 论文方法"""

    def __init__(self, slm_model, slm_tokenizer, llm_model, llm_tokenizer, prob_threshold=0.1):
        self.slm = slm_model
        self.slm_tokenizer = slm_tokenizer
        self.llm = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.prob_threshold = prob_threshold

    def route_decision(self, task_text, num_preview_tokens=10):
        """生成部分token后验证是否需要路由"""
        start_time = time.time()

        # 1. SLM先生成几个token
        inputs = self.slm_tokenizer(task_text, return_tensors="pt")

        with torch.no_grad():
            # 生成预览token
            preview_outputs = self.slm.generate(
                **inputs,
                max_new_tokens=num_preview_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=True,
                temperature=0.7
            )

        preview_tokens = preview_outputs.sequences[0][inputs['input_ids'].shape[1]:]

        # 2. LLM评估这些token的概率
        full_text = self.slm_tokenizer.decode(preview_outputs.sequences[0])
        llm_inputs = self.llm_tokenizer(full_text, return_tensors="pt")

        with torch.no_grad():
            llm_outputs = self.llm(**llm_inputs)
            llm_probs = torch.softmax(llm_outputs.logits[0], dim=-1)

        # 3. 检查是否有低概率token
        should_route = False
        min_prob = float('inf')

        for i, token_id in enumerate(preview_tokens):
            if i < llm_probs.shape[0] - 1:  # 确保索引有效
                token_prob = llm_probs[-(len(preview_tokens) - i), token_id].item()
                min_prob = min(min_prob, token_prob)

                if token_prob < self.prob_threshold:
                    should_route = True
                    break

        decision_time = time.time() - start_time

        return {
            'should_route': should_route,
            'min_token_prob': min_prob,
            'decision_time': decision_time,
            'method': 'verification',
            'preview_tokens': len(preview_tokens)
        }


class RouterComparison:
    """路由方法对比实验"""

    def __init__(self):
        # 加载模型
        self.slm_name = "microsoft/DialoGPT-small"
        self.llm_name = "microsoft/DialoGPT-medium"

        # SLM
        self.slm_tokenizer = AutoTokenizer.from_pretrained(self.slm_name)
        self.slm_model = AutoModelForCausalLM.from_pretrained(self.slm_name, output_attentions=True)

        # LLM
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.llm_name)

        # 设置pad_token
        if self.slm_tokenizer.pad_token is None:
            self.slm_tokenizer.pad_token = self.slm_tokenizer.eos_token
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        # 注意力分析器
        from your_previous_code import EnhancedAttentionAnalyzer  # 需要导入你之前的代码
        attention_analyzer = EnhancedAttentionAnalyzer()

        # 初始化路由器
        self.predictive_router = PredictiveRouter(
            self.slm_model, self.slm_tokenizer, attention_analyzer
        )

        self.verification_router = VerificationRouter(
            self.slm_model, self.slm_tokenizer,
            self.llm_model, self.llm_tokenizer,
            prob_threshold=0.1
        )

    def generate_full_answer(self, model, tokenizer, question):
        """生成完整答案"""
        inputs = tokenizer(question, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text[len(question):].strip()
        return answer

    def extract_answer(self, text):
        """从文本中提取数值答案"""
        import re
        numbers = re.findall(r'\b\d+\.?\d*\b', text)
        return numbers[-1] if numbers else None

    def evaluate_answer(self, predicted, ground_truth):
        """评估答案正确性"""
        pred_num = self.extract_answer(str(predicted))
        truth_num = self.extract_answer(str(ground_truth))

        if pred_num is None or truth_num is None:
            return False

        try:
            return float(pred_num) == float(truth_num)
        except:
            return pred_num.strip().lower() == truth_num.strip().lower()

    def run_comparison_experiment(self, dataset):
        """运行对比实验"""
        results = []

        print("🔬 开始预测式 vs 验证式路由对比实验")
        print("=" * 60)

        for i, sample in enumerate(dataset):
            question = sample['question']
            ground_truth = sample['answer']

            print(f"\n📝 样本 {i + 1}: {question[:50]}...")

            # 1. 预测式路由决策
            pred_decision = self.predictive_router.route_decision(question)

            # 2. 验证式路由决策
            verif_decision = self.verification_router.route_decision(question)

            # 3. 生成答案
            slm_answer = self.generate_full_answer(self.slm_model, self.slm_tokenizer, question)
            llm_answer = self.generate_full_answer(self.llm_model, self.llm_tokenizer, question)

            # 4. 根据路由决策选择最终答案
            if pred_decision['should_route']:
                pred_final_answer = llm_answer
                pred_used_model = "LLM"
            else:
                pred_final_answer = slm_answer
                pred_used_model = "SLM"

            if verif_decision['should_route']:
                verif_final_answer = llm_answer
                verif_used_model = "LLM"
            else:
                verif_final_answer = slm_answer
                verif_used_model = "SLM"

            # 5. 评估准确性
            slm_correct = self.evaluate_answer(slm_answer, ground_truth)
            llm_correct = self.evaluate_answer(llm_answer, ground_truth)
            pred_correct = self.evaluate_answer(pred_final_answer, ground_truth)
            verif_correct = self.evaluate_answer(verif_final_answer, ground_truth)

            # 6. 记录结果
            result = {
                'question': question,
                'ground_truth': ground_truth,
                'slm_answer': slm_answer,
                'llm_answer': llm_answer,
                'slm_correct': slm_correct,
                'llm_correct': llm_correct,

                # 预测式结果
                'pred_should_route': pred_decision['should_route'],
                'pred_complexity_score': pred_decision['complexity_score'],
                'pred_decision_time': pred_decision['decision_time'],
                'pred_final_answer': pred_final_answer,
                'pred_used_model': pred_used_model,
                'pred_correct': pred_correct,

                # 验证式结果
                'verif_should_route': verif_decision['should_route'],
                'verif_min_prob': verif_decision['min_token_prob'],
                'verif_decision_time': verif_decision['decision_time'],
                'verif_final_answer': verif_final_answer,
                'verif_used_model': verif_used_model,
                'verif_correct': verif_correct,
            }

            results.append(result)

            # 实时显示结果
            print(f"   预测式: {'✅' if pred_correct else '❌'} | 路由: {pred_decision['should_route']} | "
                  f"用时: {pred_decision['decision_time']:.3f}s | 使用: {pred_used_model}")
            print(f"   验证式: {'✅' if verif_correct else '❌'} | 路由: {verif_decision['should_route']} | "
                  f"用时: {verif_decision['decision_time']:.3f}s | 使用: {verif_used_model}")

        return self.analyze_comparison_results(results)

    def analyze_comparison_results(self, results):
        """分析对比结果"""
        total = len(results)

        # 基础准确率
        slm_accuracy = sum(r['slm_correct'] for r in results) / total
        llm_accuracy = sum(r['llm_correct'] for r in results) / total
        pred_accuracy = sum(r['pred_correct'] for r in results) / total
        verif_accuracy = sum(r['verif_correct'] for r in results) / total

        # 路由统计
        pred_routing_rate = sum(r['pred_should_route'] for r in results) / total
        verif_routing_rate = sum(r['verif_should_route'] for r in results) / total

        # 决策时间
        pred_avg_time = sum(r['pred_decision_time'] for r in results) / total
        verif_avg_time = sum(r['verif_decision_time'] for r in results) / total

        # 路由一致性
        routing_agreement = sum(
            r['pred_should_route'] == r['verif_should_route'] for r in results
        ) / total

        # 成本计算
        pred_cost = self.calculate_routing_cost(results, 'pred')
        verif_cost = self.calculate_routing_cost(results, 'verif')
        slm_cost = total * 1.0
        llm_cost = total * 10.0

        # 效率分析
        pred_efficiency = pred_accuracy / (pred_cost / slm_cost)
        verif_efficiency = verif_accuracy / (verif_cost / slm_cost)

        report = {
            'accuracy': {
                'slm_only': slm_accuracy,
                'llm_only': llm_accuracy,
                'predictive': pred_accuracy,
                'verification': verif_accuracy
            },
            'routing': {
                'predictive_rate': pred_routing_rate,
                'verification_rate': verif_routing_rate,
                'agreement': routing_agreement
            },
            'timing': {
                'predictive_avg': pred_avg_time,
                'verification_avg': verif_avg_time,
                'speedup_ratio': verif_avg_time / pred_avg_time
            },
            'cost': {
                'slm_cost': slm_cost,
                'llm_cost': llm_cost,
                'predictive_cost': pred_cost,
                'verification_cost': verif_cost
            },
            'efficiency': {
                'predictive': pred_efficiency,
                'verification': verif_efficiency
            },
            'detailed_results': results
        }

        self.print_comparison_report(report)
        return report

    def calculate_routing_cost(self, results, method_prefix):
        """计算路由成本"""
        slm_calls = 0
        llm_calls = 0

        for r in results:
            if method_prefix == 'pred':
                if r['pred_should_route']:
                    llm_calls += 1
                else:
                    slm_calls += 1
                # 预测式还需要计算预测成本
                slm_calls += 0.1  # 预测的额外成本很小
            else:  # verification
                slm_calls += 1  # 总是需要SLM先生成
                if r['verif_should_route']:
                    llm_calls += 1
                llm_calls += 0.1  # LLM验证的额外成本

        return slm_calls * 1.0 + llm_calls * 10.0

    def print_comparison_report(self, report):
        """打印对比报告"""
        print("\n" + "=" * 70)
        print("📊 预测式 vs 验证式路由对比报告")
        print("=" * 70)

        acc = report['accuracy']
        print(f"\n📈 准确率对比:")
        print(f"  SLM Only:      {acc['slm_only']:.3f} ({acc['slm_only'] * 100:.1f}%)")
        print(f"  LLM Only:      {acc['llm_only']:.3f} ({acc['llm_only'] * 100:.1f}%)")
        print(f"  预测式路由:     {acc['predictive']:.3f} ({acc['predictive'] * 100:.1f}%)")
        print(f"  验证式路由:     {acc['verification']:.3f} ({acc['verification'] * 100:.1f}%)")

        routing = report['routing']
        print(f"\n🎯 路由行为对比:")
        print(f"  预测式路由率:   {routing['predictive_rate']:.3f} ({routing['predictive_rate'] * 100:.1f}%)")
        print(f"  验证式路由率:   {routing['verification_rate']:.3f} ({routing['verification_rate'] * 100:.1f}%)")
        print(f"  路由决策一致性: {routing['agreement']:.3f} ({routing['agreement'] * 100:.1f}%)")

        timing = report['timing']
        print(f"\n⏱️ 效率对比:")
        print(f"  预测式平均决策时间: {timing['predictive_avg']:.3f}s")
        print(f"  验证式平均决策时间: {timing['verification_avg']:.3f}s")
        print(f"  预测式速度优势:     {timing['speedup_ratio']:.1f}x")

        cost = report['cost']
        print(f"\n💰 成本对比:")
        print(f"  SLM成本:      {cost['slm_cost']:.1f}")
        print(f"  LLM成本:      {cost['llm_cost']:.1f}")
        print(f"  预测式成本:    {cost['predictive_cost']:.1f}")
        print(f"  验证式成本:    {cost['verification_cost']:.1f}")

        eff = report['efficiency']
        print(f"\n⚡ 效率指标 (准确率/相对成本):")
        print(f"  预测式效率:    {eff['predictive']:.3f}")
        print(f"  验证式效率:    {eff['verification']:.3f}")

        # 综合评价
        print(f"\n🏆 综合评价:")
        if acc['predictive'] >= acc['verification'] and timing['predictive_avg'] < timing['verification_avg']:
            print("✅ 预测式方法在准确率和效率上都优于验证式方法")
        elif acc['predictive'] > acc['verification']:
            print("✅ 预测式方法准确率更高")
        elif timing['predictive_avg'] < timing['verification_avg']:
            print("✅ 预测式方法效率更高")
        else:
            print("⚠️ 两种方法各有优劣，需要根据具体需求选择")


def run_router_comparison():
    """运行路由器对比实验"""
    comparison = RouterComparison()

    # 加载测试数据集
    evaluator = RoutingEffectivenessEvaluator()
    dataset = evaluator.load_gsm8k_dataset(sample_size=20)  # 小样本测试

    # 运行对比
    results = comparison.run_comparison_experiment(dataset)

    return results


# 你的实施步骤
def your_implementation_roadmap():
    """你的逐步实施路线图"""
    print("🗺️ 基于当前实验的实施路线图")
    print("=" * 50)

    steps = [
        {
            "step": 1,
            "title": "完善当前Baseline",
            "tasks": [
                "使用GSM8K数据集替换当前的认知复杂度任务",
                "实现准确率评估机制",
                "建立成本计算模型"
            ],
            "time": "3-5天"
        },
        {
            "step": 2,
            "title": "实现路由效果评估",
            "tasks": [
                "构建RoutingEffectivenessEvaluator",
                "测试不同阈值下的性能",
                "生成权衡分析图表"
            ],
            "time": "2-3天"
        },
        {
            "step": 3,
            "title": "实现验证式路由对比",
            "tasks": [
                "实现VerificationRouter",
                "进行预测式vs验证式对比",
                "分析两种方法的优劣"
            ],
            "time": "3-4天"
        },
        {
            "step": 4,
            "title": "构建Attention Predictor",
            "tasks": [
                "基于Baseline训练专门的预测器",
                "实现在线学习机制",
                "优化预测精度"
            ],
            "time": "1-2周"
        },
        {
            "step": 5,
            "title": "综合评估和论文撰写",
            "tasks": [
                "多数据集验证",
                "与其他方法对比",
                "撰写实验报告"
            ],
            "time": "1-2周"
        }
    ]

    for step in steps:
        print(f"\n🎯 步骤{step['step']}: {step['title']} ({step['time']})")
        for task in step['tasks']:
            print(f"   • {task}")

    print(f"\n⏰ 总预计时间: 4-6周")
    print(f"🎓 完成后的成果: 完整的SLM-LLM路由系统 + 学术论文")


if __name__ == "__main__":
    # 显示实施路线图
    your_implementation_roadmap()

    print("\n" + "=" * 50)
    print("🚀 开始运行对比实验...")

    # 运行实际对比实验
    results = run_router_comparison()