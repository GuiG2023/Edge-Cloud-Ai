"""
路由前后准确率评估框架
"""
import torch
import json
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM


class RoutingEffectivenessEvaluator:
    def __init__(self, slm_name="microsoft/DialoGPT-small", llm_name="microsoft/DialoGPT-medium"):
        # 本地SLM
        self.slm_tokenizer = AutoTokenizer.from_pretrained(slm_name)
        self.slm_model = AutoModelForCausalLM.from_pretrained(slm_name, output_attentions=True)

        # 云端LLM (模拟)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_name)

        # 设置pad_token
        if self.slm_tokenizer.pad_token is None:
            self.slm_tokenizer.pad_token = self.slm_tokenizer.eos_token
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        # 注意力熵分析器
        from your_previous_code import EnhancedAttentionAnalyzer
        self.attention_analyzer = EnhancedAttentionAnalyzer()

    def load_gsm8k_dataset(self, sample_size=100):
        """加载GSM8K数学推理数据集"""
        # 这里模拟GSM8K数据集，实际使用时需要下载真实数据
        gsm8k_samples = [
            {
                "question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many meters does he run a week?",
                "answer": "540",
                "solution_steps": "3 sprints * 3 times a week = 9 sprints per week. 9 sprints * 60 meters = 540 meters."
            },
            {
                "question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. If Wendi has 20 chickens, how many cups of feed does she need a day?",
                "answer": "60",
                "solution_steps": "20 chickens * 3 cups per chicken = 60 cups"
            },
            {
                "question": "Kylar went to the store to buy glasses for his new apartment. One glass costs $5. If Kylar bought 16 glasses, how much did he spend?",
                "answer": "80",
                "solution_steps": "16 glasses * $5 per glass = $80"
            }
            # 实际应用中需要加载完整的GSM8K数据集
        ]
        return gsm8k_samples[:sample_size]

    def extract_numerical_answer(self, text):
        """从生成的文本中提取数值答案"""
        import re
        # 寻找数字模式
        numbers = re.findall(r'\b\d+\.?\d*\b', text)
        if numbers:
            return numbers[-1]  # 返回最后一个数字作为答案
        return None

    def evaluate_single_answer(self, predicted, ground_truth):
        """评估单个答案的正确性"""
        pred_num = self.extract_numerical_answer(str(predicted))
        truth_num = self.extract_numerical_answer(str(ground_truth))

        if pred_num is None or truth_num is None:
            return False

        try:
            return float(pred_num) == float(truth_num)
        except:
            return pred_num.strip().lower() == truth_num.strip().lower()

    def generate_answer(self, model, tokenizer, question, max_length=100):
        """使用指定模型生成答案"""
        inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer_part = generated_text[len(question):].strip()
        return answer_part

    def predict_routing_decision(self, question):
        """基于注意力熵预测是否需要路由"""
        features = self.attention_analyzer.extract_multi_dimensional_features(
            question, self.slm_model, self.slm_tokenizer
        )
        result = self.attention_analyzer.predict_complexity_enhanced(features)
        return result['is_complex'], result['complexity_score']

    def run_routing_evaluation(self, dataset):
        """运行完整的路由评估实验"""
        results = []

        print(f"🧪 开始路由效果评估，共{len(dataset)}个样本...")

        for i, sample in enumerate(dataset):
            question = sample['question']
            ground_truth = sample['answer']

            print(f"\n📝 样本 {i + 1}: {question[:50]}...")

            # 1. SLM单独处理
            slm_answer = self.generate_answer(self.slm_model, self.slm_tokenizer, question)
            slm_correct = self.evaluate_single_answer(slm_answer, ground_truth)

            # 2. LLM单独处理
            llm_answer = self.generate_answer(self.llm_model, self.llm_tokenizer, question)
            llm_correct = self.evaluate_single_answer(llm_answer, ground_truth)

            # 3. 预测路由决策
            should_route, complexity_score = self.predict_routing_decision(question)

            # 4. 混合策略结果
            if should_route:
                hybrid_answer = llm_answer  # 路由到云端
                hybrid_correct = llm_correct
                used_model = "LLM"
            else:
                hybrid_answer = slm_answer  # 本地处理
                hybrid_correct = slm_correct
                used_model = "SLM"

            # 5. 记录结果
            result = {
                'question': question,
                'ground_truth': ground_truth,
                'slm_answer': slm_answer,
                'llm_answer': llm_answer,
                'hybrid_answer': hybrid_answer,
                'slm_correct': slm_correct,
                'llm_correct': llm_correct,
                'hybrid_correct': hybrid_correct,
                'should_route': should_route,
                'complexity_score': complexity_score,
                'used_model': used_model
            }

            results.append(result)

            print(f"   SLM: {'✅' if slm_correct else '❌'} | LLM: {'✅' if llm_correct else '❌'} | "
                  f"Hybrid: {'✅' if hybrid_correct else '❌'} | Route: {should_route} | Used: {used_model}")

        return self.analyze_routing_results(results)

    def analyze_routing_results(self, results):
        """分析路由结果"""
        total = len(results)

        # 基础准确率
        slm_accuracy = sum(r['slm_correct'] for r in results) / total
        llm_accuracy = sum(r['llm_correct'] for r in results) / total
        hybrid_accuracy = sum(r['hybrid_correct'] for r in results) / total

        # 路由统计
        routed_count = sum(r['should_route'] for r in results)
        routing_rate = routed_count / total

        # 路由决策质量分析
        correct_routing_decisions = 0
        for r in results:
            if r['should_route'] and r['llm_correct'] and not r['slm_correct']:
                correct_routing_decisions += 1  # 正确路由到LLM
            elif not r['should_route'] and (r['slm_correct'] or not r['llm_correct']):
                correct_routing_decisions += 1  # 正确保留在SLM

        routing_decision_accuracy = correct_routing_decisions / total

        # 成本分析（模拟）
        slm_cost = total * 1.0  # SLM单位成本
        llm_cost = total * 10.0  # LLM单位成本
        hybrid_cost = (total - routed_count) * 1.0 + routed_count * 10.0

        cost_savings = (llm_cost - hybrid_cost) / llm_cost

        report = {
            'accuracy': {
                'slm_only': slm_accuracy,
                'llm_only': llm_accuracy,
                'hybrid': hybrid_accuracy
            },
            'routing': {
                'routing_rate': routing_rate,
                'routing_decision_accuracy': routing_decision_accuracy,
                'routed_samples': routed_count,
                'total_samples': total
            },
            'cost': {
                'slm_cost': slm_cost,
                'llm_cost': llm_cost,
                'hybrid_cost': hybrid_cost,
                'cost_savings': cost_savings
            },
            'detailed_results': results
        }

        self.print_evaluation_report(report)
        return report

    def print_evaluation_report(self, report):
        """打印评估报告"""
        print("\n" + "=" * 60)
        print("📊 路由效果评估报告")
        print("=" * 60)

        acc = report['accuracy']
        print(f"\n📈 准确率对比:")
        print(f"  SLM Only:  {acc['slm_only']:.3f} ({acc['slm_only'] * 100:.1f}%)")
        print(f"  LLM Only:  {acc['llm_only']:.3f} ({acc['llm_only'] * 100:.1f}%)")
        print(f"  Hybrid:    {acc['hybrid']:.3f} ({acc['hybrid'] * 100:.1f}%)")

        routing = report['routing']
        print(f"\n🎯 路由决策分析:")
        print(f"  路由率: {routing['routing_rate']:.3f} ({routing['routing_rate'] * 100:.1f}%)")
        print(f"  路由决策准确性: {routing['routing_decision_accuracy']:.3f}")
        print(f"  路由样本数: {routing['routed_samples']}/{routing['total_samples']}")

        cost = report['cost']
        print(f"\n💰 成本效益分析:")
        print(f"  SLM总成本: {cost['slm_cost']:.1f}")
        print(f"  LLM总成本: {cost['llm_cost']:.1f}")
        print(f"  混合成本: {cost['hybrid_cost']:.1f}")
        print(f"  成本节省: {cost['cost_savings']:.3f} ({cost['cost_savings'] * 100:.1f}%)")

        # 综合评价
        if acc['hybrid'] >= acc['slm_only'] and cost['cost_savings'] > 0:
            print(
                f"\n✅ 路由策略有效: 准确率提升 {(acc['hybrid'] - acc['slm_only']) * 100:.1f}%, 成本节省 {cost['cost_savings'] * 100:.1f}%")
        else:
            print(f"\n⚠️ 路由策略需要优化")


# 使用示例
def run_routing_effectiveness_experiment():
    """运行路由效果实验"""
    evaluator = RoutingEffectivenessEvaluator()

    # 加载数据集
    dataset = evaluator.load_gsm8k_dataset(sample_size=50)

    # 运行评估
    results = evaluator.run_routing_evaluation(dataset)

    return results


if __name__ == "__main__":
    results = run_routing_effectiveness_experiment()