import numpy as np
import math
from collections import deque
from typing import Dict, List, Tuple

'''
adaptive_inference_decision_module.py
有待验证可行性
'''
class AdaptiveInferenceDecisionModule:
    def __init__(self, task_type="general"):
        # 基础配置
        self.task_bias = {
            "qa": 0.2,
            "summarization": 0.1,
            "cot": 0.3,
            "general": 0.2
        }

        # 动态阈值系统
        self.adaptive_thresholds = {
            "entropy": {"initial": 2.0, "current": 2.0, "history": deque(maxlen=1000)},
            "gap": {"initial": 0.2, "current": 0.2, "history": deque(maxlen=1000)},
            "attn_score": {"initial": 0.5, "current": 0.5, "history": deque(maxlen=1000)},
            "coverage": {"initial": 0.7, "current": 0.7, "history": deque(maxlen=1000)}
        }

        # 性能跟踪
        self.performance_history = deque(maxlen=1000)
        self.offload_history = deque(maxlen=1000)

        # 统计基础阈值（基于理论分析）
        self.theoretical_baselines = self._calculate_theoretical_baselines()

    def _calculate_theoretical_baselines(self) -> Dict:
        """基于信息论计算理论基线阈值"""
        # 假设词汇表大小为50000（GPT类模型典型值）
        vocab_size = 50000

        # 理论最大熵（均匀分布）
        max_entropy = math.log(vocab_size)

        # 设定阈值为最大熵的某个百分比
        return {
            "entropy_high_uncertainty": max_entropy * 0.15,  # 15%认为高不确定性
            "entropy_medium_uncertainty": max_entropy * 0.08,  # 8%认为中等不确定性
            "gap_confident": 0.5,  # top1-top2差距>0.5认为足够自信
            "gap_uncertain": 0.1,  # 差距<0.1认为很不确定
        }

    def estimate_entropy(self, probs: List[float]) -> float:
        """计算softmax分布的熵值"""
        return -sum(p * math.log(p + 1e-8) for p in probs if p > 0)

    def top_k_gap(self, probs: List[float], k: int = 2) -> float:
        """计算top-k概率差距"""
        if len(probs) < k:
            return 1.0
        sorted_probs = sorted(probs, reverse=True)
        return sorted_probs[0] - sorted_probs[1]

    def _update_adaptive_thresholds(self, metrics: Dict, performance_feedback: float):
        """
        根据历史性能动态调整阈值
        performance_feedback: 1.0=完美, 0.5=一般, 0.0=很差
        """
        for metric_name, value in metrics.items():
            if metric_name in self.adaptive_thresholds:
                threshold_info = self.adaptive_thresholds[metric_name]
                threshold_info["history"].append((value, performance_feedback))

                # 每100次决策后重新校准阈值
                if len(threshold_info["history"]) % 100 == 0:
                    self._recalibrate_threshold(metric_name)

    def _recalibrate_threshold(self, metric_name: str):
        """重新校准特定指标的阈值"""
        threshold_info = self.adaptive_thresholds[metric_name]
        history = list(threshold_info["history"])

        if len(history) < 50:  # 数据不足，保持当前阈值
            return

        # 分析：在什么阈值下效果最好
        good_decisions = [val for val, perf in history if perf > 0.7]
        bad_decisions = [val for val, perf in history if perf < 0.3]

        if good_decisions and bad_decisions:
            # 寻找最优分割点
            good_mean = np.mean(good_decisions)
            bad_mean = np.mean(bad_decisions)
            new_threshold = (good_mean + bad_mean) / 2

            # 平滑更新，避免剧烈波动
            alpha = 0.1  # 学习率
            threshold_info["current"] = (1 - alpha) * threshold_info["current"] + alpha * new_threshold

    def get_dynamic_thresholds(self, task_type: str) -> Dict:
        """获取当前的动态阈值"""
        base_thresholds = {
            "entropy": self.adaptive_thresholds["entropy"]["current"],
            "gap": self.adaptive_thresholds["gap"]["current"],
            "attn_score": self.adaptive_thresholds["attn_score"]["current"],
            "coverage": self.adaptive_thresholds["coverage"]["current"]
        }

        # 根据任务类型调整
        task_multiplier = {
            "cot": 0.9,  # CoT任务降低阈值，更容易上云
            "qa": 1.0,  # QA任务保持标准
            "summarization": 1.1,  # 摘要任务提高阈值，不易上云
            "general": 1.0
        }

        multiplier = task_multiplier.get(task_type, 1.0)
        return {k: v * multiplier for k, v in base_thresholds.items()}

    def should_offload(
            self,
            slm_probs: List[float],
            attn_predictor_score: float,
            token_coverage: float,
            task_type: str = "general",
            performance_feedback: float = None
    ) -> Tuple[bool, Dict]:
        """
        智能决策是否上云
        """
        # 计算基础指标
        entropy = self.estimate_entropy(slm_probs)
        gap = self.top_k_gap(slm_probs)

        # 获取动态阈值
        thresholds = self.get_dynamic_thresholds(task_type)

        # 使用理论基线进行辅助判断
        theoretical = self.theoretical_baselines

        # 多层次评分系统
        difficulty_score = 0.0
        reasons = []

        # 熵值评估（权重0.4）
        if entropy > thresholds["entropy"]:
            difficulty_score += 0.4
            reasons.append(f"高熵值({entropy:.3f}>{thresholds['entropy']:.3f})")
        elif entropy > theoretical["entropy_medium_uncertainty"]:
            difficulty_score += 0.2
            reasons.append(f"中等熵值({entropy:.3f})")

        # 概率差距评估（权重0.3）
        if gap < thresholds["gap"]:
            difficulty_score += 0.3
            reasons.append(f"低概率差距({gap:.3f}<{thresholds['gap']:.3f})")
        elif gap < theoretical["gap_uncertain"]:
            difficulty_score += 0.15
            reasons.append(f"中等概率差距({gap:.3f})")

        # 注意力分数评估（权重0.2）
        if attn_predictor_score < thresholds["attn_score"]:
            difficulty_score += 0.2
            reasons.append(f"低注意力分数({attn_predictor_score:.3f})")

        # 覆盖率评估（权重0.1）
        if token_coverage < thresholds["coverage"]:
            difficulty_score += 0.1
            reasons.append(f"低覆盖率({token_coverage:.3f})")

        # 动态决策阈值
        decision_threshold = 0.6 + self.task_bias.get(task_type, 0.2)
        decision = difficulty_score > decision_threshold

        # 记录决策历史
        metrics = {
            "entropy": entropy,
            "gap": gap,
            "attn_score": attn_predictor_score,
            "coverage": token_coverage
        }

        # 如果有性能反馈，更新阈值
        if performance_feedback is not None:
            self._update_adaptive_thresholds(metrics, performance_feedback)

        return decision, {
            "entropy": entropy,
            "top1_gap": gap,
            "attn_score": attn_predictor_score,
            "coverage": token_coverage,
            "difficulty_score": difficulty_score,
            "threshold": decision_threshold,
            "dynamic_thresholds": thresholds,
            "theoretical_baselines": theoretical,
            "reasons": reasons,
            "offload": decision
        }

    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        if not self.performance_history:
            return {"msg": "暂无性能数据"}

        recent_performance = list(self.performance_history)[-100:]  # 最近100次
        return {
            "avg_performance": np.mean(recent_performance),
            "offload_rate": np.mean(list(self.offload_history)[-100:]),
            "total_decisions": len(self.performance_history),
            "current_thresholds": {k: v["current"] for k, v in self.adaptive_thresholds.items()}
        }


# 使用示例和对比
if __name__ == "__main__":
    # 创建自适应决策模块
    adaptive_module = AdaptiveInferenceDecisionModule()

    print("=== 理论基线阈值 ===")
    print(f"理论基线: {adaptive_module.theoretical_baselines}")
    print()

    # 场景测试
    test_cases = [
        {
            "name": "高确定性场景",
            "probs": [0.8, 0.1, 0.05, 0.03, 0.02],
            "attn_score": 0.8,
            "coverage": 0.9,
            "task": "qa"
        },
        {
            "name": "中等不确定性",
            "probs": [0.4, 0.3, 0.15, 0.1, 0.05],
            "attn_score": 0.6,
            "coverage": 0.7,
            "task": "general"
        },
        {
            "name": "高不确定性",
            "probs": [0.25, 0.24, 0.20, 0.16, 0.15],
            "attn_score": 0.3,
            "coverage": 0.4,
            "task": "cot"
        }
    ]

    for case in test_cases:
        print(f"=== {case['name']} ===")
        decision, info = adaptive_module.should_offload(
            slm_probs=case["probs"],
            attn_predictor_score=case["attn_score"],
            token_coverage=case["coverage"],
            task_type=case["task"]
        )

        print(f"决策: {'上云' if decision else '本地'}")
        print(f"难度分数: {info['difficulty_score']:.3f}")
        print(f"决策原因: {', '.join(info['reasons'])}")
        print(f"熵值: {info['entropy']:.3f}")
        print(f"概率差距: {info['top1_gap']:.3f}")
        print()