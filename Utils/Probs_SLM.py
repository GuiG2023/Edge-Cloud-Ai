
'''
Created on Jun 9
@author: <Guiran>
尝试判断置信度的模块设计 以及相关 function
'''
def estimate_uncertainty(probs):
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    top1_gap = probs[0] - probs[1] if len(probs) > 1 else 1.0
    return entropy, top1_gap


def should_offload(slm_output_probs, attention_predictor_score, token_coverage, task_type):
    entropy, top1_gap = estimate_uncertainty(slm_output_probs)

    # 动态权重调整
    difficulty = 0.0
    if entropy > 2.0: difficulty += 0.4
    if top1_gap < 0.2: difficulty += 0.3
    if attention_predictor_score < 0.5: difficulty += 0.2
    if token_coverage < 0.7: difficulty += 0.3

    # 可根据任务动态调整容忍度
    task_bias = {"qa": 0.2, "summarization": 0.1, "cot": 0.3}
    threshold = 0.6 + task_bias.get(task_type, 0.2)

    return difficulty > threshold

def get_kv_coverage(predicted_attn_block_indices, current_cache_blocks):
    """
    计算 AttentionPredictor 预测的关键 block 中，有多少实际存在于当前 KV cache 中。

    参数:
    ----------
    predicted_attn_block_indices : list/set/array
        AttentionPredictor 输出的 block 索引列表，表示模型预测将来会关注的 token 区域（压缩后的 block 级）
        示例: [2, 5, 7, 10]

    current_cache_blocks : list/set/array
        当前实际保留在 KV cache 中的 block 索引（可能已经剪枝）
        示例: [1, 2, 3, 5, 8]

    返回:
    ----------
    coverage_ratio : float
        命中比例，范围 [0, 1]。值越高，说明保留的 cache 能覆盖更多预测关注区域。
    """

    # 转换为集合，去重并便于判断
    predicted_set = set(predicted_attn_block_indices)
    cache_set = set(current_cache_blocks)

    if not predicted_set:
        # 如果没有预测任何关注区域，返回 1.0 表示“无需关注 → 等于全覆盖”
        return 1.0

    # 命中的 block 数量
    hit_blocks = predicted_set & cache_set
    num_hits = len(hit_blocks)

    # 覆盖率 = 命中数量 / 预测数量
    coverage_ratio = num_hits / len(predicted_set)
    return coverage_ratio
#--------------------------------------------------------
predicted_blocks = [3, 5, 7, 9]
current_cache = [1, 3, 5, 10]

coverage = get_kv_coverage(predicted_blocks, current_cache)
print(f"KV cache 命中率: {coverage:.2f}")  # 输出: 0.50
#--------------------------------------------------------


'''
完整 代码模块：SLM 是否上云判断器 模块with gpt 
'''
import numpy as np
import math


class InferenceDecisionModule:
    def __init__(self, task_type="general"):
        # 不同任务类型的偏置权重
        self.task_bias = {
            "qa": 0.2,  # 问答任务
            "summarization": 0.1,  # 摘要任务
            "cot": 0.3,  # 思维链推理
            "general": 0.2  # 通用任务
        }

        # 各种判断阈值 （暂定，疑似人工定义太多。。办法：蒙特卡洛模拟探索中）

        self.entropy_threshold = 2.0  # 熵阈值：超过此值认为模型不够确定
        self.gap_threshold = 0.2  # 概率差距阈值：小于此值认为模型"摇摆不定"
        self.attn_score_threshold = 0.5  # 注意力分数阈值
        self.coverage_threshold = 0.7  # KV缓存覆盖率阈值

    def estimate_entropy(self, probs):
        """
        计算softmax分布的熵值
        熵越高 = 模型越不确定 = 越需要上云

        Args:
            probs: softmax概率分布列表
        Returns:
            float: 熵值
        """
        return -sum(p * math.log(p + 1e-8) for p in probs if p > 0)

    def top_k_gap(self, probs, k=2):
        """
        计算top-1与top-2概率的差距
        差距越小 = 模型越"摇摆不定" = 越需要上云

        Args:
            probs: 概率分布列表
            k: 比较的top-k个概率
        Returns:
            float: 概率差距
        """
        if len(probs) < k:
            return 1.0
        sorted_probs = sorted(probs, reverse=True)
        return sorted_probs[0] - sorted_probs[1]

    def should_offload(
            self,
            slm_probs,  # 当前token的softmax概率分布
            attn_predictor_score,  # 注意力预测器输出的注意力集中程度(0~1)
            token_coverage,  # 当前KV cache是否包含预测token(0~1)
            task_type="general"  # 当前任务类型
    ):
        """
        综合判断是否需要将推理任务上云

        Args:
            slm_probs: 小模型输出的概率分布
            attn_predictor_score: 注意力预测分数
            token_coverage: token覆盖率
            task_type: 任务类型

        Returns:
            tuple: (是否上云的决策, 详细信息字典)
        """

        # 步骤1：分析softmax输出的不确定性
        entropy = self.estimate_entropy(slm_probs)  # 计算熵值
        gap = self.top_k_gap(slm_probs)  # 计算top-1与top-2的差距

        # 步骤2：综合打分（累积风险分数）
        difficulty_score = 0.0

        # 如果熵值过高（模型不够确定）
        if entropy > self.entropy_threshold:
            difficulty_score += 0.4

        # 如果top-1与top-2差距过小（模型摇摆不定）
        if gap < self.gap_threshold:
            difficulty_score += 0.3

        # 如果注意力预测分数过低（注意力不够集中）
        if attn_predictor_score < self.attn_score_threshold:
            difficulty_score += 0.2

        # 如果token覆盖率过低（KV cache信息不足）
        if token_coverage < self.coverage_threshold:
            difficulty_score += 0.3

        # 步骤3：根据任务类型动态调整阈值
        threshold = 0.6 + self.task_bias.get(task_type, 0.2)

        # 步骤4：最终决策
        decision = difficulty_score > threshold

        # 返回决策结果和详细信息
        return decision, {
            "entropy": entropy,
            "top1_gap": gap,
            "attn_score": attn_predictor_score,
            "coverage": token_coverage,
            "difficulty_score": difficulty_score,
            "threshold": threshold,
            "offload": decision
        }

    #---------------------------------
    # 使用示例
    if __name__ == "__main__":
        # 创建决策模块
        decision_module = InferenceDecisionModule()

        # 模拟场景1：模型很确定的情况
        confident_probs = [0.8, 0.1, 0.05, 0.03, 0.02]  # 第一个token概率很高
        decision1, info1 = decision_module.should_offload(
            slm_probs=confident_probs,
            attn_predictor_score=0.8,  # 注意力集中
            token_coverage=0.9,  # 覆盖率高
            task_type="qa"
        )
        print("场景1 - 模型确定:")
        print(f"是否上云: {decision1}")
        print(f"详细信息: {info1}")
        print()

        # 模拟场景2：模型不确定的情况
        uncertain_probs = [0.3, 0.25, 0.2, 0.15, 0.1]  # 概率分布比较平均
        decision2, info2 = decision_module.should_offload(
            slm_probs=uncertain_probs,
            attn_predictor_score=0.3,  # 注意力不集中
            token_coverage=0.4,  # 覆盖率低
            task_type="cot"  # 复杂推理任务
        )
        print("场景2 - 模型不确定:")
        print(f"是否上云: {decision2}")
        print(f"详细信息: {info2}")
        # ---------------------------------

#一步步完善，简单的
