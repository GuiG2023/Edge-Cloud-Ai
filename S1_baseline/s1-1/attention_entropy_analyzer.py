"""
最简单的开始：只看注意力熵这一个维度
目标：验证"高熵=高复杂度"的假设
"""
import torch
import math
from typing import Dict, List


class AttentionEntropyAnalyzer:
    def __init__(self, entropy_threshold=2.0):
        self.entropy_threshold = entropy_threshold
        self.complexity_history = []

    def calculate_attention_entropy(self, attention_weights):
        """
        计算单个token的注意力熵
        输入: attention_weights [num_heads, seq_len]
        输出: 每个head的熵值
        """
        entropies = []
        for head_attn in attention_weights:
            # 避免log(0)
            head_attn = head_attn + 1e-8
            entropy = -torch.sum(head_attn * torch.log(head_attn))
            entropies.append(entropy.item())
        return entropies

    def predict_complexity(self, attention_weights):
        """
        基于注意力熵预测复杂度
        返回: 0-1之间的复杂度分数
        """
        entropies = self.calculate_attention_entropy(attention_weights)
        avg_entropy = sum(entropies) / len(entropies)

        # 简单的阈值判断
        complexity_score = min(avg_entropy / self.entropy_threshold, 1.0)

        return {
            'complexity_score': complexity_score,
            'avg_entropy': avg_entropy,
            'head_entropies': entropies,
            'is_complex': complexity_score > 0.5
        }

    def should_route_to_cloud(self, attention_weights, threshold=0.5):
        """最简单的路由决策"""
        result = self.predict_complexity(attention_weights)
        return result['complexity_score'] > threshold, result