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
        self.name, self.model_path, self.cost_per_token, self.avg_latency_ms = name, model_path, cost_per_token, avg_latency_ms

# ... (在此处粘贴上一回答中所有的类定义) ...

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