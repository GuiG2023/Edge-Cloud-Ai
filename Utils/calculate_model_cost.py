
'''
调用成本估算器
后续可可视化调用对比
'''

class CostModel:
    def __init__(self):
        # 基于参数量的相对成本系数
        self.model_costs = {
            "Llama-3.2-3B": 0.167,  # 最小
            "Llama-2-7B": 7.0,  # 7倍

            #备选模型
            # "DialoGPT-small": 1.0,  # 基准
            #"DialoGPT-medium": 3.0,  # 3倍
        }

    def calculate_cost(self, model_name, num_calls):
        """计算总成本"""
        unit_cost = self.model_costs[model_name]
        total_cost = unit_cost * num_calls
        return total_cost

    def calculate_hybrid_cost(self, slm_calls, llm_calls, slm_name, llm_name):
        """计算混合模式成本"""
        slm_cost = self.calculate_cost(slm_name, slm_calls)
        llm_cost = self.calculate_cost(llm_name, llm_calls)
        return slm_cost + llm_cost


# 示例
cost_model = CostModel()

# 场景1: 全部用SLM
slm_total_cost = cost_model.calculate_cost("Llama-3.2-3B", 100)  # 100.0

# 场景2: 全部用LLM
llm_total_cost = cost_model.calculate_cost("Llama-2-7B", 100)  # 700.0

# 场景3: 混合模式（30%路由到LLM）
hybrid_cost = cost_model.calculate_hybrid_cost(
    slm_calls=70, llm_calls=30,
    slm_name="Llama-3.2-3B", llm_name="Llama-2-7B"
)  # 70*1 + 30*7 = 280.0

cost_savings = (llm_total_cost - hybrid_cost) / llm_total_cost  # 60%节省