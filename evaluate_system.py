import os
from common_utils import GSM8KAccuracyEvaluator

# 在 evaluate_system.py 文件底部

if __name__ == "__main__":
    import time # 导入时间库

    # 从环境变量中读取 Colab 传递过来的信息
    PROJECT_PATH = os.getenv('PROJECT_PATH_GDRIVE', '.')
    hf_token = os.getenv('HUGGINGFACE_TOKEN')

    router_model_path = os.path.join(PROJECT_PATH, "router_model.pth")

    print(f"--- [诊断模式启动] --- @ {time.ctime()}")

    if not os.path.exists(router_model_path):
        print(f"❌ Error: 路由器模型未找到!")
    else:
        evaluator = GSM8KAccuracyEvaluator(
            hf_token=hf_token,
            max_samples=1000, # 这里max_samples不重要，因为我们下面会用更小的n_samples
            project_path=PROJECT_PATH
        )

        # 【【【核心修改：将样本量缩减到最小】】】
        # 我们只用4个样本（2个简单，2个复杂）来快速跑通全流程
        print(f"\n--- [诊断] 即将以 n_samples=4 的超小规模运行评估... --- @ {time.ctime()}")
        evaluator.run_gsm8k_evaluation(n_samples=4, simple_ratio=0.5)

        print(f"\n--- [诊断模式结束] --- @ {time.ctime()}")