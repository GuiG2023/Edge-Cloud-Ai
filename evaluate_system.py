import os
from common_utils import GSM8KAccuracyEvaluator

if __name__ == "__main__":
    # 从环境变量中读取 Colab 传递过来的信息
    PROJECT_PATH = os.getenv('PROJECT_PATH_GDRIVE', '.')
    hf_token = os.getenv('HUGGINGFACE_TOKEN') # <--- 新增：读取token

    router_model_path = os.path.join(PROJECT_PATH, "router_model.pth")

    if not os.path.exists(router_model_path):
        print(f"❌ Error: Router model not found at '{router_model_path}'!")
        print("   Please run 'train_router.py' first.")
    else:
        # 【核心修改】将读取到的token传递给评估器
        evaluator = GSM8KAccuracyEvaluator(
            hf_token=hf_token,
            max_samples=1000,
            project_path=PROJECT_PATH
        )
        evaluator.run_gsm8k_evaluation(n_samples=50, simple_ratio=0.5)