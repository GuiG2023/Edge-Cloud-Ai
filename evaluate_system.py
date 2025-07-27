import os
from common_utils import GSM8KAccuracyEvaluator

if __name__ == "__main__":
    PROJECT_PATH = os.getenv('PROJECT_PATH_GDRIVE', '.')
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    router_model_path = os.path.join(PROJECT_PATH, "router_model.pth")

    if not os.path.exists(router_model_path):
        print(f"❌ Error: Router model not found at '{router_model_path}'!")
        print("   Please run 'train_router.py' first.")
    else:
        evaluator = GSM8KAccuracyEvaluator(hf_token=hf_token, max_samples=1000, project_path=PROJECT_PATH)
        evaluator.run_gsm8k_evaluation(n_samples=200, simple_ratio=0.5)