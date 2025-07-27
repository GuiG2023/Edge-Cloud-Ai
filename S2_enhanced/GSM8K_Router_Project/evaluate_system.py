# evaluate_system.py

import os
from common_utils import GSM8KAccuracyEvaluator # 从共享文件导入

# 在脚本的开头
PROJECT_PATH = os.getenv('PROJECT_PATH_GDRIVE', '.') # 如果环境变量不存在，则默认为当前目录

if __name__ == "__main__":
    print("--- 启动智能路由系统评估流程 ---")

    # 检查路由器模型是否已训练
    router_model_path = "router_model.pth"
    if not os.path.exists(router_model_path):
        print(f"\n❌ 错误: 路由器模型 '{router_model_path}' 未找到！")
        print("   请先运行 'train_router.py' 来训练并生成模型。")
    else:
        # 初始化评估器，它会自动加载训练好的模型
        # 这里的max_samples用于最终评估，可以设小一些以快速看到结果
        evaluator = GSM8KAccuracyEvaluator(max_samples=1000)

        # 运行完整的端到端评估
        results = evaluator.run_gsm8k_evaluation(n_samples=200, simple_ratio=0.5)

        if results:
            print("\n\n--- 评估报告摘要 ---")
            summary = results.get('evaluation_summary', {})
            print(f"建议: {summary.get('recommendation', 'N/A')}")
            print(f"理由: {summary.get('reason', 'N/A')}")

    print("--- 评估流程结束 ---")