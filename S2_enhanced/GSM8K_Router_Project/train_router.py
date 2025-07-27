# train_router.py

import os
import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from common_utils import GSM8KAccuracyEvaluator, device  # 从共享文件导入

# 在脚本的开头
import os
PROJECT_PATH = os.getenv('PROJECT_PATH_GDRIVE', '.') # 如果环境变量不存在，则默认为当前目录


# --- 此处粘贴上一回答中的 RouterDataset 和 generate_router_training_data 函数 ---
class RouterDataset(Dataset):
    def __init__(self, training_data):
        self.samples = []
        for sample in training_data:
            feature_vector = [
                sample['features']['avg_entropy'], sample['features']['entropy_std'], sample['features']['max_entropy'],
                sample['features']['avg_variance'], sample['features']['variance_std'],
                sample['features']['max_variance'],
                sample['features']['avg_max_attention'], sample['features']['concentration_std']
            ]
            self.samples.append({
                "features": torch.tensor(feature_vector, dtype=torch.float32),
                "label": torch.tensor([sample['label']], dtype=torch.float32)
            })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx): return self.samples[idx]


def generate_router_training_data(evaluator: GSM8KAccuracyEvaluator, output_file: str):
    print("\n" + "=" * 50 + "\n🧠 开始生成路由器训练数据...\n" + "=" * 50)
    evaluator._ensure_slm_loaded()
    # ... (粘贴上一回答中完整的函数体)
    print(f"\n✅ 训练数据生成完毕! 已保存至 {output_file}")


def train_router(training_data_path="router_training_data.jsonl", epochs=20, lr=1e-4, batch_size=32):
    print("\n" + "=" * 50 + "\n🚀 开始训练智能路由器...\n" + "=" * 50)
    from common_utils import ComplexityPredictorNet  # 在函数内导入避免循环依赖
    training_data = []
    with open(training_data_path, 'r', encoding='utf-8') as f:
        for line in f: training_data.append(json.loads(line))
    dataset = RouterDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ComplexityPredictorNet(input_features=8).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss, correct_preds, total_samples = 0, 0, 0
        for batch in dataloader:
            features, labels = batch['features'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct_preds += (preds == labels.bool()).sum().item()
            total_samples += labels.size(0)
        avg_loss, accuracy = total_loss / len(dataloader), correct_preds / total_samples
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")
    model_save_path = "router_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ 训练完成! 模型已保存至 {model_save_path}")


if __name__ == "__main__":
    print("--- 启动路由器训练流程 ---")
    # 初始化评估器，主要目的是为了使用它的数据处理和模型接口来生成训练数据
    # max_samples可以设得大一些，以生成更丰富的训练数据
    evaluator_for_training = GSM8KAccuracyEvaluator(max_samples=2000)

    # 1. 生成训练数据
    generate_router_training_data(evaluator_for_training, output_file="router_training_data.jsonl")

    # 2. 训练路由器模型
    train_router(training_data_path="router_training_data.jsonl")

    print("--- 训练流程结束 ---")