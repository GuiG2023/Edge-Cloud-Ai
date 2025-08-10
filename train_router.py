import os
import json
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from common_utils import GSM8KAccuracyEvaluator, device, LearnedAttentionRouter, AccuracyValidator, \
    ComplexityPredictorNet


# ========================================================================
# ===== 在 train_router.py 文件中，使用这个新版本的函数来替换旧的 =====
# ========================================================================

# ========================================================================
# ===== 在 train_router.py 文件中，使用这个【带实时计数】的版本 =====
# ========================================================================

def generate_router_training_data(evaluator, output_file):
    """
    生成用于训练路由器的数据集。
    【【【健壮版：支持实时保存、断点续传和实时计数】】】
    """
    import time

    print("\n" + "=" * 50 + "\n🧠 Generating router training data (Resumable Mode)...\n" + "=" * 50)

    processed_samples = set()
    processed_count = 0
    # --- 【【【新增】】】初始化简单/复杂计数器 ---
    simple_label_count = 0
    complex_label_count = 0
    # ------------------------------------

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'question' in data:
                        processed_samples.add(data['question'])
                        # --- 【【【新增】】】从已有文件中恢复计数 ---
                        if data.get('label') == 0.0:
                            simple_label_count += 1
                        else:
                            complex_label_count += 1
                        # ------------------------------------
                except:
                    continue
        processed_count = len(processed_samples)
        print(f"🔄 Found existing data file with {processed_count} samples. Resuming...")
        print(f"   Initial counts: Simple Labels = {simple_label_count}, Complex Labels = {complex_label_count}")

    evaluator._ensure_slm_loaded()
    slm_interface = evaluator.slm
    all_problems = evaluator.data_processor.samples
    total_to_process = len(all_problems)
    print(f"📊 Total problems to process: {total_to_process}. Already processed: {processed_count}.")

    start_time = time.time()  # 记录开始时间

    with open(output_file, 'a', encoding='utf-8') as f:
        temp_feature_extractor = LearnedAttentionRouter("dummy_path.pth", device)
        validator = AccuracyValidator()

        for i, problem in enumerate(all_problems):
            if problem['question'] in processed_samples:
                continue

            current_progress = processed_count + 1

            try:
                slm_response = slm_interface.predict(problem['question'])
                slm_answer = validator.extract_final_answer(slm_response)
                gt_answer = evaluator.data_processor.extract_answer(problem['answer'])
                is_slm_correct = validator.is_correct(slm_answer, gt_answer)

                label = 1.0 if not is_slm_correct else 0.0

                # --- 【【【新增】】】更新计数器 ---
                if label == 0.0:
                    simple_label_count += 1
                else:
                    complex_label_count += 1
                # --------------------------------

                features = temp_feature_extractor.extract_core_features(
                    problem['question'], slm_interface.model, slm_interface.tokenizer
                )

                sample_to_save = {"question": problem['question'], "features": features, "label": label}

                f.write(json.dumps(sample_to_save) + '\n')
                f.flush()
                processed_count += 1
                processed_samples.add(problem['question'])

                # --- 【【【核心修改：增加详细进度报告】】】---
                # 每处理20个样本，或者在第一个和最后一个时，打印一次清晰的进度
                if current_progress % 20 == 0 or current_progress == 1 or current_progress == total_to_process:
                    elapsed_time = time.time() - start_time
                    samples_per_second = (
                                                     current_progress - processed_count + simple_label_count + complex_label_count) / elapsed_time if elapsed_time > 0 else 0
                    print(f"\n--- Progress Update ---")
                    print(f"   Processed: {current_progress}/{total_to_process}")
                    print(
                        f"   Label Counts: Simple (Correct) = {simple_label_count}, Complex (Incorrect) = {complex_label_count}")
                    print(f"   Speed: {samples_per_second:.2f} samples/sec")
                    print(f"-----------------------")
                # --- 进度报告结束 ---


            except Exception as e:
                print(f"\n   ⚠️ Skipped problem #{i} ('{problem['question'][:30]}...') due to error: {e}")
                continue

    print(f"\n✅ Training data generation complete! Total {processed_count} samples saved to {output_file}")
    print(f"   Final Label Distribution: Simple = {simple_label_count}, Complex = {complex_label_count}")
# 在 train_router.py 中
class RouterDataset(Dataset):
    def __init__(self, data_path):
        self.samples = []
        self.feature_keys = [
            'mid_avg_entropy', 'mid_entropy_std', 'mid_max_entropy', 'mid_avg_variance',
            'mid_variance_std', 'mid_max_variance', 'mid_avg_max_attention', 'mid_concentration_std',
            'last_avg_entropy', 'last_entropy_std', 'last_max_entropy', 'last_avg_variance',
            'last_variance_std', 'last_max_variance', 'last_avg_max_attention', 'last_concentration_std',
            'entropy_diff', 'variance_diff'
        ]
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                feature_vector = [sample['features'].get(key, 0.0) for key in self.feature_keys]
                self.samples.append({
                    "features": torch.tensor(feature_vector, dtype=torch.float32),
                    "label": torch.tensor([sample['label']], dtype=torch.float32)
                })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx): return self.samples[idx]


def train_router(training_data_path, model_save_path, epochs=20, lr=1e-4, batch_size=32):
    from common_utils import ComplexityPredictorNet  # Local import
    print("\n" + "=" * 50 + "\n🚀 Training the smart router...\n" + "=" * 50)
    dataset = RouterDataset(training_data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ComplexityPredictorNet().to(device)
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
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_preds / total_samples
        print(f"Epoch {epoch + 1:02d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")

    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ Training complete! Model saved to {model_save_path}")


# train_router.py (最终修正版)

# ==========================================================
# ===== 修改点 1: RouterDataset 现在接收 feature_subset =====
# ==========================================================
class RouterDataset(Dataset):
    def __init__(self, data_path, feature_subset: list):
        self.samples = []
        self.feature_keys = feature_subset  # 直接使用传入的特征列表

        print(f"--- Dataset Initialized using {len(self.feature_keys)} features: {self.feature_keys} ---")

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    # 严格按照传入的feature_keys顺序和数量构建特征向量
                    feature_vector = [sample['features'].get(key, 0.0) for key in self.feature_keys]
                    self.samples.append({
                        "features": torch.tensor(feature_vector, dtype=torch.float32),
                        "label": torch.tensor([sample['label']], dtype=torch.float32)
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping a malformed line in dataset. Error: {e}")
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ==============================================================
# ===== 修改点 2: train_router 现在也接收 feature_subset =====
# ==============================================================
def train_router(training_data_path, model_save_path, feature_subset: list, epochs=20, lr=1e-4, batch_size=32):
    print("\n" + "=" * 50 + f"\n🚀 Training the smart router with {len(feature_subset)} features...\n" + "=" * 50)

    # 1. 使用传入的特征子集来创建数据集
    dataset = RouterDataset(training_data_path, feature_subset=feature_subset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. 【关键】根据特征子集的数量，动态创建模型
    num_features = len(feature_subset)
    model = ComplexityPredictorNet(input_features=num_features).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. 训练循环 (保持不变)
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

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_preds / total_samples
        print(f"Epoch {epoch + 1:02d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")

    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ Training complete! Model saved to {model_save_path}")


if __name__ == "__main__":
    # 从环境变量中读取 Colab 传递过来的信息
    PROJECT_PATH = os.getenv('PROJECT_PATH_GDRIVE', '.')
    hf_token = os.getenv('HUGGINGFACE_TOKEN')

    training_file = os.path.join(PROJECT_PATH, "router_training_data_rich_features.jsonl")

    # 首先，确保富特征数据集存在
    if not os.path.exists(training_file):
        print(f"❌ 错误: 富特征数据文件 '{training_file}' 未找到!")
        print("   请先运行完整的数据生成流程来创建这个文件。")
        # 您可以在这里添加调用 generate_router_training_data 的逻辑
    else:
        # 您的特征重要性排名 (从高到低)
        all_18_features_ranked = [
            'last_avg_max_attention', 'last_max_entropy', 'last_concentration_std', 'variance_diff',
            'mid_max_entropy', 'last_entropy_std', 'mid_entropy_std', 'mid_variance_std',
            'mid_concentration_std', 'last_avg_variance', 'last_avg_entropy', 'mid_avg_max_attention',
            'mid_avg_entropy', 'mid_max_variance', 'last_max_variance', 'entropy_diff',
            'last_variance_std', 'mid_avg_variance'
        ]

        # --- 【【【实验选择区】】】---
        # 一次只取消注释一个实验，运行完成后，再注释掉，换下一个。

        # --- 实验1：只使用最重要的前5个特征 ---
        print("\n--- 正在运行实验1：Top 5 特征 ---")
        selected_features = all_18_features_ranked[:5]
        model_save_path = os.path.join(PROJECT_PATH, "router_model_top5.pth")
        # -----------------------------------

        # # --- 实验2：只使用最重要的前10个特征 ---
        # print("\n--- 正在运行实验2：Top 10 特征 ---")
        # selected_features = all_18_features_ranked[:10]
        # model_save_path = os.path.join(PROJECT_PATH, "router_model_top10.pth")
        # # ------------------------------------

        # # --- 实验3：使用全部18个特征 (作为对比基准) ---
        # print("\n--- 正在运行实验3：全部 18 个特征 ---")
        # selected_features = all_18_features_ranked
        # model_save_path = os.path.join(PROJECT_PATH, "router_model_all.pth")
        # # ------------------------------------

        # --- 执行选定的实验 ---
        train_router(training_data_path=training_file,
                     model_save_path=model_save_path,
                     feature_subset=selected_features)  # 将选定的特征列表传进去

    print("\n✅ 训练流程结束！")