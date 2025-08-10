import os
import json
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from common_utils import GSM8KAccuracyEvaluator, device, LearnedAttentionRouter, AccuracyValidator


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


if __name__ == "__main__":
    PROJECT_PATH = os.getenv('PROJECT_PATH_GDRIVE', '.')
    hf_token = os.getenv('HUGGINGFACE_TOKEN')

    # evaluator = GSM8KAccuracyEvaluator(hf_token=hf_token, max_samples=100, project_path=PROJECT_PATH)
    #
    # training_file = os.path.join(PROJECT_PATH, "router_training_data.jsonl")
    # model_file = os.path.join(PROJECT_PATH, "router_model.pth")
    #
    # generate_router_training_data(evaluator, output_file=training_file)
    # train_router(training_data_path=training_file, model_save_path=model_file)

    training_file = os.path.join(PROJECT_PATH, "router_training_data.jsonl")

    # 目前特征重要性排名 (从高到低)
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
    # 确保数据已生成
    if not os.path.exists(training_file):
        print("❌ 错误：训练数据文件未找到，请先运行完整的数据生成流程。")
    else:
        # 您还需要稍微修改 train_router 和 RouterDataset 的定义，让它们能接收 feature_subset
        # 这里假设您已经修改完毕
        train_router(training_data_path=training_file,
                     model_save_path=model_save_path,
                     feature_subset=selected_features)  # 将选定的特征列表传进去