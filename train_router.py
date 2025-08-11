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
                # --- 【【【新的标签逻辑】】】---
                steps = evaluator.data_processor.count_solution_steps(problem['answer'])
                label = 1.0 if steps > 6 else 0.0  # 使用步骤数作为客观标签
                # --- 【【【摄像头1号：在调用前打印参数】】】---
                print("\n--- [CALLER SIDE] Preparing to call extract_core_features ---")
                print(f"   - Arg 1 (question): type={type(problem['question'])}")
                print(f"   - Arg 2 (model): type={type(slm_interface.model)}")
                print(f"   - Arg 3 (tokenizer): type={type(slm_interface.tokenizer)}")
                print(f"   - Arg 4 (slm_interface): type={type(slm_interface)}")
                print("-----------------------------------------------------------------")
                # --- 打印结束 ---
                # --- 【【【新的特征提取调用】】】---
                features = temp_feature_extractor.extract_core_features(
                    problem['question'],
                    slm_interface.model,
                    slm_interface.tokenizer,
                    slm_interface
                )
                # --- 【【【核心修改：不再依赖SLM的对错】】】---

                # 1. 直接从问题的标准答案中计算解题步骤数
                # problem['answer'] 此时还是原始的、包含解题步骤的答案文本
                # steps = evaluator.data_processor.count_solution_steps(problem['answer'])

                # 2. 根据步骤数，设定一个清晰、客观的“复杂”标签
                # 这里的阈值“4”是一个很好的起点，您可以后续进行敏感性分析
                # label = 1.0 if steps > 6 else 0.0

                # ----------------------------------------------------

                # 特征提取部分保持不变，依然需要SLM的“思考过程”
                # features = temp_feature_extractor.extract_core_features(
                #     problem['question'], slm_interface.model, slm_interface.tokenizer
                # )

                # 如果特征提取失败，则跳过该样本
                if not features:
                    print(f"\n   ⚠️ Skipped problem #{i} due to feature extraction failure.")
                    continue

                # 后续的计数、保存逻辑保持不变
                if label == 0.0:
                    simple_label_count += 1
                else:
                    complex_label_count += 1

                sample_to_save = {
                    "question": problem['question'],
                    "features": features,
                    "label": label
                }

                f.write(json.dumps(sample_to_save) + '\n')
                f.flush()
                processed_count += 1
                processed_samples.add(problem['question'])
                # slm_response = slm_interface.predict(problem['question'])
                # slm_answer = validator.extract_final_answer(slm_response)
                # gt_answer = evaluator.data_processor.extract_answer(problem['answer'])
                # is_slm_correct = validator.is_correct(slm_answer, gt_answer)
                #
                # label = 1.0 if not is_slm_correct else 0.0
                #
                # # --- 【【【新增】】】更新计数器 ---
                # if label == 0.0:
                #     simple_label_count += 1
                # else:
                #     complex_label_count += 1
                # # --------------------------------
                #
                # features = temp_feature_extractor.extract_core_features(
                #     problem['question'], slm_interface.model, slm_interface.tokenizer
                # )
                #
                # sample_to_save = {"question": problem['question'], "features": features, "label": label}
                #
                # f.write(json.dumps(sample_to_save) + '\n')
                # f.flush()
                # processed_count += 1
                # processed_samples.add(problem['question'])

                # --- 【【【最终版：增加详细进度报告】】】---
                # 每处理20个样本，或者在第一个和最后一个时，打印一次清晰的进度
                if current_progress % 20 == 0 or current_progress == 1 or current_progress == total_to_process:
                    # --- 1. 【新的速度计算逻辑】 ---
                    # 计算从脚本开始到现在，新处理了多少样本
                    newly_processed_count = processed_count - (simple_label_count + complex_label_count)

                    elapsed_time = time.time() - start_time
                    samples_per_second = newly_processed_count / elapsed_time if elapsed_time > 0 else 0

                    print(f"\n--- Progress Update ---")
                    print(f"   Processed: {current_progress}/{total_to_process}")

                    # --- 2. 【新的标签含义说明】 ---
                    print(
                        f"   Label Counts: Simple (Steps < 6) = {simple_label_count}, Complex (Steps >= 6) = {complex_label_count}")

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
    def __init__(self, data_path, feature_subset=None):
        self.samples = []
        self.feature_keys = feature_subset if feature_subset else ['entropy_mean', 'entropy_std', 'entropy_max',
                                                                   'entropy_trend']
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


# ==============================================================
# ===== 修改点 2: train_router 现在也接收 feature_subset =====
# ==============================================================
def train_router(training_data_path, model_save_path, feature_subset: list, epochs=20, lr=1e-4, batch_size=32):
    print("\n" + "=" * 50 + f"\n🚀 Training the smart router with {len(feature_subset)} features...\n" + "=" * 50)

    # 1. 使用传入的特征子集来创建数据集
    dataset = RouterDataset(training_data_path, feature_subset=feature_subset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. 【关键】根据特征子集的数量，动态创建模型
    num_features = len(feature_subset) if feature_subset else 4
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


# ========================================================================
# ===== 在 train_router.py 底部，使用这个【最终简化版】的主流程 =====
# ========================================================================

if __name__ == "__main__":
    # --- 1. 从环境变量中读取 Colab 传递过来的信息 ---
    PROJECT_PATH = os.getenv('PROJECT_PATH_GDRIVE', '.')
    hf_token = os.getenv('HUGGINGFACE_TOKEN')

    # --- 2. 定义文件路径 ---
    # 建议使用新文件名，清晰地表明这是基于动态特征的
    training_file = os.path.join(PROJECT_PATH, "router_training_data_dynamic.jsonl")
    model_file = os.path.join(PROJECT_PATH, "router_model_dynamic.pth")

    # --- 3. 执行数据生成 ---
    # 初始化评估器实例，用于数据生成
    # max_samples 决定了您要生成多少训练数据，2000是一个很好的起点
    evaluator_for_data_gen = GSM8KAccuracyEvaluator(hf_token=hf_token, max_samples=2000, project_path=PROJECT_PATH)

    # 调用数据生成函数。
    # 它会使用我们最新的“动态特征提取”和“步骤数标签”逻辑。
    # 如果数据已存在，这步会因“断点续传”而很快完成。
    generate_router_training_data(evaluator_for_data_gen, output_file=training_file)

    # --- 4. 执行模型训练 ---
    # 只有在数据文件确认存在后，才继续进行
    if os.path.exists(training_file) and os.path.getsize(training_file) > 0:

        # 定义我们新的4个动态特征
        dynamic_features = ['entropy_mean', 'entropy_std', 'entropy_max', 'entropy_trend']

        # 直接调用训练函数，使用全部的动态特征
        train_router(training_data_path=training_file,
                     model_save_path=model_file,
                     feature_subset=dynamic_features)
    else:
        print(f"❌ 关键错误：数据文件 '{training_file}' 未找到或为空，训练无法继续。")

    print("\n✅ 训练流程结束！")
