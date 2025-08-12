import os
import json

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from common_utils import GSM8KAccuracyEvaluator, device, LearnedAttentionRouter, AccuracyValidator, \
    ComplexityPredictorNet
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle  # 用于保存标准化处理器



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
                # # --- 【【【debug摄像头1号：在调用前打印参数】】】---
                # print("\n--- [CALLER SIDE] Preparing to call extract_core_features ---")
                # print(f"   - Arg 1 (question): type={type(problem['question'])}")
                # print(f"   - Arg 2 (model): type={type(slm_interface.model)}")
                # print(f"   - Arg 3 (tokenizer): type={type(slm_interface.tokenizer)}")
                # print(f"   - Arg 4 (slm_interface): type={type(slm_interface)}")
                # print("-----------------------------------------------------------------")
                # # --- 打印结束 ---
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
    def __init__(self, data_path, feature_subset: list):  # <--- 【核心修正】在这里接收 feature_subset
        self.samples = []
        self.feature_keys = feature_subset if feature_subset else [
            'entropy_mean',
            'entropy_std',
            'entropy_max'
        ]

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


# ========================================================================
# ===== 在 train_router.py 中，使用这个【最终稳健版】的训练函数 =====
# ========================================================================

def train_router(training_data_path, model_save_path, feature_subset, epochs=20, lr=1e-4, batch_size=32):
    # 导入完成这个函数所需的全部库
    from common_utils import ComplexityPredictorNet
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pickle
    import numpy as np
    import os

    print(f"\n🚀 Training the smart router with {len(feature_subset)} features...")

    # 1. 加载数据 (逻辑不变)
    dataset = RouterDataset(training_data_path, feature_subset=feature_subset)
    if len(dataset) == 0:
        print("❌ Error: Dataset is empty. Cannot start training.")
        return

    all_features = np.array([s['features'].numpy() for s in dataset])
    all_labels = np.array([s['label'].numpy() for s in dataset])

    # 检查原始特征中是否有nan/inf值
    if not np.all(np.isfinite(all_features)):
        print("⚠️ Warning: NaN or infinity found in raw features. Replacing with 0.")
        all_features = np.nan_to_num(all_features)

    # 2. 划分训练集和验证集 (80/20)，用于客观评估模型学习效果
    X_train, X_val, y_train, y_val = train_test_split(
        all_features, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    print(f"--- Data split: {len(X_train)} for training, {len(X_val)} for validation ---")

    # --- 【【【核心修复 1：特征标准化】】】---
    # 创建一个标准化处理器，并用【训练集】的数据进行拟合
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # 用同一个scaler来转换验证集
    X_val_scaled = scaler.transform(X_val)

    # 保存这个scaler，以便未来在评估时使用
    scaler_path = os.path.join(os.path.dirname(model_save_path), "router_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Feature scaler saved to {scaler_path}")
    # --- 特征标准化结束 ---

    # 4. 创建PyTorch的Dataset和DataLoader
    train_tensor_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                         torch.tensor(y_train, dtype=torch.float32))
    val_tensor_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32),
                                       torch.tensor(y_val, dtype=torch.float32))
    train_dataloader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_tensor_dataset, batch_size=batch_size)

    # 5. 模型、损失和优化器 (逻辑不变)
    model = ComplexityPredictorNet(input_features=len(feature_subset)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 6. 训练循环 (增加了稳定性保护)
    print("--- Starting training loop ---")
    for epoch in range(epochs):
        model.train()
        for features, labels in train_dataloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)

            # --- 【【【核心修复 2：检查并跳过NaN Loss】】】---
            if torch.isnan(loss):
                print(f"Epoch {epoch + 1}: Detected NaN loss. Skipping this batch.")
                continue

            loss.backward()

            # --- 【【【核心修复 3：梯度裁剪】】】---
            # 强制将过大的梯度“拉回”到一个合理的范围内，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        # 每个epoch后进行验证 (逻辑不变)
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for features, labels in val_dataloader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == labels.bool()).sum().item()
                val_total += labels.size(0)
        print(f"Epoch {epoch + 1:02d}/{epochs} | Val Acc: {val_correct / val_total if val_total > 0 else 0:.2%}")

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
    evaluator_for_data_gen = GSM8KAccuracyEvaluator(hf_token=hf_token, max_samples=200, project_path=PROJECT_PATH)

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
