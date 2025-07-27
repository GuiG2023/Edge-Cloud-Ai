import os
import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from common_utils import GSM8KAccuracyEvaluator, device, LearnedAttentionRouter, AccuracyValidator


def generate_router_training_data(evaluator, output_file):
    print("\n" + "=" * 50 + "\n🧠 Generating router training data...\n" + "=" * 50)
    evaluator._ensure_slm_loaded()
    slm_interface = evaluator.slm
    all_problems = evaluator.data_processor.samples
    print(f"📊 Processing {len(all_problems)} problems to generate features and labels.")

    training_samples = []
    # Use a temporary router instance just for its feature extraction method
    temp_feature_extractor = LearnedAttentionRouter("dummy_path.pth", device)
    validator = AccuracyValidator()

    for i, problem in enumerate(all_problems):
        if i % 20 == 0 and i > 0: print(f"   Progress: {i}/{len(all_problems)}")
        try:
            slm_response = slm_interface.predict(problem['question'])
            slm_answer = validator.extract_final_answer(slm_response)
            gt_answer = evaluator.data_processor.extract_answer(problem['answer'])
            is_slm_correct = validator.is_correct(slm_answer, gt_answer)

            label = 1.0 if not is_slm_correct else 0.0
            features = temp_feature_extractor.extract_core_features(
                problem['question'], slm_interface.model, slm_interface.tokenizer
            )
            training_samples.append({"features": features, "label": label})
        except Exception as e:
            print(f"   ⚠️ Skipped problem {i} due to error: {e}")
            continue

    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in training_samples: f.write(json.dumps(sample) + '\n')
    print(f"\n✅ Training data generation complete! {len(training_samples)} samples saved to {output_file}")


class RouterDataset(Dataset):
    def __init__(self, data_path):
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                feature_vector = [
                    sample['features']['avg_entropy'], sample['features']['entropy_std'],
                    sample['features']['max_entropy'],
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


def train_router(training_data_path, model_save_path, epochs=20, lr=1e-4, batch_size=32):
    from common_utils import ComplexityPredictorNet  # Local import
    print("\n" + "=" * 50 + "\n🚀 Training the smart router...\n" + "=" * 50)
    dataset = RouterDataset(training_data_path)
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
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_preds / total_samples
        print(f"Epoch {epoch + 1:02d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")

    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ Training complete! Model saved to {model_save_path}")


if __name__ == "__main__":
    PROJECT_PATH = os.getenv('PROJECT_PATH_GDRIVE', '.')
    hf_token = os.getenv('HUGGINGFACE_TOKEN')

    evaluator = GSM8KAccuracyEvaluator(hf_token=hf_token, max_samples=2000, project_path=PROJECT_PATH)

    training_file = os.path.join(PROJECT_PATH, "router_training_data.jsonl")
    model_file = os.path.join(PROJECT_PATH, "router_model.pth")

    generate_router_training_data(evaluator, output_file=training_file)
    train_router(training_data_path=training_file, model_save_path=model_file)