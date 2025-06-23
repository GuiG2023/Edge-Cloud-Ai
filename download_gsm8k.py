import requests
import os
import json

print("📥 Downloading GSM8K dataset...")

# 创建目录
os.makedirs('gsm8k_data', exist_ok=True)

# 下载URL
url = 'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl'

try:
    print("⬇️ Fetching data from GitHub...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    
    # 保存文件
    filepath = 'gsm8k_data/train.jsonl'
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(response.text)
    
    print(f"✅ Successfully downloaded to {filepath}")
    
    # 验证文件
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"📊 Total samples: {len(lines)}")
        
        # 解析第一个样本
        if lines:
            first_sample = json.loads(lines[0])
            print(f"📝 Sample question: {first_sample['question'][:80]}...")
            print(f"📝 Sample answer: {first_sample['answer'][:80]}...")
    
    print("🎉 GSM8K download completed!")
    
except Exception as e:
    print(f"❌ Download failed: {e}")
    print("💡 You can run this script in Colab instead")

