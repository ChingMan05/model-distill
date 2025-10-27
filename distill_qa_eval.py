import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from evaluate import load
import json

# ==================== 配置 ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = "/root/autodl-tmp/model-distill/distilled_qwen2_1.5B_qa_20251027_184528"  # 换成你的模型路径

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
student = AutoModelForCausalLM.from_pretrained(
    model_dir, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
).eval()

# ==================== 数据集 ====================
dataset = load_dataset("squad")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(examples):
    texts = []
    for q, c in zip(examples["question"], examples["context"]):
        texts.append(f"Question: {q}\nContext: {c}\nAnswer:")
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["validation"].column_names)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
test_dataset = tokenized["validation"].select(range(200))  # 测 200 条即可
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

# ==================== 评估指标 ====================
rouge = load("rouge")
bleu = load("bleu")

all_preds, all_refs = [], []

# ==================== 生成预测 ====================
for batch in tqdm(test_loader, desc="Evaluating QA (Generative)"):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with torch.no_grad():
        outputs = student.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.7
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for text in decoded:
        # 抽取生成答案部分
        ans = text.split("Answer:")[-1].strip() if "Answer:" in text else text.strip()
        all_preds.append(ans)

# 提取参考答案
for i in range(len(all_preds)):
    ref_texts = dataset["validation"][i]["answers"]["text"]
    all_refs.append(ref_texts[0] if ref_texts else "")

# ==================== 计算生成式指标 ====================
rouge_result = rouge.compute(predictions=all_preds, references=all_refs)
bleu_result = bleu.compute(predictions=all_preds, references=[[r] for r in all_refs])

# ==================== 输出结果 ====================
print("📊 Generative QA Evaluation Results:")
print(f"ROUGE-1: {rouge_result['rouge1']:.4f}")
print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")
print(f"BLEU:    {bleu_result['bleu']:.4f}")
print(f"Avg Length (pred/ref): {sum(len(p.split()) for p in all_preds)/len(all_preds):.1f} / {sum(len(r.split()) for r in all_refs)/len(all_refs):.1f}")

# ==================== 保存结果 ====================
os.makedirs("eval_results", exist_ok=True)
result_file = "eval_results/qa_eval_rouge_bleu.json"
with open(result_file, "w") as f:
    json.dump({
        "ROUGE": rouge_result,
        "BLEU": bleu_result,
        "Sample": list(zip(all_preds[:5], all_refs[:5]))
    }, f, indent=2)

print(f"✅ Evaluation done! Results saved to {result_file}")
