# teacher_logits_topk.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os, json

# === 读取配置 ===
with open("config_general.json") as f:
    config = json.load(f)

teacher_path = config["teacher_model"]
dataset_path = config["dataset_path"]
output_dir = "teacher_logits_topk"
os.makedirs(output_dir, exist_ok=True)

batch_size = config.get("batch_size", 2)
top_k = config.get("top_k", 128)  # 默认只保存前128个token的logits

# === 加载模型与分词器 ===
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(teacher_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(teacher_path, torch_dtype=torch.float16).to(device)
model.eval()

# === 加载数据集 ===
dataset = load_dataset("json", data_files=dataset_path, split="train")

# === 提取 teacher logits ===
with torch.no_grad():
    for i, batch in enumerate(dataset):
        text = batch["instruction"] + "\n" + batch.get("input", "") + "\n" + batch["output"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits.squeeze(0).float().cpu()

        # 只保留 top-k logits
        top_values, top_indices = torch.topk(logits, top_k, dim=-1)

        torch.save({
            "input_ids": input_ids.cpu(),
            "attn_mask": attn_mask.cpu(),
            "top_indices": top_indices,  # 保存 top-k 的 token 索引
            "top_values": top_values     # 保存对应 logits 值
        }, os.path.join(output_dir, f"logits_{i:06d}.pt"))

        if i % 10 == 0:
            print(f"[{i}/{len(dataset)}] Saved top-{top_k} logits for sample {i}")

print(f"✅ Finished saving Top-{top_k} logits to {output_dir}")
