import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
import json
from tqdm import tqdm
import torch.nn.functional as F
import os

# =======================
# 配置读取
# =======================
with open("config_general.json", "r") as f:
    config = json.load(f)

# GPU 设备，主要给 Teacher 用，Student 全部放 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("/root/autodl-tmp/offload_teacher", exist_ok=True)

# =======================
# 加载 Teacher（GPU offload）+ Student（CPU）
# =======================
print(" Loading Teacher model (offload)...")
teacher = AutoModelForCausalLM.from_pretrained(
    config["teacher_model"],
    torch_dtype=torch.float16,
    device_map="auto",               # 自动放到 GPU/CPU
    offload_folder="/root/autodl-tmp/offload_teacher",
    offload_state_dict=True,
    offload_buffers=True
)
teacher.eval()  # Teacher 不训练

print(" Loading Student model (CPU)...")
student = AutoModelForCausalLM.from_pretrained(
    config["student_model"],
    torch_dtype=torch.float16,
    device_map="cpu",  # 全部放 CPU
)
student.train()

# =======================
# Tokenizer
# =======================
tokenizer = AutoTokenizer.from_pretrained(config["student_model"], use_fast=False)

# =======================
# 加载数据
# =======================
print(" Loading dataset...")
with open(config["dataset_path"], "r") as f:
    raw_data = json.load(f)

def tokenize(examples):
    return tokenizer(
        [ex["instruction"] + "\n" + ex.get("input", "") for ex in examples],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.get("max_length", 256)  # 小 max_length
    )

data = tokenize(raw_data)

# DataLoader，batch_size 建议小一点，比如 1-2
loader = DataLoader(
    list(zip(data["input_ids"], data["attention_mask"])),
    batch_size=config.get("batch_size", 1),
    shuffle=True
)

# =======================
# 优化器
# =======================
optimizer = AdamW(student.parameters(), lr=config.get("learning_rate", 5e-5))

# =======================
# 蒸馏循环
# =======================
print(" Starting distillation...")

temperature = config.get("temperature", 2.0)
logging_steps = config.get("logging_steps", 10)
num_epochs = config.get("num_train_epochs", 1)

for epoch in range(num_epochs):
    for step, (input_ids, attn_mask) in enumerate(tqdm(loader)):
        # input_ids 和 attn_mask 都在 CPU
        input_ids = input_ids
        attn_mask = attn_mask

        # Teacher 前向（offload 自动管理）
        with torch.no_grad():
            input_ids_teacher = input_ids.to(teacher.device)
            attn_mask_teacher = attn_mask.to(teacher.device)
            teacher_logits = teacher(input_ids=input_ids_teacher, attention_mask=attn_mask_teacher).logits / temperature
            
        teacher_logits = teacher_logits.to("cpu")
        # Student 前向
        student_logits = student(input_ids=input_ids, attention_mask=attn_mask).logits / temperature

        # KL loss
        loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean"
        ) * (temperature ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % logging_steps == 0:
            print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

# =======================
# 保存 Student
# =======================
output_dir = config.get("output_dir", "./output")
os.makedirs(output_dir, exist_ok=True)
torch.save(student.state_dict(), f"{output_dir}/student_general.pt")
print(" Distillation completed! Student saved.")
