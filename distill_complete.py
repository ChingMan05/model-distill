import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from textbrewer import GeneralDistiller, TrainingConfig, DistillationConfig
import datetime

# ==================== 配置设备 ====================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 模型路径 ====================
teacher_path = "/root/autodl-tmp/models/Qwen2-7B"
student_path = "/root/autodl-tmp/models/Qwen2-1.5B"
dataset_path = "/root/autodl-tmp/datasets/wikitext-2-raw-v1"

# ==================== 加载 tokenizer ====================
print("🔹 Loading tokenizer and models ...")
tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)

teacher = AutoModelForCausalLM.from_pretrained(
    teacher_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

student = AutoModelForCausalLM.from_pretrained(
    student_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# ==================== 数据集加载 ====================
print("🔹 Loading dataset ...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=dataset_path)

def encode(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

tokenized = dataset.map(encode, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

# ==================== 数据选择 ====================
USE_FULL_DATA = True  # True 使用完整数据，False 使用子集
if USE_FULL_DATA:
    train_dataset = tokenized["train"]
    dataset_flag = "full"
    batch_size = 8
    gradient_accumulation_steps = 16
    num_epochs = 3
    print(f"📦 Using FULL dataset: {len(train_dataset)} samples")
else:
    train_dataset = tokenized["train"].select(range(5000))
    dataset_flag = "subset"
    batch_size = 2
    gradient_accumulation_steps = 8
    num_epochs = 1
    print(f"📦 Using SUBSET for quick test: {len(train_dataset)} samples")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ==================== 对齐 logits 工具函数 ====================
_align_printed = False
def align_logits(logits_S, logits_T):
    global _align_printed
    min_seq_len = min(logits_S.size(1), logits_T.size(1))
    logits_S_aligned = logits_S[:, :min_seq_len, :]
    logits_T_aligned = logits_T[:, :min_seq_len, :]
    
    min_vocab = min(logits_S_aligned.size(2), logits_T_aligned.size(2))
    logits_S_aligned = logits_S_aligned[:, :, :min_vocab]
    logits_T_aligned = logits_T_aligned[:, :, :min_vocab]
    
    if not _align_printed and logits_S.size() != logits_T.size():
        print(f"[ALIGN] Teacher {logits_T.size()} → Student {logits_S.size()} → Truncated to {logits_S_aligned.size()}")
        _align_printed = True
    
    return logits_S_aligned, logits_T_aligned

# ==================== adaptor ====================
def adaptor_T(batch, outputs):
    return {"logits": outputs.logits}
def adaptor_S(batch, outputs):
    return {"logits": outputs.logits}

# ==================== 自定义 KD loss ====================
def custom_kd_loss(logits_S, logits_T, temperature):
    logits_S, logits_T = align_logits(logits_S, logits_T)
    beta_logits_S = logits_S / temperature
    beta_logits_T = logits_T / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
    return loss * (temperature ** 2)

# ==================== 自动生成保存路径和日志路径 ====================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./logs_qwen2_{dataset_flag}_{timestamp}"
save_dir = f"./distilled_qwen2_1.5B_{dataset_flag}_{timestamp}"

# ==================== TextBrewer 配置 ====================
train_config = TrainingConfig(
    gradient_accumulation_steps=gradient_accumulation_steps,
    log_dir=log_dir,
    device="cuda"
)
distill_config = DistillationConfig(
    temperature=2.0,
    kd_loss_weight=1.0,
    hard_label_weight=0.0,
    kd_loss_type='ce',
)
optimizer = torch.optim.AdamW(student.parameters(), lr=2e-5)

# ==================== Patch GeneralDistiller.compute_loss ====================
import textbrewer.distiller_general as dg

def patched_compute_loss(self, results_S, results_T):
    losses_dict = dict()
    logits_list_S = results_S['logits']
    logits_list_T = results_T['logits']
    
    if not isinstance(logits_list_S, (list, tuple)):
        logits_list_S = [logits_list_S]
    if not isinstance(logits_list_T, (list, tuple)):
        logits_list_T = [logits_list_T]
    
    total_kd_loss = 0
    for l_S, l_T in zip(logits_list_S, logits_list_T):
        kd_loss = custom_kd_loss(l_S, l_T, self.d_config.temperature)
        total_kd_loss += kd_loss * self.d_config.kd_loss_weight
    
    losses_dict['unweighted_kd_loss'] = total_kd_loss
    return total_kd_loss, losses_dict

dg.GeneralDistiller.compute_loss = patched_compute_loss

# ==================== 定义蒸馏器 ====================
distiller = GeneralDistiller(
    train_config=train_config,
    distill_config=distill_config,
    model_T=teacher,
    model_S=student,
    adaptor_T=adaptor_T,
    adaptor_S=adaptor_S
)

# ==================== 训练 ====================
print("🚀 Starting dynamic distillation (with safe alignment)...")
print(f"📊 Training info:")
print(f"   - Total batches: {len(train_loader)}")
print(f"   - Batch size: {batch_size}")
print(f"   - Gradient accumulation: {gradient_accumulation_steps}")
print(f"   - Effective batch size: {batch_size * gradient_accumulation_steps}")
print(f"   - Total samples: {len(train_dataset)}")
print(f"   - Num epochs: {num_epochs}")
print(f"   - Estimated steps per epoch: {len(train_loader) // gradient_accumulation_steps}")
print()

distiller.train(
    optimizer=optimizer,
    dataloader=train_loader,
    num_epochs=num_epochs,
    max_grad_norm=1.0
)

# ==================== 保存 ====================
os.makedirs(save_dir, exist_ok=True)
student.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"✅ 蒸馏完成，模型保存至: {save_dir}")
