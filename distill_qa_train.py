import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from textbrewer import GeneralDistiller, TrainingConfig, DistillationConfig
import datetime
import torch.nn.functional as F

# ==================== 配置 ====================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = "cuda" if torch.cuda.is_available() else "cpu"

teacher_path = "/root/autodl-tmp/models/Qwen2-7B"
student_path = "/root/autodl-tmp/models/Qwen2-1.5B"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./logs_qa_{timestamp}"
save_dir = f"./distilled_qwen2_1.5B_qa_{timestamp}"

# 自动创建目录
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.expanduser("~/.cache/huggingface"), exist_ok=True)

# ==================== 数据集 ====================
dataset = load_dataset("squad")
tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(examples):
    texts = []
    for q, c, ans in zip(examples["question"], examples["context"], examples["answers"]):
        answer_text = ans["text"][0] if ans["text"] else "No answer"
        texts.append(f"Question: {q}\nContext: {c}\nAnswer: {answer_text}")
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

train_dataset = tokenized["train"].select(range(3000))  # 快速实验
batch_size = 4
gradient_accumulation_steps = 8
num_epochs = 2

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ==================== 加载模型 ====================
teacher = AutoModelForCausalLM.from_pretrained(
    teacher_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
).eval()

student = AutoModelForCausalLM.from_pretrained(
    student_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

# ==================== 自定义 KD loss ====================
def align_logits(logits_S, logits_T):
    min_seq = min(logits_S.size(1), logits_T.size(1))
    min_vocab = min(logits_S.size(2), logits_T.size(2))
    return logits_S[:, :min_seq, :min_vocab], logits_T[:, :min_seq, :min_vocab]

def custom_kd_loss(logits_S, logits_T, temperature=2.0):
    logits_S, logits_T = align_logits(logits_S, logits_T)
    p_T = torch.nn.functional.softmax(logits_T / temperature, dim=-1)
    loss = -(p_T * torch.nn.functional.log_softmax(logits_S / temperature, dim=-1)).sum(dim=-1).mean()
    return loss * (temperature ** 2)

def adaptor_T(batch, outputs):
    return {"logits": outputs.logits}

def adaptor_S(batch, outputs):
    return {"logits": outputs.logits}

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

# Patch compute_loss
import textbrewer.distiller_general as dg
def patched_compute_loss(self, results_S, results_T):
    logits_list_S = results_S['logits']
    logits_list_T = results_T['logits']
    if not isinstance(logits_list_S, (list, tuple)):
        logits_list_S = [logits_list_S]
    if not isinstance(logits_list_T, (list, tuple)):
        logits_list_T = [logits_list_T]
    total_kd_loss = sum(custom_kd_loss(l_S, l_T, self.d_config.temperature)
                        for l_S, l_T in zip(logits_list_S, logits_list_T))
    return total_kd_loss, {"unweighted_kd_loss": total_kd_loss}

dg.GeneralDistiller.compute_loss = patched_compute_loss

distiller = GeneralDistiller(
    train_config=train_config,
    distill_config=distill_config,
    model_T=teacher,
    model_S=student,
    adaptor_T=adaptor_T,
    adaptor_S=adaptor_S
)

# ==================== 训练 ====================
distiller.train(optimizer=optimizer, dataloader=train_loader, num_epochs=num_epochs, max_grad_norm=1.0)

# ==================== 保存学生模型 ====================
student.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"✅ Distillation done! Saved to {save_dir}")
