import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os, json, gc

# 尝试导入 Qwen2
try:
    from transformers import Qwen2ForCausalLM
    HAS_QWEN2 = True
except ImportError:
    HAS_QWEN2 = False

# 配置
with open("config_general.json", "r") as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_fp16 = config.get("fp16", True)
batch_size = config.get("batch_size", 1)
grad_accum_steps = config.get("gradient_accumulation_steps", 16)
num_epochs = config.get("num_train_epochs", 2)
temperature = config.get("temperature", 2.0)
alpha_ce = config.get("alpha_ce", 0.7)
alpha_hard = config.get("alpha_hard", 0.3)
save_steps = config.get("save_steps", 500)
log_steps = config.get("logging_steps", 100)
output_dir = config.get("output_dir", "./output_distill_general")
os.makedirs(output_dir, exist_ok=True)

print(f"[INFO] Device: {device}, FP16: {use_fp16}, Batch: {batch_size}, GradAccum: {grad_accum_steps}")

# 加载模型
def load_model(model_path, device='cpu', use_fp16=True):
    print(f"[STEP] Loading from {model_path}...")
    # CPU 必须使用 FP32，GPU 可以使用 FP16
    dtype = torch.float16 if (use_fp16 and device == 'cuda') else torch.float32
    
    if HAS_QWEN2:
        try:
            model = Qwen2ForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, low_cpu_mem_usage=True
            )
            print(f"[INFO] Loaded with Qwen2ForCausalLM ({dtype})")
            return model.to(device)
        except Exception as e:
            print(f"[WARN] Qwen2 failed: {e}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    print(f"[INFO] Loaded with AutoModel ({dtype})")
    return model.to(device)

print("[STEP] Loading Teacher (CPU in FP32)...")
teacher = load_model(config['teacher_model'], device='cpu', use_fp16=False)
teacher.eval()

print("[STEP] Loading Student (GPU in FP16)...")
student = load_model(config['student_model'], device=device, use_fp16=use_fp16)
student.train()

# Tokenizer & Dataset
tokenizer = AutoTokenizer.from_pretrained(config["student_model"], use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if os.path.exists(config["dataset_path"]):
    with open(config["dataset_path"], "r") as f:
        raw_data = json.load(f)
    texts = [ex["instruction"] + "\n" + ex.get("input", "") + "\n" + ex["output"] for ex in raw_data]
else:
    raw_data = load_dataset(config["dataset_path"])["train"]
    texts = [ex["instruction"] + "\n" + ex.get("input", "") + "\n" + ex["output"] for ex in raw_data]

encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, 
                      max_length=config.get("max_length", 256))
dataset = list(zip(encodings["input_ids"], encodings["attention_mask"]))
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(student.parameters(), lr=config["learning_rate"])
scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

# Teacher 前向 (CPU FP32)
@torch.no_grad()
def get_teacher_logits(input_ids, attn_mask):
    ids_cpu = input_ids.cpu()
    mask_cpu = attn_mask.cpu()
    # CPU 上运行，不使用 autocast
    outputs = teacher(input_ids=ids_cpu, attention_mask=mask_cpu)
    # 转换为 FP16 并移到 GPU
    return outputs.logits.half().to(device) if use_fp16 else outputs.logits.to(device)

# 训练循环
global_step = 0
optimizer.zero_grad()

for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
    epoch_loss = 0.0

    for step, (input_ids, attn_mask) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
        try:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            # Teacher logits (FP32 on CPU -> FP16 on GPU)
            teacher_logits = get_teacher_logits(input_ids, attn_mask)

            # Student forward
            with torch.cuda.amp.autocast(enabled=use_fp16):
                student_outputs = student(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
                student_logits = student_outputs.logits
                hard_loss = student_outputs.loss

                # KL loss
                min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
                t_logits = teacher_logits[..., :min_vocab] / temperature
                s_logits = student_logits[..., :min_vocab] / temperature
                
                kl_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(s_logits, dim=-1),
                    torch.nn.functional.softmax(t_logits, dim=-1),
                    reduction="batchmean"
                ) * (temperature ** 2)

                loss = (alpha_ce * kl_loss + alpha_hard * hard_loss) / grad_accum_steps

            # Backward
            scaler.scale(loss).backward()
            
            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * grad_accum_steps

            # 日志
            if global_step % log_steps == 0 and (step + 1) % grad_accum_steps == 0:
                print(f"[Step {global_step}] loss={loss.item()*grad_accum_steps:.4f} (KL={kl_loss.item():.4f}, Hard={hard_loss.item():.4f})")

            # 保存
            if global_step % save_steps == 0 and (step + 1) % grad_accum_steps == 0:
                save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                student.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"[SAVE] Checkpoint saved → {save_path}")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[WARN] OOM at step {step}, skipping...")
                torch.cuda.empty_cache()
                gc.collect()
                optimizer.zero_grad()
                continue
            else:
                raise e

    print(f"Epoch {epoch+1} finished. Avg loss={epoch_loss/len(loader):.4f}")

# 保存最终模型
final_path = os.path.join(output_dir, "final_model")
student.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"\n[✅ DONE] Model saved to {final_path}")
