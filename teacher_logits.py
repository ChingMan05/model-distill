# teacher_logits_dump.py
import torch, json, time, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

with open("config_general.json") as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

teacher = AutoModelForCausalLM.from_pretrained(
    config["teacher_model"],
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="/root/autodl-tmp/offload_teacher",
    offload_state_dict=True,
    offload_buffers=True
)
teacher.config.use_cache = False
teacher.eval()

with open(config["dataset_path"], "r") as f:
    raw = json.load(f)
tokenizer = AutoTokenizer.from_pretrained(config["student_model"], use_fast=False)

def tokenize(examples):
    return tokenizer([ex["instruction"] + "\n" + ex.get("input","") for ex in examples],
                     return_tensors="pt", padding=True, truncation=True,
                     max_length=config.get("max_length", 256))
data = tokenize(raw)
loader = DataLoader(list(zip(data["input_ids"], data["attention_mask"])),
                    batch_size=config.get("batch_size", 1), shuffle=False)

os.makedirs("teacher_logits", exist_ok=True)
print("Start dumping teacher logits (this may take long)...")
for i, (input_ids, attn_mask) in enumerate(loader):
    t0 = time.time()
    with torch.no_grad():
        out = teacher(input_ids=input_ids.to(teacher.device),
                      attention_mask=attn_mask.to(teacher.device)).logits
    out = out.cpu()
    torch.save({"logits": out, "input_ids": input_ids, "attn_mask": attn_mask}, f"teacher_logits/logits_{i:06d}.pt")
    print(f"Saved batch {i} in {time.time()-t0:.2f}s")
print("All teacher logits saved to teacher_logits/*.pt")
