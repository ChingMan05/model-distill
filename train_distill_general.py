import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
import json
from tqdm import tqdm
import torch.nn.functional as F

# 读取配置
with open("config_general.json", "r") as f:
    config = json.load(f)

device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

# 加载 teacher 和 student
print(" Loading models...")
teacher = AutoModelForCausalLM.from_pretrained(config["teacher_model"], torch_dtype=torch.float16, device_map="auto")
student = AutoModelForCausalLM.from_pretrained(config["student_model"], torch_dtype=torch.float16, device_map="auto")

teacher.eval()  # 不训练 teacher
tokenizer = AutoTokenizer.from_pretrained(config["student_model"], use_fast=False)

# 加载数据
print(" Loading dataset...")
with open(config["dataset_path"], "r") as f:
    raw_data = json.load(f)

def tokenize(examples):
    return tokenizer(
        [ex["instruction"] + "\n" + ex["input"] for ex in examples],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config["max_length"]
    )

data = tokenize(raw_data)
loader = DataLoader(list(zip(data["input_ids"], data["attention_mask"])), batch_size=config["batch_size"], shuffle=True)

optimizer = AdamW(student.parameters(), lr=config["learning_rate"])

# 蒸馏循环
print(" Starting distillation...")
for epoch in range(config["num_train_epochs"]):
    for step, (input_ids, attn_mask) in enumerate(tqdm(loader)):
        input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)

        with torch.no_grad():
            teacher_logits = teacher(input_ids=input_ids, attention_mask=attn_mask).logits / config["temperature"]

        student_logits = student(input_ids=input_ids, attention_mask=attn_mask).logits / config["temperature"]

        loss_ce = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean"
        ) * (config["temperature"] ** 2)

        optimizer.zero_grad()
        loss_ce.backward()
        optimizer.step()

        if step % config["logging_steps"] == 0:
            print(f"Epoch {epoch} | Step {step} | Loss: {loss_ce.item():.4f}")

torch.save(student.state_dict(), f"{config['output_dir']}/student_general.pt")
print(" 通用蒸馏完成！")
