import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==================== 路径配置 ====================
teacher_path = "/root/autodl-tmp/models/Qwen2-7B"
student_original_path = "/root/autodl-tmp/models/Qwen2-1.5B"
student_distilled_path = "/root/autodl-tmp/model-distill/distilled_qwen2_1.5B_full_20251023_103726"
dataset_path = "/root/autodl-tmp/datasets/wikitext-2-raw-v1"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 1. 困惑度评估 (Perplexity) ====================
def calculate_perplexity(model, tokenizer, texts, max_length=256):
    """计算模型在测试集上的困惑度"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing perplexity"):
            if not text.strip():
                continue
                
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length
            ).to(device)
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # 累计 loss 和 token 数
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity

# ==================== 2. 输出相似度评估 (KL Divergence) ====================
def calculate_kl_divergence(teacher, student, tokenizer, texts, max_length=256):
    """计算学生和教师输出的 KL 散度"""
    teacher.eval()
    student.eval()
    
    total_kl = 0
    count = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing KL divergence"):
            if not text.strip():
                continue
                
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length
            ).to(device)
            
            # 获取 teacher 和 student 的 logits
            teacher_outputs = teacher(**inputs)
            student_outputs = student(**inputs)
            
            teacher_logits = teacher_outputs.logits
            student_logits = student_outputs.logits
            
            # 对齐词表大小
            min_vocab = min(teacher_logits.size(-1), student_logits.size(-1))
            teacher_logits = teacher_logits[:, :, :min_vocab]
            student_logits = student_logits[:, :, :min_vocab]
            
            # 计算 KL 散度
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            
            kl = (teacher_probs * (teacher_probs.log() - student_log_probs)).sum(dim=-1).mean()
            total_kl += kl.item()
            count += 1
    
    return total_kl / count if count > 0 else float('inf')

# ==================== 3. 生成质量评估 ====================
def evaluate_generation(model, tokenizer, prompts):
    """评估生成质量"""
    model.eval()
    results = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            "prompt": prompt,
            "generated": generated
        })
    
    return results

# ==================== 4. 主评估流程 ====================
def main():
    print("="*80)
    print("🔍 开始评估蒸馏效果")
    print("="*80)
    
    # 加载模型
    print("\n📦 Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
    
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    
    student_original = AutoModelForCausalLM.from_pretrained(
        student_original_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    
    student_distilled = AutoModelForCausalLM.from_pretrained(
        student_distilled_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    
    # 加载测试数据
    print("\n📚 Loading test dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=dataset_path)
    test_texts = [t for t in dataset["test"]["text"][:500] if t.strip()]  # 使用 500 条测试
    
    # ==================== 评估 1: 困惑度 ====================
    print("\n" + "="*80)
    print("📊 评估 1: 困惑度 (Perplexity) - 越低越好")
    print("="*80)
    
    print("\n🔹 Teacher (Qwen2-7B)...")
    teacher_ppl = calculate_perplexity(teacher, tokenizer, test_texts)
    
    print("\n🔹 Student Original (Qwen2-1.5B 原始)...")
    student_orig_ppl = calculate_perplexity(student_original, tokenizer, test_texts)
    
    print("\n🔹 Student Distilled (Qwen2-1.5B 蒸馏后)...")
    student_dist_ppl = calculate_perplexity(student_distilled, tokenizer, test_texts)
    
    print("\n" + "-"*80)
    print(f"Teacher Perplexity:           {teacher_ppl:.2f}")
    print(f"Student Original Perplexity:  {student_orig_ppl:.2f}")
    print(f"Student Distilled Perplexity: {student_dist_ppl:.2f}")
    print(f"Improvement:                  {student_orig_ppl - student_dist_ppl:.2f} ({'↓' if student_dist_ppl < student_orig_ppl else '↑'})")
    print(f"Gap to Teacher:               {student_dist_ppl - teacher_ppl:.2f}")
    print("-"*80)
    
    # ==================== 评估 2: KL 散度 ====================
    print("\n" + "="*80)
    print("📊 评估 2: 与教师的 KL 散度 - 越低越好（表示越接近教师）")
    print("="*80)
    
    print("\n🔹 Student Original vs Teacher...")
    kl_orig = calculate_kl_divergence(teacher, student_original, tokenizer, test_texts[:100])
    
    print("\n🔹 Student Distilled vs Teacher...")
    kl_dist = calculate_kl_divergence(teacher, student_distilled, tokenizer, test_texts[:100])
    
    print("\n" + "-"*80)
    print(f"Student Original KL:  {kl_orig:.4f}")
    print(f"Student Distilled KL: {kl_dist:.4f}")
    print(f"Improvement:          {kl_orig - kl_dist:.4f} ({'↓' if kl_dist < kl_orig else '↑'})")
    print("-"*80)
    
    # ==================== 评估 3: 生成质量 ====================
    print("\n" + "="*80)
    print("📊 评估 3: 生成质量对比")
    print("="*80)
    
    test_prompts = [
        "The history of artificial intelligence began",
        "In recent years, climate change has",
        "The key to successful machine learning is"
    ]
    
    print("\n🔹 Generating with Teacher...")
    teacher_results = evaluate_generation(teacher, tokenizer, test_prompts)
    
    print("\n🔹 Generating with Student Original...")
    student_orig_results = evaluate_generation(student_original, tokenizer, test_prompts)
    
    print("\n🔹 Generating with Student Distilled...")
    student_dist_results = evaluate_generation(student_distilled, tokenizer, test_prompts)
    
    # 打印对比结果
    for i, prompt in enumerate(test_prompts):
        print(f"\n{'='*80}")
        print(f"Prompt {i+1}: {prompt}")
        print(f"{'='*80}")
        print(f"\n🎓 Teacher:\n{teacher_results[i]['generated']}")
        print(f"\n📘 Student Original:\n{student_orig_results[i]['generated']}")
        print(f"\n📗 Student Distilled:\n{student_dist_results[i]['generated']}")
    
    # ==================== 总结 ====================
    print("\n" + "="*80)
    print("📝 评估总结")
    print("="*80)
    
    improvements = []
    if student_dist_ppl < student_orig_ppl:
        improvements.append(f"✅ 困惑度降低 {student_orig_ppl - student_dist_ppl:.2f}")
    else:
        improvements.append(f"❌ 困惑度上升 {student_dist_ppl - student_orig_ppl:.2f}")
    
    if kl_dist < kl_orig:
        improvements.append(f"✅ KL 散度降低 {kl_orig - kl_dist:.4f} (更接近教师)")
    else:
        improvements.append(f"❌ KL 散度上升 {kl_dist - kl_orig:.4f}")
    
    print("\n改进情况:")
    for imp in improvements:
        print(f"  {imp}")
    
    print("\n模型大小对比:")
    print(f"  Teacher:  7B 参数")
    print(f"  Student:  1.5B 参数 (21% 大小)")
    
    print("\n" + "="*80)
    print("✅ 评估完成！")
    print("="*80)

if __name__ == "__main__":
    main()