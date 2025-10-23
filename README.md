# 🧠 Qwen2 知识蒸馏脚本（使用 TextBrewer 框架）

本项目使用 [TextBrewer](https://github.com/airaria/TextBrewer) 框架，对 **Qwen2-7B** 教师模型进行知识蒸馏，得到 **Qwen2-1.5B** 学生模型。  
蒸馏任务为 **语言建模（Language Modeling）**，在 **WikiText-2** 数据集上进行。

---

## 🚀 项目简介

### 🎯 目标

- 将 Qwen2-7B 的语言建模知识迁移到 Qwen2-1.5B
    
- 使用 **soft label 蒸馏（KD loss）**，提升学生模型的语言理解与生成能力
    
- 对齐教师与学生模型的 logits，避免维度不匹配问题
    

### ⚙️ 核心特性

- ✅ 使用 **TextBrewer** 框架进行知识蒸馏
    
- ✅ 自动对齐教师/学生模型的输出 logits
    
- ✅ 支持子集/完整数据训练模式切换
    
- ✅ 自定义 KD Loss 实现
    
- ✅ 自动保存蒸馏模型与 tokenizer
    

---

## 🧩 环境依赖

```bash
conda create -n distill python=3.10
conda activate distill

pip install torch transformers datasets textbrewer accelerate

```
建议使用 GPU 环境，并开启多卡训练（示例脚本默认使用 GPU 0 和 1）。

---

## 📁 目录结构

```bash
/root/autodl-tmp/
│
├── models/
│   ├── Qwen2-7B/           # 教师模型
│   └── Qwen2-1.5B/         # 学生模型
│
├── datasets/
│   └── wikitext-2-raw-v1/  # 数据集缓存目录
│
└── model-distill/
    ├── distill_qwen2.py     # 蒸馏主脚本
    └── README.md            # 本说明文件

```
---

## 📦 运行说明

### 1️⃣ 设置路径

在脚本中修改以下变量以匹配你的环境：

```python
teacher_path = "/root/autodl-tmp/models/Qwen2-7B"
student_path = "/root/autodl-tmp/models/Qwen2-1.5B"
dataset_path = "/root/autodl-tmp/datasets/wikitext-2-raw-v1"

```

---

### 2️⃣ 选择数据规模

快速测试可仅使用部分样本（默认 5000 条）：

```python
USE_FULL_DATA = False  # 改为 True 使用全部数据
```

---

### 3️⃣ 启动蒸馏训练

```python
python distill_qwen2.py
```

训练日志将自动保存至：

```python
./logs_qwen2/
```

---

## 🔍 训练配置摘要

|配置项|值|
|---|---|
|批大小|2|
|梯度累积步数|8|
|温度（Temperature）|2.0|
|KD Loss 权重|1.0|
|硬标签权重|0.0|
|优化器|AdamW|
|学习率|2e-5|
|训练轮数|1|

---

## 🧠 蒸馏机制说明

### 自定义 KD Loss

使用温度缩放的交叉熵损失（soft target）：

```python
loss = -(p_T * log_softmax(logits_S / T)).sum().mean() * T^2
```

### 对齐机制

蒸馏过程中，教师与学生的输出维度可能不同：

- **序列长度不一致** → 截断到最短长度
    
- **词表大小不一致** → 截断到最小词表维度
    
- 自动打印一次对齐信息，例如：
    
```css
    [ALIGN] Teacher torch.Size([2, 256, 152064]) → Student torch.Size([2, 256, 151936]) → Truncated to torch.Size([2, 256, 151936])

```
    

---

## 💾 模型保存

训练完成后自动保存：

`./distilled_qwen2_1.5B_fix/`

目录中包含：

- `pytorch_model.bin`：蒸馏后的学生模型权重
    
- `config.json`：模型配置文件
    
- `tokenizer.json` / `vocab.json`：分词器文件
    

---

## 📊 示例输出

```yaml
🚀 Starting dynamic distillation (with safe alignment)...
📊 Training info:
   - Total batches: 2500
   - Batch size: 2
   - Gradient accumulation: 8
   - Effective batch size: 16
   - Total samples: 5000
   - Estimated steps: 312

[ALIGN] Teacher torch.Size([2, 256, 152064]) → Student torch.Size([2, 256, 151936]) → Truncated to torch.Size([2, 256, 151936])
✅ 蒸馏完成，模型保存至: ./distilled_qwen2_1.5B_fix

```



