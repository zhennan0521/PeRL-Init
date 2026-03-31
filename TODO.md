# LoRA Init

## Principles

1. 代码管理：从 AReaL fork 然后 apply PR，**不要在 AReaL 仓库下写自己的训练脚本**（脚本不可能推到官方仓库）
2. 训练设置：DeepSeek-R1-Distill-Qwen-1.5B/7B + DAPO-Math-17k
3. 首先跑通 LoRA 的各种初始化方法
4. 先把 baseline 做 solid，再研究新方法

---

## Plan

### Step 1: "Light Up" Base & LoRA Example Using AReaL (1 day)

- **Data Recipe:** `DeepSeek-R1-Distill-Qwen-1.5B/7B` + DAPO (train) + AIME / etc.
- **Eval:** Load from MikaEval

### Step 2: Reconstruct PEFT Methods on AReaL (1-2 days)

- Copy & adapt from PEFT library

### Step 3: Implement MiLoRA++ on AReaL (1-2 days)

---

## Progress

- [x] 代码管理：fork AReaL，merge PR #1015 到 `lora_init` 分支，main 保持与上游同步
- [ ] Step 1: 跑通 base & LoRA example
- [ ] Step 2: 移植 PEFT 方法到 AReaL
- [ ] Step 3: 实现 MiLoRA++
