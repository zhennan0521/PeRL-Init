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
- [ ] Step 1: 跑通 base & LoRA example (异步和非异步)
  - [x] Step 1.1: full/LoRA × sync/async baseline (lr=2e-5)
    - 结论：async (offpolicy=2) 优于 sync，full 和 LoRA 均如此
  - [x] Step 1.2: LoRA sync lr sweep (5e-5, 1e-5, 5e-6, 1e-6)
    - 暂定：lr=5e-5 优于 lr=2e-5
  - [ ] Step 1.3: 多机 run LoRA async lr=5e-5 实验
- [ ] Step 2: 移植 PEFT 方法到 AReaL
- [ ] Step 3: 实现 MiLoRA++
