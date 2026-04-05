# LoRA Init

## Principles

1. 代码管理：从 AReaL fork 然后 apply PR，**不要在 AReaL 仓库下写自己的训练脚本**（脚本不可能推到官方仓库）
2. 训练设置：DeepSeek-R1-Distill-Qwen-1.5B/7B + DAPO-Math-17k
3. 首先跑通 LoRA 的各种初始化方法
4. 先把 baseline 做 solid，再研究新方法

---

## Plan

### Step 1: "Light Up" Base & LoRA Example Using AReaL ✅

- **Data Recipe:** `DeepSeek-R1-Distill-Qwen-1.5B/7B` + DAPO (train) + AIME / etc.
- **Eval:** Load from MikaEval

### Step 2: Reconstruct PEFT Methods on AReaL

#### Step 2.1: 迁移各类 PEFT 方法（从 PeRL 的 module 实现借鉴）

从 `modules/PeRL/modules/trl/perl/lora/` 迁移到 AReaL 的 LoRA 框架。按从易到难排序：

**仅改初始化（不改 forward/optimizer，最简单）：**

| # | 方法 | PeRL 实现 | 迁移难度 | 说明 |
|---|------|-----------|---------|------|
| 1 | rsLoRA | `adapter.py:apply_rslora` | ⭐ | 只改 scaling: `α/√r` 代替 `α/r`，改一行 |
| 2 | PiSSA | `adapter.py:apply_pissa` | ⭐⭐ | SVD 取主成分初始化 A/B，需修改 base weight（W -= BA） |
| 3 | MiLoRA | `milora.py` | ⭐⭐ | SVD 取最小奇异值方向初始化 A/B，需修改 base weight |
| 4 | MiLoRA++ | `milora_plus.py` | ⭐⭐ | SVD 取方向初始化 A（正交），B=0，不改 base weight |

**改 forward 或结构（中等）：**

| # | 方法 | PeRL 实现 | 迁移难度 | 说明 |
|---|------|-----------|---------|------|
| 5 | DoRA | `adapter.py:apply_dora` | ⭐⭐⭐ | 分解 weight 为 magnitude + direction，forward 额外计算 norm |
| 6 | SliceFine | `slicefine.py` | ⭐⭐⭐ | 完全不同的结构：切片 weight 矩阵，训练一个 slice，需自定义 layer |

**改 optimizer（需动优化器逻辑）：**

| # | 方法 | PeRL 实现 | 迁移难度 | 说明 |
|---|------|-----------|---------|------|
| 7 | LoRA+ | `adapter.py:apply_lora_plus` | ⭐⭐⭐ | A/B 用不同学习率，需改 optimizer param groups |
| 8 | LoRA-FA | `adapter.py:apply_lorafa` | ⭐⭐⭐ | 冻结 A 只训 B，需改 optimizer + 可能影响 grad sync |
| 9 | AdaLoRA | `adapter.py:apply_adalora` | ⭐⭐⭐⭐ | 动态调整 rank（SVD 重参数化），需 step callback，侵入性最强 |

**其他方法（结构差异大，优先级低）：**

| # | 方法 | PeRL 实现 | 迁移难度 | 说明 |
|---|------|-----------|---------|------|
| 10 | IA3 | `adapter.py:apply_IA3` | ⭐⭐⭐ | 不是 LoRA，是逐元素缩放，需不同的 layer 实现 |
| 11 | VeRA | `adapter.py:apply_vera` | ⭐⭐⭐ | 共享随机矩阵 + 可训练对角缩放，结构不同 |
| 12 | LayerNorm Tuning | `adapter.py:apply_layernorm` | ⭐⭐ | 只训 LayerNorm 参数，不涉及 LoRA |
| 13 | HRA | `adapter.py:apply_hra` | ⭐⭐⭐ | Householder 反射适配器，结构不同 |
| 14 | MiSS | `adapter.py:apply_miss` | ⭐⭐⭐ | PEFT MissConfig，需确认 AReaL 兼容性 |

#### Step 2.2: 补充评测

### Step 3: 实现 MiLoRA++

---

## Progress

- [x] Step 1: 跑通 base & LoRA example (异步和非异步)
  - [x] Step 1.1: full/LoRA × sync/async baseline (lr=2e-5)
    - 结论：async (offpolicy=2) 优于 sync，full 和 LoRA 均如此
  - [x] Step 1.2: LoRA sync lr sweep (5e-5, 1e-5, 5e-6, 1e-6)
    - 暂定：lr=5e-5 优于 lr=2e-5 (full的2e-5好于1e-5)
  - [x] Step 1.3: 单机 run LoRA async lr=5e-5 实验
  - [x] Step 1.4: 摸清现在lora修改逻辑，看看实现有无bug，以及确认后续peft接口
    - [x] Step 1.4.1: 启动 tp=2 dp=2 实验验证 sync_lora_grads bug（详见 pitfalls.md Bug #2: tp=1时LoRA梯度不同步）
    - [x] Step 1.4.2: 多机bug修复 （cluster nfs_record_root: ${cluster.fileroot}/name_resolve）
    - [x] Step 1.4.3: LoRA 逻辑实现排查（plain tensor设计、checkpoint save/load、weight update 全链路 review）
      - 结论：6 个已知 bug 确认，无新 bug。详见 tutorial.md
  - [ ] Step 1.5: 确认 baseline setting 🔄
    - 评估中：sync vs async、lr 大小（5e-5 vs 2e-5）
    - 4node sync 实验已配置，待跑
- [ ] Step 2: 移植 PEFT 方法到 AReaL
  - [ ] Step 2.1: 迁移各类 PEFT 方法（从 PeRL 借鉴）
    - [ ] rsLoRA（改 scaling 一行）
    - [ ] PiSSA（SVD 主成分初始化，改 base weight）
    - [ ] MiLoRA（SVD 最小奇异值初始化，改 base weight）
    - [ ] MiLoRA++（SVD 方向初始化 A，B=0，不改 base weight）
    - [ ] DoRA（magnitude + direction 分解，改 forward）
    - [ ] SliceFine（切片 weight，自定义 layer）
    - [ ] LoRA+（A/B 不同学习率，改 optimizer）
    - [ ] LoRA-FA（冻结 A 只训 B，改 optimizer）
    - [ ] AdaLoRA（动态 rank，侵入性强）
  - [ ] Step 2.2: 补充评测
- [ ] Step 3: 实现 MiLoRA++
