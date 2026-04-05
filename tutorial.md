# AReaL 框架 Code Map

> 所有路径相对于 `modules/AReaL/`，行号格式 `[file:L123]`。
> 分两大部分：Part A 是框架全流程（启动→训练→结束），Part B 是 LoRA 实现细节。

---

# Part A: AReaL 框架全流程

## 0. 启动入口

### 0.1 CLI 入口

[scripts/run_rl.py:31-60]（在 PeRL-Init 仓库，不在 AReaL 内）

- L32: `load_expr_config(args, GRPOConfig)` — 解析 yaml + 命令行覆盖 → 生成 `GRPOConfig` 对象
- L34-35: 加载 train/valid dataset
- L46-56: 创建 `PPOTrainer` 上下文 → 调用 `trainer.train(workflow=RLVRWorkflow)`

config 解析: [areal/api/cli_args.py:L2343] `load_expr_config` — 用 OmegaConf 合并 yaml + CLI overrides

### 0.2 PPOTrainer 初始化（核心编排器）

[areal/trainer/rl_trainer.py:95-290]

**PPOTrainer.init 按顺序做了这些事：**


| 2步骤 | 行号       | 做什么                                                                         |
| --- | -------- | --------------------------------------------------------------------------- |
| 1   | L113     | `_init_scheduler()` — 根据 `scheduler.type` 创建 Local/Ray/Slurm 调度器            |
| 2   | L119-121 | 解析 `ModelAllocation` — 从 `backend: archon:d8` 提取 backend=archon, dp=8 等     |
| 3   | L130     | `_create_train_engine(actor_config)` — 创建 actor（选 Archon/FSDP/Megatron）     |
| 4   | L140     | 同上创建 ref model（如果 kl_ctl > 0）                                               |
| 5   | L168     | 创建 dataloader                                                               |
| 6   | L190     | `actor.initialize(ft_spec=ft_spec, role="actor")` — **初始化引擎（加载模型、并行化、优化器）** |
| 7   | L205     | `_save_initial_lora_weights()` — LoRA 模式下保存初始 adapter 给 SGLang 预加载          |
| 8   | L208     | `_init_rollout()` — 创建 RolloutController，启动 SGLang 推理服务                     |
| 9   | L227-267 | 构造 `WeightUpdateMeta`（disk 或 xccl 模式的元信息）                                   |
| 10  | L267     | `actor.connect_engine(rollout, weight_update_meta)` — 连接训练和推理引擎             |
| 11  | L270-288 | 初始化 Evaluator, Saver, RecoverHandler, StatsLogger                           |


### 0.3 _create_train_engine（创建训练引擎）

[areal/trainer/rl_trainer.py:623-648]

- L635-638: `backend == "archon"` → 选择 `ArchonPPOActor`
- L643-644: **single-controller 模式**（Ray/Local）→ `actor_cls.as_controller(config, scheduler)` 返回 `PPOActorController`（一个 `TrainController` 子类）
- L647: `actor.create_process_group()` — 初始化分布式通信

关键区别：

- **Single-controller**: 主进程是 controller，通过 RPC 调远端 worker 上的 engine
- **Multi-controller (SPMD)**: 每个进程直接运行 engine，无 controller 层

### 0.4 TrainController 初始化引擎

[areal/infra/controller/train_controller.py:186-257]

`TrainController.initialize()`:

- L195-210: 通过 scheduler 创建 N 个 worker（Ray actor / 本地进程）
- L215: 设置 MASTER_ADDR / MASTER_PORT（用于 `dist.init_process_group`）
- L232-241: 在每个 worker 上实例化 engine（`ArchonPPOActor.__init_`_）
- L242-256: 在每个 worker 上调 `engine.initialize(ft_spec)`

### 0.5 ArchonEngine 初始化（每个 worker 内部）

[areal/experimental/engine/archon_engine.py:296-340]

`ArchonEngine.initialize()`:


| 步骤  | 行号   | 做什么                                                                           |
| --- | ---- | ----------------------------------------------------------------------------- |
| 1   | L300 | `_create_device_model()` — 在 meta device 上创建模型结构                              |
| 2   | L320 | `prepare_training_config()` — 构建 AC config, 决定 enable_compile, pad_to_maximum |
| 3   | L330 | `_setup_parallelism(ac_config, enable_compile)` — 应用并行化（见 Part B §2.2）        |
| 4   | L332 | `dist.barrier()` — 等所有 rank 完成并行化                                             |
| 5   | L334 | `_load_weights()` — 加载 HF checkpoint，FSDP2 物化参数                               |
| 6   | L336 | `_freeze_non_lora_params()` — LoRA 时冻结 base weight（见 Part B §2.4）             |
| 7   | L338 | `_create_optimizer(ft_spec)` — 创建 Adam 优化器 + LR scheduler                     |


### 0.6 RolloutController 初始化

[areal/infra/controller/rollout_controller.py:156-310]

- 通过 scheduler 创建 24 个 SGLang worker
- 每个 worker 启动 SGLang HTTP server
- 如果 LoRA 模式：SGLang 启动时加载 initial_lora adapter

---

## 1. 训练主循环

### 1.1 train() 总控

[areal/trainer/rl_trainer.py:292-541]

```
for global_step in range(start_step, max_steps):    # L329
    ① prepare_batch (rollout)                        # L349-356
    ② compute_logp (recompute proximal π)            # L381-384
    ③ compute_advantages                             # L426-427
    ④ ppo_update (train step)                        # L439-441
    ⑤ rollout.pause()                                # L457
    ⑥ update_weights                                 # L468-477
    ⑦ save_hf / save_recover                         # L487-499
    ⑧ evaluate                                       # L509-515
    ⑨ rollout.resume()                               # L539
```

### 1.2 ① prepare_batch（异步 Rollout 收集）

[areal/infra/controller/rollout_controller.py:936-980]

- L957-967: `task_input_generator()` — 无限循环从 dataloader 取数据，yield 成 rollout task
- L974-976: `dispatcher.active_submit_and_wait(generator, batch_size)` — 持续提交 + 等待 batch_size 个结果返回
- 内部：SGLang 异步生成 response，通过 callback 返回结果
- **关键**：异步模式下 rollout 与上一步训练重叠执行（`max_head_offpolicyness=2` 允许最多用落后 2 个版本的模型）

### 1.3 ② recompute_logp（可选）

[areal/trainer/rl_trainer.py:372-384]

- 当 `recompute_logprob=true` 且 `use_decoupled_loss=true` 时
- 用当前 actor forward 重新计算 log_p（作为 proximal policy π_prox）
- 走 `engine.eval_batch()` → forward only

### 1.4 ③ compute_advantages

[areal/trainer/ppo/actor.py:133-242]

- GRPO/DAPO 风格：group-level reward normalization → token-level advantage

### 1.5 ④ ppo_update（核心训练步）

[areal/trainer/ppo/actor.py:244-355]

- L245: `batched_call(self._ppo_update, data)` — 逐 batch 处理
- L334-354: 对每个 minibatch 调 `engine.train_batch(mb, loss_fn, loss_weight_fn)`
- `engine.train_batch` 见 Part B §3.1

### 1.6 ⑤⑥ pause → update_weights → resume

[areal/trainer/rl_trainer.py:457-539]

- L457: `rollout.pause()` — 停止接受新 rollout 请求
- L470: `actor.update_weights(versioned_meta)` — 保存 LoRA + 通知 SGLang 加载（见 Part B §5）
- L472-477: 更新 actor/critic/rollout 的 version 号
- L539: `rollout.resume()` — 恢复 rollout，SGLang 用新模型继续生成

### 1.7 ⑦⑧ save + evaluate

- `_save_hf` [rl_trainer.py:789]: 定期保存 HF 格式 checkpoint
- `_evaluate` [rl_trainer.py:861]: 用 eval_rollout 跑验证集

---

## 2. 架构总览图

```
┌─────────────────────────────────────────────────────────┐
│ Head Node (Main Process)                                 │
│                                                          │
│  run_rl.py                                               │
│    └── PPOTrainer                                        │
│          ├── PPOActorController (TrainController)         │
│          │     └── RayScheduler ──RPC──→ Worker 0..7     │
│          │           每个 Worker 运行 ArchonPPOActor      │
│          │             └── ArchonEngine (FSDP2 + LoRA)   │
│          │                   └── train_batch / save ...  │
│          │                                               │
│          ├── RolloutController                            │
│          │     └── RayScheduler ──RPC──→ SGLang 0..23    │
│          │           每个 SGLang server: HTTP generation  │
│          │                                               │
│          └── train loop:                                 │
│                prepare_batch (async rollout)              │
│                → recompute_logp → advantages             │
│                → ppo_update → pause                      │
│                → update_weights → save → eval            │
│                → resume                                  │
└─────────────────────────────────────────────────────────┘
```

---

# Part B: LoRA 实现 Code Map

> 阅读指引：按顺序走一遍 LoRA 从注入到训练到保存的完整链路。

---

## 1. LoRA 核心模块

### 1.1 LoRALinear 类定义

[areal/experimental/models/archon/lora/lora_linear.py:21-95]

- `__init_`_ (L35-68): 构造函数，注意 L59-64 用 `object.__setattr__` 存 `_lora_a_weight` / `_lora_b_weight` 为 **plain tensor**（非 `nn.Parameter`），FSDP2 看不到它们
- `forward` (L77-94): 分两条路径 — `_tp_enabled=True` 走 `_tp_lora_forward`，否则走普通 `F.linear(x, A) → F.linear(h, B)`
- `scaling = alpha / rank` (L49): LoRA 的缩放系数

### 1.2 TP 感知的 LoRA Forward

[areal/experimental/models/archon/lora/lora_linear.py:96-138]

- `_tp_lora_forward`: TP 场景下的 LoRA 前向
- L112: **detach input** — LoRA 分支不回传梯度到 base path 的输入，避免 TP 梯度通信冲突
- L115-118: rowwise TP 时对 `_lora_a_weight` 做列切片（Bug #3：A 矩阵创建时是全尺寸，这里只用一段）
- L125-127: colwise TP 时对 LoRA 输出做行切片
- L129-136: 把 local tensor 包成 DTensor 再和 base_out 相加，保持 autograd 连接

### 1.3 from_linear 工厂方法（LoRA 注入点）

[areal/experimental/models/archon/lora/lora_linear.py:201-280]

- L216-217: 手动 `__new_`_ + `nn.Module.__init__`，不走 `__init__`
- L227-231: 搬运原 Linear 的 weight 和 bias
- L233-239: 创建 LoRA A/B plain tensor，`requires_grad_(True)`
- L241-262: 检测原 weight 是否是 DTensor（TP 后），设置 `_tp_enabled`, `_tp_style` (colwise/rowwise/replicate)
- L264-265: 初始化 A(kaiming), B(zeros) — 使 LoRA 初始输出为零
- L267-278: **关键** — 拷贝原 Linear 的 forward hooks（TP 的 pre/post hooks），不拷贝则 TP 通信丢失

### 1.4 梯度同步

[areal/experimental/models/archon/lora/lora_linear.py:294-327]

- `sync_lora_grads`: backward 之后、optimizer_step 之前调用
- L316-317: 遍历所有 `LoRALinear` 模块（Bug #2 已修：不再检查 `_tp_enabled`）
- L324-327: 先 all_reduce on tp_group，再 all_reduce on dp_group
- 为什么需要这个函数：FSDP2 只对 `nn.Parameter` 自动做 DP reduce，plain tensor 必须手动同步

### 1.5 State Dict 辅助（plain tensor 序列化）

[areal/experimental/models/archon/lora/lora_linear.py:169-195]

- `_save_to_state_dict` (L169): 在标准 state_dict 基础上追加 `_lora_a_weight` / `_lora_b_weight`
- `_load_from_state_dict` (L178): 从 state_dict 中 pop 并 copy 到 plain tensor
- 注意 L171: meta device 上的 tensor 跳过保存

### 1.6 Adapter 工具函数

[areal/experimental/models/archon/lora/adapter.py]

- `get_adapter_params(model)`: 遍历所有模块，找 `AdapterModule` 协议的模块，返回 `{name: tensor}` dict
- `set_trainable_params(model, adapter_names)`: 冻结非 adapter 的 `nn.Parameter`。注意 Bug #4：LoRA plain tensor 不在 `named_parameters()` 里，所以匹配永远失败，但效果碰巧正确（base weight 全部被冻结）

---

## 2. LoRA 注入流程（Engine 端）

### 2.1 模型创建（meta device）

[areal/experimental/engine/archon_engine.py:1112]

- `with torch.device("meta"):` 创建模型结构，所有 tensor 在 meta 设备上，不占内存

### 2.2 并行化顺序

[areal/experimental/models/archon/qwen2/infra/parallelize.py:106-159]

- L113-119: **TP** — `apply_tp(model, tp_mesh)`
- L121-124: **CP** — Context Parallelism（可选）
- L126-127: **LoRA** — `apply_lora_fn(model)`，此时 `parallelize_module` 已经把 nn.Linear 的 weight 变成 DTensor，`from_linear` 可以检测 TP
- L130-136: **AC** — Activation Checkpointing
- L139-140: **Compile** — `apply_compile(model)`（我们关掉了）
- L143-153: **FSDP** — `apply_fsdp(model, dp_mesh)`

顺序的约束：LoRA 必须在 TP 之后（才能检测 DTensor），在 FSDP 之前（FSDP 不能看到 plain tensor）

### 2.3 _apply_lora（替换 nn.Linear → LoRALinear）

[areal/experimental/engine/archon_engine.py:990-1032]

- L999-1005: 解析 target_modules，支持 HF 名字（q_proj）和 Archon 名字（wq）的映射
- L1008-1031: 递归遍历 `named_children`，把匹配的 `nn.Linear` 替换为 `LoRALinear.from_linear(...)`
- L1025: 设置 `_debug_name` 用于调试

### 2.4 权重加载 + LoRA 物化

[areal/experimental/engine/archon_engine.py:1042-1081]

- `_freeze_non_lora_params`:
- L1054-1056: `materialize_lora(device)` — 把 meta 设备上的 LoRA tensor 搬到真实 GPU
- L1064-1069: 重新初始化 A(kaiming) / B(zeros)（meta 上的初始化是无意义的）
- L1072: `set_trainable_params` — 冻结所有 base weight

### 2.5 优化器创建

[areal/experimental/engine/archon_engine.py:1083-1195]

- `_get_all_parameters` (L1083-1092): 收集 `m.parameters()` + `module.lora_parameters()`，确保 LoRA plain tensor 被包含
- `_create_optimizer` (L1182-1195): 用 `_get_all_parameters()` 创建 optimizer — **正确包含 LoRA 参数**
- 对比 `_get_model_name_parameters` (L1094-1096): 只返回 `named_parameters()`，**漏掉 LoRA**（Bug #1，影响 xccl 模式）

---

## 3. 训练循环

### 3.1 train_batch

[areal/experimental/engine/archon_engine.py:479-520]

- L487: `optimizer_zero_grad()`
- L489: `_prepare_mb_list` — 把 batch 切成 microbatch
- L508: `forward_backward_batch(mb_list, process_output, forward_only=False)`
- L510-518: **LoRA 梯度同步** — `sync_lora_grads(model, tp_group, dp_group)`
- L520: `optimizer_step()` — 更新 LoRA 权重

### 3.2 forward 内部（PPO loss 计算）

[areal/experimental/engine/archon_engine.py:495-506]

- `process_output` 闭包 → `_compute_logprobs_and_loss` → 计算 PPO/DAPO loss
- loss backward 时，base weight 的梯度走 FSDP2 自动 reduce，LoRA 梯度留在 plain tensor 上等 `sync_lora_grads`

---

## 4. Checkpoint 保存/加载

### 4.1 save_lora_adapter

[areal/experimental/engine/archon_lora_checkpoint.py:34-131]

- L66: `get_adapter_params(engine.model)` — 拿到所有 LoRA tensor
- L78-86: DTensor → `full_tensor()` 收集完整 tensor → strip AC/compile prefix → clone 到 CPU
- L87: `engine.state_dict_adapter.to_hf(archon_state)` — Archon 内部名（wq, w1 等）转 HF 名（q_proj, gate_proj 等）
- L90: 加 `base_model.model.` PEFT 前缀
- L93-95: rank 0 写 `adapter_model.safetensors`
- L98-107: 从参数名反推 target_modules（Bug #6：依赖命名约定）
- L117-124: rank 0 写 `adapter_config.json`
- L129-130: `dist.barrier(group=engine.cpu_group)` — 所有 rank 等待

### 4.2 load_lora_adapter

[areal/experimental/engine/archon_lora_checkpoint.py:133-236]

- L173: 读 safetensors
- L179-185: strip `base_model.model.` 前缀
- L188: `from_hf(hf_state)` — HF → Archon 名转换
- L222-227: 按 key 匹配，`param.data.copy_(value)` 写入 plain tensor

### 4.3 State Dict 转换器

[areal/experimental/models/archon/qwen2/model/state_dict_adapter.py:15-82]

- L24-60: `from_hf_map` — 完整的双向映射，包括所有 7 个 target_modules 的 lora_A / lora_B
- L72-81: `to_peft_module_map` — Archon 模块名 → HF PEFT 模块名映射

---

## 5. Weight Update（训练 → 推理同步）

### 5.1 触发入口

[areal/experimental/engine/archon_engine.py:670-690]

- `update_weights`: 根据 `meta.type` 分发到 disk / xccl 路径

### 5.2 update_weights_from_disk

[areal/experimental/engine/archon_weight_sync.py:216-261]

- L224-225: rank 0 异步发 HTTP 给 SGLang
- L228-235: 所有 rank 调 `save_lora_adapter`（内含 barrier）
- L239-257: rank 0 写 ready file + 等 SGLang 加载完成 (`fut.result()`)
- L259-260: `current_platform.synchronize()` + `dist.barrier()` — 全局同步

### 5.3 SGLang 端 HTTP 请求构造

[areal/engine/sglang_remote.py:126-156]

- `build_disk_weight_update_requests`: LoRA 时发 POST `/load_lora_adapter`，payload 含 `lora_name="lora-v{version}"` 和 `lora_path`

---

## 6. 已知 Bug 速查


| #   | 位置                               | 问题                                                | 状态          |
| --- | -------------------------------- | ------------------------------------------------- | ----------- |
| 1   | archon_engine.py:1094-1096       | `_get_model_name_parameters` 漏掉 LoRA plain tensor | 不影响 disk 模式 |
| 2   | lora_linear.py:317               | `sync_lora_grads` 原先跳过 tp=1                       | **已修复**     |
| 3   | lora_linear.py:115-118, 234      | rowwise TP 时 A 矩阵全尺寸创建、forward 只用一段               | 低优          |
| 4   | adapter.py:60-71                 | `set_trainable_params` 语义错但效果碰巧正确                 | 低优          |
| 5   | archon_engine.py:1020            | LoRA dropout 未暴露配置                                | 低优          |
| 6   | archon_lora_checkpoint.py:99-107 | target_modules 检测依赖命名约定                           | 脆弱但正确       |

---

## 附录 B: TP 下 LoRA 的 detach 近似

> 对应代码 `_tp_lora_forward` [lora_linear.py:L112]

### 现象

TP 模式下，LoRA 分支对输入 `x` 做了 `detach()`，切断了 LoRA 对 `x` 的梯度贡献：

```python
def _tp_lora_forward(self, x, base_out):
    h = x.detach()                               # ← 切断
    h = F.linear(h, lora_a_w)
    lora_out = F.linear(h, self._lora_b_weight)
```

### 数学分析

完整的前向：`y = Wx + BAx = (W + BA)x`

完整的 x 梯度应为：`dy/dx = Wᵀ + AᵀBᵀ`

detach 后 x 只收到 `Wᵀ`，丢掉了 `AᵀBᵀ` 这一项。

### 为什么是合理的近似

1. `BA` 是低秩矩阵（rank=32 vs hidden=2048），相对于 `W` 贡献很小
2. `scaling = α/r` 进一步缩小 LoRA 的幅度
3. x 的梯度用于反向传播到更早的层，`Wᵀ` 已提供主要信号

### 为什么要 detach

TP 下 `x` 的梯度通信由 base path 的 DTensor hooks 管理。如果 LoRA 分支也给 `x` 回传梯度，这部分梯度是 local tensor，不会经过 DTensor 的 TP all-reduce，导致梯度在不同 rank 之间不一致。detach 是用一个低秩小量的近似换取 TP 通信的正确性。

### 注意

非 TP 模式（`_tp_enabled=False`）走普通 `forward`，没有 detach，梯度是精确的。此近似仅在 TP > 1 时存在。

---

# Appendix C: PEFT 方法迁移笔记

## C.1 rsLoRA

**改动范围：** 仅 scaling 计算

- 新增 `_compute_lora_scaling(alpha, rank, peft_type)` 辅助函数
- `peft_type="rslora"` 时 scaling = `α/√r`（标准 LoRA 是 `α/r`）
- 通过 `peft_type` 字段从 yaml → `cli_args.py` → `LoRAConfig` → `LoRALinear.from_linear` 一路传递
- **不改 forward/backward/optimizer**

## C.2 MiLoRA++ 迁移方案

### 背景

MiLoRA++ 是 PeRL 提出的改进初始化方法。核心思想：用 SVD 找到原始权重的**最小奇异值子空间方向**来初始化 LoRA A 矩阵，让 LoRA 从训练一开始就"瞄准"模型欠表达的子空间。

与 MiLoRA 的区别：
- MiLoRA：A/B 都带奇异值幅度，且需要 `W -= BA` 修改 base weight
- MiLoRA++：A 只取方向（正交），B=0，**不修改 base weight**（更稳定，更易实现）

### 算法

```
输入：原始权重 W [out_dim, in_dim]，rank r
1. SVD: U, S, Vh = svd(W)          # Vh: [min(out,in), in_dim]
2. A = Vh[-r:, :]                   # 取最小 r 个奇异值对应的右奇异向量（正交）
3. B = zeros(out_dim, r)            # 零初始化
4. 不修改 W
```

初始化后 LoRA 输出 = scaling × B @ A @ x = 0（因为 B=0），保证训练起点稳定。
A 的行向量天然正交（SVD 性质），提供干净的梯度信号方向。

### 迁移到 AReaL 的修改点

**注入方式：** 复用 rsLoRA 的 `peft_type` 机制，`peft_type="milora_plus"`

#### 1. `lora_linear.py` — 核心改动

**位置：** `from_linear()` 方法，L242-274（初始化 A/B 的部分）

当前代码（L242-274）：
```python
# 分配 A/B
local_w = getattr(linear.weight, "_local_tensor", linear.weight)
_a = torch.empty(rank, linear.in_features, device=..., dtype=...)
_b = torch.empty(linear.out_features, rank, device=..., dtype=...)
# ... TP 处理 ...
nn.init.kaiming_uniform_(lora_linear._lora_a_weight, a=math.sqrt(5))
nn.init.zeros_(lora_linear._lora_b_weight)
```

修改方案：在 L273-274（kaiming init 之后）加入 peft_type 判断：

```python
if peft_type == "milora_plus":
    _init_milora_plus(lora_linear)
else:
    nn.init.kaiming_uniform_(lora_linear._lora_a_weight, a=math.sqrt(5))
    nn.init.zeros_(lora_linear._lora_b_weight)
```

新增 `_init_milora_plus(lora_linear)` 函数：

```python
def _init_milora_plus(lora_linear: "LoRALinear"):
    """MiLoRA++ init: A = min singular value directions (orthonormal), B = 0."""
    # 1. 获取 local weight（可能是 DTensor shard）
    w = getattr(lora_linear.weight, "_local_tensor", lora_linear.weight)

    # 2. SVD（在 local weight 上做，float32 精度）
    _, _, Vh = torch.linalg.svd(w.float(), full_matrices=False)

    # 3. 取最小奇异值方向
    rank = lora_linear.rank
    A_init = Vh[-rank:, :].contiguous().to(w.dtype)

    # 4. 赋值
    lora_linear._lora_a_weight.data.copy_(A_init)
    nn.init.zeros_(lora_linear._lora_b_weight)
```

**TP 兼容性分析：**

关键问题：TP 下 weight 是 DTensor（按 dim=0 或 dim=1 shard），SVD 在 local shard 上做是否正确？

- **colwise TP**（Shard dim=0）：W local shape = [out_dim/tp, in_dim]，A shape = [rank, in_dim]
  - SVD(W_local) 得到的 Vh 是 [min(out/tp, in), in_dim]
  - A = Vh[-r:, :] shape = [rank, in_dim] ✅ 形状匹配
  - 每个 TP rank 用自己 shard 的 SVD 方向，语义上各自瞄准自己负责的输出子空间

- **rowwise TP**（Shard dim=1）：W local shape = [out_dim, in_dim/tp]，A shape = [rank, in_dim/tp]
  - SVD(W_local) 得到的 Vh 是 [min(out, in/tp), in_dim/tp]
  - A = Vh[-r:, :] shape = [rank, in_dim/tp] ✅ 形状匹配

两种 TP 情况下，对 local shard 做 SVD 取方向都是合理的，因为 A/B 本身也是 plain tensor，按 local 维度存储。

#### 2. `archon_engine.py` — 无需额外修改

`peft_type` 已经通过 rsLoRA 的改动从 config 传到 `from_linear`，`milora_plus` 自动走同一路径。

#### 3. `cli_args.py` — 无需修改

`peft_type` 是 str 字段，yaml 里写 `peft_type: milora_plus` 即可。

#### 4. 新增启动脚本和 yaml

- 新建 `lora_sync_4node_1.5b_dapo_milora_plus.yaml`（或复制现有 yaml，改 `peft_type: milora_plus`）
- 新建 `run_milora_plus_sync_4node_1.5b.sh`

### 注意事项

1. **SVD 开销**：每个 LoRALinear 初始化时做一次 SVD，只在启动时执行，不影响训练速度
2. **float32 精度**：SVD 必须在 float32 下做（bf16 精度不够），结果再转回 bf16
3. **scaling**：MiLoRA++ 使用标准 LoRA scaling（α/r），不用 rsLoRA 的 α/√r
4. **B=0 保证**：初始化时 LoRA 输出为零，训练开始时模型行为与原始模型一致
