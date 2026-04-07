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

## C.3 PiSSA & MiLoRA 迁移方案

### 背景

PiSSA 和 MiLoRA 都是 SVD-based 初始化方法，与 MiLoRA++ 不同的是它们**带上奇异值幅度**初始化 A/B，且**需要修改 base weight**。

| 方法 | SVD 方向 | A/B 初始化 | Base Weight |
|------|----------|-----------|-------------|
| PiSSA | 最大奇异值（主成分） | A = √(S/scaling) · Vh, B = U · √(S/scaling) | W -= scaling · BA |
| MiLoRA | 最小奇异值（次要成分） | 同上，但取最小 r 个 | W -= scaling · BA |
| MiLoRA++ | 最小奇异值 | A = Vh（方向only），B = 0 | 不修改 |

### 算法

```
输入：W [out_dim, in_dim]，rank r，scaling = α/r
1. SVD: U, S, Vh = svd(W)
2. 选取 r 个分量：
   - PiSSA: 前 r 个（最大奇异值）
   - MiLoRA: 后 r 个（最小奇异值）
3. S_scaled = S_sel / scaling
4. A = diag(√S_scaled) @ Vh_sel     # [r, in_dim]
5. B = U_sel @ diag(√S_scaled)       # [out_dim, r]
6. W -= scaling * B @ A              # 保证 W_new + scaling*BA = W_orig
```

### 实现

**统一函数 `_init_svd_lora(lora_linear, mode)`**：PiSSA 用 `mode="max"`，MiLoRA 用 `mode="min"`。

**TP 限制**：这两个方法修改 base weight（W -= delta），在 DTensor/TP 下 weight 是 shard 的，直接修改语义不正确。因此加了 TP guard：

```python
if peft_type in _BASE_WEIGHT_MODIFY_TYPES and lora_linear._tp_enabled:
    raise ValueError(f"peft_type='{peft_type}' modifies base weights and is not compatible with TP.")
```

**使用**：yaml 里 `peft_type: pissa` 或 `peft_type: milora`，必须 TP=1。

### Async Rollout 兼容性问题与修复

**问题**：PiSSA/MiLoRA 在 `_freeze_non_lora_params` 中修改了 actor 端的 base weight（`W -= scaling * BA`），但 SGLang rollout 加载的是**原始 HF 模型**。SGLang 的 forward 变成 `W_orig @ x + scaling * BA @ x`，而不是正确的 `W_modified @ x + scaling * BA @ x`，导致 SVD 分量被 double 加上，输出完全是乱码。

此问题仅影响 `_BASE_WEIGHT_MODIFY_TYPES`（`pissa`, `milora`）。MiLoRA++（B=0，不改 base weight）和标准 LoRA 不受影响。

**修复方案**：在 PiSSA/MiLoRA 修改 base weight 后，保存修改后的 base model 到磁盘，让 SGLang 加载这个修改后的模型。

**修改点：**

#### 1. `archon_engine.py` — `_freeze_non_lora_params` 保存修改后的 base model

```python
# 在 _freeze_non_lora_params 末尾，如果 peft_type 是 pissa/milora:
if self.lora_config.peft_type in ("pissa", "milora"):
    path = self._save_pissa_base_model()
    self._pissa_base_model_path = path
```

`_save_pissa_base_model()` 用已有的 `save_model_to_hf()` 保存完整模型（只 rank 0 写磁盘）。保存路径：`{fileroot}/checkpoints/{exp}/{trial}/actor/pissa_base_model/`。

#### 2. `rl_trainer.py` — `_save_initial_lora_weights` 返回 base model path

```python
# 如果 actor engine 有 _pissa_base_model_path，传给 _init_rollout
# _init_rollout 用它替换 sglang.model_path
```

#### 3. SGLang 端 — 无需改动

SGLang 的 `model_path` 指向修改后的模型即可。LoRA adapter update 流程不变（只更新 A/B weights，base model 不变）。

**注意**：保存完整 base model 会有一次性的 I/O 开销（1.5B ≈ 3GB，7B ≈ 14GB），但只在初始化时发生一次。

## C.4 LoRA-FA 迁移方案

### 背景

LoRA-FA (LoRA with Frozen-A) 出自论文 *LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning*。核心思想：**冻结 A 矩阵（随机初始化后不再更新），只训练 B 矩阵**。这样可以节省约一半的 LoRA 梯度内存和优化器状态，同时性能接近标准 LoRA。

### 算法

1. A 用 kaiming 初始化（与标准 LoRA 相同），然后 **freeze**（`requires_grad = False`）
2. B 初始化为 0（与标准 LoRA 相同）
3. Forward 不变：`output = x @ W^T + scaling * x @ A^T @ B^T`
4. Backward 只更新 B，A 不产生梯度

### PeRL 实现

PeRL 使用 HuggingFace PEFT 的 `create_lorafa_optimizer` [adapter.py:98]，它内部把 A 矩阵从优化器 param groups 中移除。

### 迁移到 AReaL 的修改点

AReaL 的 LoRA 权重是 plain tensor（不是 nn.Parameter），通过 `requires_grad_(True)` 来接收梯度。LoRA-FA 只需在初始化后把 A 的 `requires_grad` 设为 False。

#### 1. `lora_linear.py` — 核心改动

在 `from_linear()` 的 peft_type dispatch 中加入 `lorafa` 分支：

```python
if peft_type == "lorafa":
    nn.init.kaiming_uniform_(lora_linear._lora_a_weight, a=math.sqrt(5))
    nn.init.zeros_(lora_linear._lora_b_weight)
    lora_linear._lora_a_weight.requires_grad_(False)  # freeze A
```

**影响分析：**

- **`lora_parameters()`**：返回 `[_lora_a_weight, _lora_b_weight]`，不需要改。优化器会收到两个 tensor，但 A 的 `requires_grad=False`，优化器会自动跳过（Adam 对 grad=None 的参数不更新）。
- **`sync_lora_grads()`**：遍历 A 和 B 的 grad，A.grad 为 None 时跳过（已有 `if tensor.grad is not None` 守卫 [lora_linear.py:398]），无需修改。
- **`forward()`**：A 参与前向计算但不接收梯度，PyTorch autograd 自动处理，无需修改。
- **`_save_to_state_dict` / `_load_from_state_dict`**：A 仍然需要保存和加载（checkpoint 恢复时需要相同的 A），无需修改。
- **SGLang 推理端**：不受影响，adapter 格式不变。

#### 2. `archon_engine.py` — 无需额外修改

`peft_type` 已通过 `LoRAConfig` 传入 `from_linear()`，无需改动。

#### 3. 新增启动脚本和 yaml

- `scripts/lorafa_async_4node_1.5b_dapo.yaml`：基于 lora yaml，`peft_type: lorafa`
- `scripts/run_lorafa_async_4node_1.5b.sh`：对应启动脚本

### 注意事项

1. LoRA-FA 与 rsLoRA 可以组合使用（rsLoRA 改 scaling，LoRA-FA 改训练参数），但目前先独立实现
2. A frozen 后优化器状态（momentum/variance）不会为 A 分配，自动节省内存

## C.5 LoRA+ 迁移方案

### 背景

LoRA+ 出自论文 *LoRA+: Efficient Low Rank Adaptation of Large Models*。核心发现：B 矩阵应该用比 A 矩阵更大的学习率，论文推荐 `lr_B = lr_A × ratio`（默认 ratio=2.0）。

### 算法

1. 初始化与标准 LoRA 相同（kaiming A + zeros B）
2. Forward 不变
3. **Optimizer 使用两个 param groups**：A 用 base_lr，B 用 base_lr × `loraplus_lr_ratio`

### PeRL 实现

PeRL 调用 HuggingFace PEFT 的 `create_loraplus_optimizer` [adapter.py:136]，传入 `loraplus_lr_ratio=2.0`。

### 迁移到 AReaL 的修改点

AReaL 当前 optimizer 链路：`_get_all_parameters()` 返回 flat list → `create_optimizer(params, config)` 创建 AdamW。LoRA+ 需要将 A 和 B 拆成不同 param groups。

#### 1. `cli_args.py` — 新增配置字段

在 PPOActorConfig 加 `loraplus_lr_ratio: float = 1.0`。ratio=1.0 等价于标准 LoRA。

#### 2. `archon_engine.py` — LoRAConfig 传入 ratio + _create_optimizer 分组

LoRAConfig 加 `loraplus_lr_ratio` 字段。`_create_optimizer` 当 `peft_type == "lora_plus"` 时，构建两个 param groups：
- Group 1: base params + lora_a → lr = base_lr
- Group 2: lora_b → lr = base_lr × ratio

通过遍历 LoRALinear modules 按 `_lora_a_weight` / `_lora_b_weight` 属性区分。

#### 3. `archon_utils.py` — create_optimizer 支持 param groups

修改签名 `params: list[nn.Parameter] | list[dict]`，当传入 param groups（list of dict）时直接传给 AdamW。

#### 4. 新增启动脚本和 yaml

- `scripts/lora_plus_async_4node_1.5b_dapo.yaml`：`peft_type: lora_plus`, `loraplus_lr_ratio: 2.0`
- `scripts/run_lora_plus_async_4node_1.5b.sh`

### 不需要改的

- `forward()` — 不变
- `sync_lora_grads()` — 不变（按 tensor.grad is not None 过滤）
- checkpoint — 不变
- `fsdp2_clip_grad_norm` — 用 `_get_all_parameters()` 的 flat list，和 optimizer groups 无关
- `create_lr_scheduler` — 两个 group 共享同一 scheduler（constant），不影响

## C.6 DoRA 迁移方案

### 背景

DoRA (Weight-Decomposed Low-Rank Adaptation) 出自论文 *DoRA: Weight-Decomposed Low-Rank Adaptation of Large Language Models* (Liu et al., 2024, NVIDIA)。核心思想：**将权重矩阵分解为 magnitude（幅度）和 direction（方向），对方向用 LoRA 更新，对幅度用独立可训练向量更新**。

研究发现 full fine-tuning 中幅度和方向的更新是不对称的，而 LoRA 倾向于将两者耦合在一起。DoRA 通过解耦让微调更接近 full FT 效果。

### 算法

**权重分解：**

预训练权重 $W_0 \in \mathbb{R}^{out \times in}$ 分解为：

$$W_0 = m \cdot \frac{V}{\|V\|_c}$$

其中 $m \in \mathbb{R}^{out}$ 是每行的 L2 范数（magnitude 向量），$V = W_0$（direction 矩阵），$\|\cdot\|_c$ 是按行计算的 L2 norm。

**初始化：**

$$m_0 = \|W_0\|_{\text{dim}=1} = [\|W_0[0,:]\|_2, \|W_0[1,:]\|_2, ..., \|W_0[\text{out}-1,:]\|_2]$$

$m_0$ 是一个 shape 为 `[out_dim]` 的可训练向量。A、B 和标准 LoRA 相同（kaiming A, zeros B）。

**Forward：**

$$\text{weight\_norm} = \|W + \text{scaling} \cdot BA\|_{\text{dim}=1} \quad \text{(detach, 不参与反向传播)}$$

$$\text{mag\_norm\_scale} = \frac{m'}{\text{weight\_norm}} \quad \text{(shape: [1, out\_dim])}$$

$$\text{output} = (\text{mag\_norm\_scale} - 1) \cdot (x @ W^T) + \text{mag\_norm\_scale} \cdot \text{scaling} \cdot (x @ A^T @ B^T)$$

关键技巧（论文 Section 4.3）：**weight_norm 要 detach**，不接收梯度。这保证梯度只通过 $m'$ 和 LoRA AB 流动。

等价展开：$\text{output} = m' \cdot \frac{W + \text{scaling} \cdot BA}{\|W + \text{scaling} \cdot BA\|_c} \cdot x$

### PeRL 实现

PeRL 直接调用 HuggingFace PEFT 的 `LoraConfig(use_dora=True)` [adapter.py:21-32]，不需要手动实现。

PEFT 的实现在 `peft/tuners/lora/dora.py:DoraLinearLayer`：
- `update_layer()`：初始化 magnitude 向量 `self.weight = nn.Parameter(||W + scaling * BA||_dim1)`
- `forward()`：计算 `mag_norm_scale = m / ||W + scaling * BA||_dim1.detach()`，输出 `(mag_norm_scale - 1) * base + mag_norm_scale * lora * scaling`

### 迁移到 AReaL 的修改点

DoRA 是之前所有 PEFT 方法中最复杂的——它不仅改初始化，还要**改 forward**。需要引入一个新的可训练参数 `_dora_magnitude`。

#### 1. `lora_linear.py` — 核心改动（初始化 + forward）

**1a. 初始化：在 `from_linear()` 加 `dora` 分支**

```python
elif peft_type == "dora":
    nn.init.kaiming_uniform_(lora_linear._lora_a_weight, a=math.sqrt(5))
    nn.init.zeros_(lora_linear._lora_b_weight)
    # magnitude 向量 = 每行 L2 norm of W（初始时 BA=0，所以 ||W + 0|| = ||W||）
    local_w = getattr(lora_linear.weight, "_local_tensor", lora_linear.weight)
    mag = torch.linalg.norm(local_w.detach().float(), dim=1).to(local_w.dtype)
    mag.requires_grad_(True)
    object.__setattr__(lora_linear, "_dora_magnitude", mag)
```

和 A/B 一样用 `object.__setattr__` 存为 plain tensor，对 FSDP2 不可见。

**1b. Forward：新增 DoRA 分支**

修改 `forward()` 方法，在标准 LoRA forward 基础上加 DoRA 逻辑：

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.disabled:
        return F.linear(x, self.weight, self.bias)

    if self._tp_enabled:
        # DoRA + TP: 见下方 TP 兼容性讨论
        return self._tp_lora_forward(x, F.linear(x, self.weight, self.bias))

    if hasattr(self, "_dora_magnitude"):
        return self._dora_forward(x)

    # 标准 LoRA forward（不变）
    base_out = F.linear(x, self.weight, self.bias)
    h = F.dropout(x, p=self._dropout_p, training=self.training)
    h = F.linear(h, self._lora_a_weight)
    lora_out = F.linear(h, self._lora_b_weight)
    return base_out + self.scaling * lora_out

def _dora_forward(self, x: torch.Tensor) -> torch.Tensor:
    """DoRA forward: magnitude × normalized(direction + lora_delta)."""
    # base output (without bias — bias added at the end)
    base_out = F.linear(x, self.weight)

    # lora output
    h = F.dropout(x, p=self._dropout_p, training=self.training)
    h = F.linear(h, self._lora_a_weight)
    lora_out = F.linear(h, self._lora_b_weight)

    # compute weight norm: ||W + scaling * B @ A||_dim1, detached
    lora_weight = self._lora_b_weight @ self._lora_a_weight
    weight_norm = torch.linalg.norm(
        self.weight.detach() + self.scaling * lora_weight.detach(),
        dim=1
    ).to(x.dtype)  # [out_dim]

    # mag_norm_scale = m / ||W + scaling*BA||, shape [1, out_dim]
    mag_norm_scale = (self._dora_magnitude / weight_norm).view(1, -1)

    # DoRA output = (scale - 1) * base + scale * scaling * lora
    result = (mag_norm_scale - 1) * base_out + mag_norm_scale * self.scaling * lora_out

    # add bias
    if self.bias is not None:
        result = result + self.bias

    return result
```

**关键设计要点：**
- `weight_norm` 要 `.detach()`：不让梯度流过 norm 分支（论文 Section 4.3）
- `base_out` 不含 bias：因为 DoRA 对 `W*x` 做缩放，bias 不参与方向分解，单独加
- `lora_weight = B @ A` 每次 forward 都要算：因为 BA 在训练中不断变化

**1c. `lora_parameters()` 返回 magnitude**

```python
def lora_parameters(self) -> list[torch.Tensor]:
    params = [self._lora_a_weight, self._lora_b_weight]
    if hasattr(self, "_dora_magnitude"):
        params.append(self._dora_magnitude)
    return params
```

**1d. `_save_to_state_dict` / `_load_from_state_dict` 加 magnitude**

```python
# _save_to_state_dict 中追加：
if hasattr(self, "_dora_magnitude"):
    m = self._dora_magnitude if keep_vars else self._dora_magnitude.detach()
    destination[prefix + "_dora_magnitude"] = m

# _load_from_state_dict 中追加：
m_key = prefix + "_dora_magnitude"
if m_key in state_dict:
    self._dora_magnitude.data.copy_(state_dict.pop(m_key))
```

**1e. `adapter_params()` 追加 magnitude**

```python
def adapter_params(self) -> list[str]:
    names = ["_lora_a_weight", "_lora_b_weight"]
    if hasattr(self, "_dora_magnitude"):
        names.append("_dora_magnitude")
    return names
```

**1f. `sync_lora_grads()` — 无需修改**

`sync_lora_grads` 目前只遍历 `_lora_a_weight` 和 `_lora_b_weight`。需要让它**也同步 `_dora_magnitude` 的梯度**：

```python
for module in model.modules():
    if isinstance(module, LoRALinear):
        tensors_to_sync = [
            ("a", module._lora_a_weight),
            ("b", module._lora_b_weight),
        ]
        if hasattr(module, "_dora_magnitude"):
            tensors_to_sync.append(("m", module._dora_magnitude))
        for _pname, tensor in tensors_to_sync:
            if tensor.grad is not None:
                ...  # all_reduce
```

#### 2. `archon_engine.py` — 无需额外修改

`peft_type` 已通过 `LoRAConfig` 传入 `from_linear()`。`_get_all_parameters()` 调用 `lora_parameters()` 获取可训练参数，magnitude 自动包含在内。

#### 3. DoRA + TP 兼容性

**问题：** DoRA forward 需要访问完整的 `self.weight` 来算 `||W + scaling * BA||_dim1`。但 TP 后 weight 是 DTensor（只有本地 shard）。

**分析：**
- colwise TP（q_proj, k_proj, v_proj, gate_proj, up_proj）：weight 按 dim=0（行）切分，每个 rank 有一部分行。`dim=1 norm` 在每行内独立，**每个 rank 可以独立计算自己的行的 norm**。magnitude 也只存对应行。✅ 兼容。
- rowwise TP（o_proj, down_proj）：weight 按 dim=1（列）切分。`dim=1 norm` 需要跨列求和，**需要 all-reduce**。❌ 需要额外通信。

**方案：先限制 DoRA 只在 TP=1 时使用**（和 PiSSA、MiLoRA 类似），在 `from_linear()` 加守卫：

```python
if peft_type == "dora" and lora_linear._tp_enabled:
    raise ValueError(
        "peft_type='dora' is not compatible with TP. "
        "Use TP=1 or choose a different method."
    )
```

如果后续需要 TP 支持，可以针对 colwise 单独实现，rowwise 加 all-reduce。

#### 4. DoRA + LoRA-FA / LoRA+ 兼容性

- **DoRA + LoRA-FA**：理论上可以（冻结 A，只训 B 和 magnitude），但初始版本不组合。
- **DoRA + LoRA+**：理论上可以（B 用更大 lr，magnitude 用 base_lr），初始版本不组合。
- **DoRA + rsLoRA**：可以组合（rsLoRA 只改 scaling 公式），但初始版本不组合。

#### 5. 性能影响

DoRA forward 多了一个 `B @ A`（shape `[out, in]`）和一个 norm 计算。对于典型的 hidden=1536, rank=32：
- `B @ A`: `[1536, 32] @ [32, 1536]` = `[1536, 1536]` — 约 7.1M FLOPs
- `norm(dim=1)`: `[1536, 1536]` 按行求和 — 忽略不计

相比 base `F.linear` 的计算量（`[batch×seq, 1536] @ [1536, 1536]`），DoRA 额外开销很小（不依赖 batch size）。

#### 6. 新增启动脚本和 yaml

- `scripts/dora_async_4node_1.5b_dapo.yaml`：基于 lora yaml，`peft_type: dora`
- `scripts/run_dora_async_4node_1.5b.sh`：对应启动脚本

### 注意事项

1. **`B @ A` 每次 forward 都要算**：不能缓存，因为训练中 A、B 一直在变。推理可以缓存（但 SGLang 端用的是自己的 adapter 实现，不受影响）
2. **weight_norm 必须 detach**：这是 DoRA 论文的关键设计。如果不 detach，梯度会通过 norm 分支影响 W（base weight 应该是 frozen 的），而且 W 是 DTensor 时会引发 FSDP2 问题
3. **magnitude 向量不大**：每个 LoRALinear 只多 `out_dim` 个标量（如 1536），总共 7 层 × 1536 ≈ 10K 参数，negligible

## C.7 MiLoRA++ 研究方向思考

### 现状

MiLoRA++（PeRL 提出）用 SVD 最小奇异值方向初始化 A，B=0，假设 RL 更新方向与预训练权重的"被遗忘子空间"相关。但实验中 MiLoRA++ 并没有比 vanilla LoRA 表现出显著优势。

### 为什么 MiLoRA++ 可能效果有限

1. **B=0 意味着初始方向仅在训练开始后间接起作用**。A 的方向选择影响的是梯度流的"起点"，但 optimizer 几步之后会自己找到方向
2. **最小奇异值方向 ≠ RL 需要的方向**。这只是预训练权重的统计性质，和 reward signal 无直接关联
3. **RL 更新方向可能本身并不特殊** — 如果 RL 和 SFT 的更新方向在谱空间上没有显著差异，那任何基于方向的 init 都不会有大突破

### 可探索的研究方向

#### 方向 1: RL 更新方向实证分析（最优先）

**核心问题**：RL 的 ΔW 到底落在原始 W 的哪些奇异值方向上？

**实验方案**：
1. 跑 vanilla LoRA RL 训练到收敛，每 N 步保存 lora_A 和 lora_B
2. 计算 ΔW = scaling × B @ A（各时间步）
3. 将 ΔW 投影到原始 W 的 SVD 基上：`coefficients = U^T @ ΔW @ Vh^T`
4. 画出 coefficients 在不同奇异值方向上的分布
5. 对比：top-r 方向 vs bottom-r 方向 vs random-r 方向，哪个解释了更多 ΔW

**结果解读**：
- 集中在 top 方向 → PiSSA 更合理
- 集中在 bottom 方向 → MiLoRA++ 思路对但实现可能有问题
- 比较均匀 → init 方向不是关键因素，应转向训练动态
- 有独特 pattern → 新方法的起点

#### 方向 2: Gradient-informed initialization（梯度引导初始化）

不用预训练权重的 SVD，而是用 RL 梯度的信息：
- 跑几步 RL，收集 ∇W（梯度矩阵）
- 对梯度矩阵做 SVD，找到 RL 梯度的主方向
- 用这些方向初始化 A

比 MiLoRA++ 更直接 — 不是猜测 RL 会往哪走，而是直接观测。代价是需要一个 warm-up phase。

#### 方向 3: 逐层差异化策略（Layer-wise adaptive）

不同层在 RL 中的角色不同：
- 底层（embedding 附近）：语义理解，可能不需要太多改动
- 顶层（output 附近）：决策层，RL reward 驱动的更新集中在这里

可以做：不同层用不同 rank、不同 init、不同 scaling。先跑实验看各层 ΔW 范数分布。

#### 方向 4: RL-specific 训练动态

如果 init 方向不是关键，那可能训练动态才是：
- RL 前期探索阶段用大 scaling/lr，后期收敛阶段缩小
- 用 reward signal 的方差自适应调整 LoRA scaling
- Reward-aware gradient clipping

#### 方向 5: 组合方法 ablation

MiLoRA++ 方向 + rsLoRA scaling + LoRA+ 差异化 lr，是否有协同效应？低成本 ablation。

### 建议执行顺序

1. **方向 1**（实证分析）— 不需要改代码，只需 post-hoc 分析脚本。结果决定后续方向
2. **方向 5**（组合方法）— 已有实现，只需配 yaml 跑实验
3. **方向 3**（逐层分析）— 可以复用方向 1 的分析框架
4. **方向 2/4** — 需要更多代码开发，视前面结果决定
