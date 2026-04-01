# AReaL LoRA Weight Update 调用流程

## 整体架构

```
┌──────────────────────────────────────────────────────────────┐
│  Main Process (head node)                                     │
│                                                               │
│  RLTrainer (rl_trainer.py)                                    │
│    ├── TrainController (self.actor)  ──Ray RPC──→  Workers    │
│    │     └── asyncio.gather: 并行调所有 worker                 │
│    ├── RolloutController (self.rollout) ──Ray RPC──→  SGLang  │
│    └── train loop: pause → update_weights → save → resume     │
└──────────────────────────────────────────────────────────────┘
```

训练主循环在 `rl_trainer.py`，但它本身不直接跑 engine，而是通过两个 Controller 用 Ray RPC 远程调用。

## 调用链（disk 模式 + LoRA）

### 第一层：RLTrainer（主进程）

文件：`areal/trainer/rl_trainer.py`

```python
rollout.pause()                              # 暂停 rollout 入队
actor.update_weights(versioned_meta)         # ← 通过 TrainController 分发
actor.set_version(new_version)
rollout.set_version(new_version)
_save_hf(...)                                # saver 保存 checkpoint
_save_recover_checkpoint(...)
_evaluate(...)
rollout.resume()                             # 恢复 rollout
```

### 第二层：TrainController → Ray Workers

文件：`areal/infra/controller/train_controller.py`

```python
def update_weights(self, meta):
    self._custom_function_call("update_weights", meta=meta)

def _custom_function_call(self, method, *args, **kwargs):
    dp_args, dp_kwargs, group_indices = self._prepare_dispatch(*args, **kwargs)
    results = run_async_task(self._call_workers, method, dp_args, dp_kwargs)
    return self._collect_results(results, group_indices)

async def _call_workers(self, method, dp_split_args, dp_split_kwargs):
    tasks = []
    for idx, worker in enumerate(self.workers):
        tasks.append(
            self.scheduler.async_call_engine(worker.id, method, ...)
        )
    return await asyncio.gather(*tasks)   # ← 等所有 worker 返回
```

**关键**：`asyncio.gather` 并行调所有 worker，任何一个卡住则整体 hang。

### 第三层：Engine（每个 worker 进程内）

根据 backend 不同走不同路径：

| backend     | Engine 类       | weight update 代码位置            |
|-------------|-----------------|-----------------------------------|
| `archon:dN` | `ArchonEngine`  | `archon_weight_sync.py`           |
| `fsdp:dN`   | `FSDPEngine`    | `fsdp_engine.py`                  |

当前配置使用 `archon:d8`，所以走 ArchonEngine：

文件：`areal/experimental/engine/archon_engine.py`

```python
def update_weights(self, meta):
    self._check_rollout_engine_connected()
    if meta.type == "xccl":
        update_weights_from_distributed(state, meta, engine)
    elif meta.type == "disk":
        update_weights_from_disk(meta, engine)    # ← 走这里
```

### 第四层：update_weights_from_disk（Archon 版）

文件：`areal/experimental/engine/archon_weight_sync.py`

```python
def update_weights_from_disk(meta, engine):
    # ---- Step 1: Rank 0 发 HTTP 请求给 SGLang (async，不等结果) ----
    if rank == 0:
        fut = engine.rollout_engine.update_weights_from_disk(meta)
        # → 内部构造 HTTP POST /load_lora_adapter 请求

    # ---- Step 2: 所有 rank 保存 LoRA adapter ----
    if engine.lora_config is not None:
        save_lora_adapter(engine, meta.path, base_model_path)
        # → archon_lora_checkpoint.py
        # → 只有 rank 0 实际写 safetensors 文件
        # → 内部有 dist.barrier() 让所有 rank 同步

    # ---- Step 3: Rank 0 写 ready 文件 + 等 SGLang 加载完 ----
    if rank == 0:
        write(.areal_weight_update_ready)     # 原子写
        name_resolve.add(...)                  # 注册 KV
        fut.result()                           # 阻塞等 SGLang 完成

    # ---- Step 4: 全局同步 ----
    current_platform.synchronize()             # GPU sync
    dist.barrier(group=engine.cpu_group)       # 跨 rank CPU barrier
```

### 第五层：SGLang 端加载

```
RemoteInfEngine.update_weights_from_disk(meta)
  └── sglang_remote.py: build_disk_weight_update_requests(meta)
        └── HTTP POST /load_lora_adapter
              payload: { lora_name: "lora-v1", lora_path: "/.../weight_update_v1" }
  └── 每个 SGLang server 进程从 NFS 读 safetensors 并加载到 GPU
```

## 单机 vs 多机的区别

核心流程一模一样，区别只在 Ray 调度把 worker 分到不同节点。

### 单机（1-2 node, 16 GPU）

```
Node 0:  [Main Process] + [Actor rank 0-7] + [SGLang server 0-7]

dist.barrier()       → 同机 IPC/shared memory，毫秒级
save_lora_adapter()  → 写本地 NFS
SGLang 加载          → 读本地 NFS
```

### 多机（4 node, 32 GPU: 24 rollout + 8 train）

```
Node 0 (head):       [Main Process] + [SGLang server 0-7]
Node 1:              [Actor rank 0-7] + [SGLang server 8-15]
Node 2:              [SGLang server 16-23]
Node 3:              (可能有 ref 或空闲)
```

| 维度                | 单机                | 多机                                     |
|---------------------|---------------------|------------------------------------------|
| `dist.barrier()`   | 本机 IPC，毫秒级     | **跨节点 NCCL/Gloo，依赖网络**             |
| LoRA save           | 本机 NFS 写          | **跨节点 NFS 写，可能有延迟**              |
| SGLang 加载         | 本机 NFS 读          | **多节点同时从 NFS 读，可能争锁**           |
| `fut.result()`     | HTTP 到本机 SGLang    | HTTP 到远程 SGLang，经过 Ray               |
| `asyncio.gather`   | 所有 worker 同机      | **worker 分布在不同节点，RPC 延迟不一致**   |

## weight_update_mode 对比：disk vs xccl

### disk 模式

```
Actor (Archon/FSDP) ──save──> NFS (safetensors) ──HTTP /load_lora_adapter──> SGLang
```

- 支持 LoRA
- 依赖 NFS，有 IO 瓶颈和文件锁风险

### xccl 模式

```
Actor (Archon/FSDP) ──NCCL broadcast──> SGLang (GPU memory)
```

- 不经过磁盘，直接 GPU-to-GPU
- **SGLang 端不支持 LoRA**（会 raise ValueError）
- 仅支持 full model weight update

## 关键代码文件索引

| 文件 | 职责 |
|------|------|
| `areal/trainer/rl_trainer.py` | 训练主循环，编排 pause/update/resume |
| `areal/infra/controller/train_controller.py` | 管理 actor workers，分发 RPC 调用 |
| `areal/infra/controller/rollout_controller.py` | 管理 SGLang servers，控制 rollout |
| `areal/experimental/engine/archon_engine.py` | Archon 训练引擎（当前使用） |
| `areal/experimental/engine/archon_weight_sync.py` | Archon 的 weight update 实现 |
| `areal/experimental/engine/archon_lora_checkpoint.py` | LoRA adapter 保存逻辑 |
| `areal/engine/fsdp_engine.py` | FSDP 训练引擎（备选 backend） |
| `areal/engine/sglang_remote.py` | SGLang 推理引擎封装，构造 HTTP 请求 |
