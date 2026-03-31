# Pitfalls Log

记录 AReaL LoRA 训练调试过程中踩过的坑。

---

## 1. `allocation_mode` 已 deprecated

**现象：** `MissingMandatoryValue: rollout.backend`

**原因：** 参考配置用的旧写法 `allocation_mode: sglang:d4+archon:d2t2`，新版 AReaL 要求每个 engine 单独声明 backend。

**修复：**
```yaml
# 旧（不再支持）
allocation_mode: sglang:d4+archon:d2t2

# 新
rollout:
  backend: sglang:d4
actor:
  backend: archon:d2t2
ref:
  backend: ${actor.backend}
```

---

## 2. `ref.backend` 也是必填

**现象：** `MissingMandatoryValue: ref.backend`

**原因：** 只改了 rollout 和 actor，忘了 ref model 也需要 backend 字段。

**修复：** `ref.backend: ${actor.backend}`

---

## 3. 数据集字段名不匹配

**现象：** `KeyError: 'label'`，且 DAPO 数据集有 179 万行

**原因：** 用错了数据集。HuggingFace 上的 `DAPO-Math-17k` parquet 版本有 179 万行（每个 prompt 重复 ~103 次），字段是 `reward_model.ground_truth` 不是 `label`。

**修复：** 换用本地处理好的 jsonl 版本（`/jpfs/.../datasets/dapo-math-17k/`），17,398 行，字段 `prompt` + `label`，直接匹配。

---

## 4. `scheduler.type: null` 不再支持

**现象：** `NotImplementedError: Unknown scheduler type: None`

**原因：** 旧版支持 null（由外部 scheduler 管理），新版只接受 `local` / `ray` / `slurm`。

**修复：** `scheduler.type: local`（单机），多机用 `ray`。

---

## 5. AReaL submodule 分支错误导致 LoRA 保存失败

**现象：** `No such file or directory: .../initial_lora/adapter_config.json`，保存出来的是完整模型而不是 LoRA adapter。

**原因：** submodule 在 `main` 分支（同步上游，没有 PR #1015 的 LoRA 代码）。`main` 上的 `archon_engine.save()` 不认识 LoRA，走了 `save_model_to_hf` 全模型保存路径。

**修复：** 确保 `modules/AReaL` 在 `lora_init` 分支（含 PR #1015）。训练前检查：
```bash
git -C modules/AReaL branch --show-current  # 应该输出 lora_init
```

---

## 6. 容器内旧 AReaL 可能覆盖 PYTHONPATH

**现象：** worker 子进程可能 import 容器内 `/AReaL/` 的旧代码而非我们的 `modules/AReaL`。

**原因：** LocalScheduler 启动的 worker 进程，`scheduling_spec.env_vars` 为空，可能没有正确继承 PYTHONPATH。

**修复：** 在 yaml 的 `scheduling_spec` 里显式设置：
```yaml
scheduling_spec:
  - env_vars:
      PYTHONPATH: /jpfs/shenzhennan.1/PeRL-Init/modules/AReaL
```

---

## 7. LoRA + torch.compile 冲突

**现象：** `Compilation of intermediate hooks requires compiled autograd`

**原因：** LoRA 实现中 `lora_linear.py` 用了 `result.register_hook(lambda grad: grad)` 注册 backward hook。`torch.compile` (TorchDynamo) 需要把 forward 编译成静态计算图，但 `register_hook` 是动态操作无法编译。FSDP2 + gradient checkpointing 进一步加剧了 hook 冲突。

**修复：** 关闭 compile：
```bash
+actor.archon.enable_compile=false
```

---

## 通用经验

- 从旧版 AReaL 配置迁移时，逐个检查 deprecated 字段
- 训练前确认 submodule 分支、PYTHONPATH、数据集字段名
- 参考项目的配置不能直接复制，需要适配当前版本
