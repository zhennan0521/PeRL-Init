# Experiments

## Models & Datasets


| ID  | Model                         | Dataset       | Notes                                        |
| --- | ----------------------------- | ------------- | -------------------------------------------- |
| M1  | DeepSeek-R1-Distill-Qwen-1.5B | DAPO-Math-17k | 默认 target_modules: q/k/v/o/gate/up/down_proj |


## Results


| #   | Model | Method  | Rank | LR   | Alpha | Async    | Infra    | Steps | AIME | Notes          | Script                                  | Log                                                        |
| --- | ----- | ------- | ---- | ---- | ----- | -------- | -------- | ----- | ---- | -------------- | --------------------------------------- | ---------------------------------------------------------- |
| 1   | M1    | Full FT | -    | 2e-5 | -     | sync     | tp1dp8×1 | -     |      | baseline       | run_full_sync_1.5b.sh                   | full_sync_1.5b_dapo/202603311429                           |
| 2   | M1    | Full FT | -    | 2e-5 | -     | async(2) | tp1dp8×1 | -     |      | baseline       | run_full_async_1.5b.sh                  | full_async_1.5b_dapo/202603311428                          |
| 3   | M1    | LoRA    | 32   | 5e-5 | 32    | async(2) | tp2dp4×1 | -     |      | baseline+debug | run_lora_async_1.5b_tp2.sh              | lora-1.5b-dapo-async-tp2-lr5e-5/202604011522               |
| 4   | M1    | LoRA    | 32   | 5e-5 | 32    | async(2) | tp1dp8×1 | -     |      | baseline+debug | run_lora_async_1.5b_tp1_fixed_review.sh | lora-1.5b-dapo-async-lr5e-5-tp1-fixed-review/202604011555  |
| 5   | M1    | LoRA    | 32   | 5e-5 | 32    | async(2) | tp1dp8×4 | -     |      | baseline+debug | run_lora_async_4node_1.5b.sh            | lora-1.5b-dapo-async-4node-lr5e-5-new-machine/202604021423 |


## Conclusions

- async (offpolicy=2) 优于 sync，full 和 LoRA 均如此
- LoRA lr=5e-5 暂优于 lr=2e-5；Full FT lr=2e-5 好于 1e-5

## Reference: PeRL 超参汇总

> 来源：`modules/PeRL/recipes/trl/openr1/dapo_*.sh`
> Model: DeepSeek-R1-Distill-Qwen-1.5B | Dataset: DAPO-Math-17k | 同步训练 (TRL)
> 规律：alpha = 2×rank, lr = 1e-5, dropout = 0.05 几乎是统一惯例

| Method | Rank | Alpha | Dropout | LR | GPUs | GradAcc | Batch | EffBatch | Steps |
|--------|------|-------|---------|----|------|---------|-------|----------|-------|
| LoRA | 16 | 32 | 0.05 | 1e-5 | 4 | 8 | 4 | 128 | 1024 |
| LoRA r=32 | 32 | 64 | 0.05 | 1e-5 | 2 | 4 | 4 | 32 | 8192 |
| rsLoRA | 32 | 64 | 0.05 | 1e-5 | 2 | 16 | 4 | 128 | 1024 |
| DoRA | 32 | 64 | 0.05 | 1e-5 | 4 | 8 | 4 | 128 | 1024 |
| PiSSA | 32 | 64 | 0.05 | 1e-5 | 4 | 8 | 4 | 128 | 1024 |
| MiLoRA | 32 | 64 | 0.05 | 1e-5 | 4 | 8 | 4 | 128 | 1024 |
| MiLoRA++ | 16 | 32 | 0.05 | 1e-5 | 4 | 8 | 4 | 128 | 1024 |
| LoRA+ | 32 | 64 | 0.05 | 1e-5 | 4 | 8 | 4 | 128 | 1024 |
| LoRA-FA | 32 | 64 | 0.05 | 1e-5 | 4 | 8 | 4 | 128 | 1024 |
| AdaLoRA | 32 | 64 | 0.05 | 1e-5 | 4 | 8 | 4 | 128 | 1024 |
| IA3 | 32 | 64 | 0.05 | 1e-5 | 4 | 8 | 4 | 128 | 1024 |
| VeRA | 32 | 64 | 0.05 | 1e-5 | 4 | 8 | 4 | 128 | 1024 |
| MiSS | 64 | 128 | 0.05 | 1e-5 | 4 | 8 | 4 | 128 | 1024 |
| SliceFine | 32 | 64 | 0.05 | 1e-5 | 4 | 8 | 1 | 32 | 1024 |
| LayerNorm | 16 | 32 | 0.05 | 1e-5 | 4 | 8 | 4 | 128 | 1024 |
| Full FT | - | - | - | 1e-6 | 4 | 8 | 4 | 128 | 1024 |