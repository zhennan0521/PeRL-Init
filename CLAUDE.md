# CLAUDE.md

## Project

- LoRA Init 项目，计划和进度见 [TODO.md](TODO.md)
- AReaL submodule 在 `modules/AReaL`，开发在 `lora_init` 分支，main 保持与上游 `inclusionAI/AReaL` 同步
- PeRL submodule 在 `modules/PeRL`
- 训练脚本放在 PeRL-Init 仓库，不要放在 AReaL 子模块里
- **时刻做好版本管理**：改动及时 commit，保持各分支和子模块状态清晰，push 前确认分支正确
- **已跑实验的脚本不可修改**：一旦某个 `.sh` 或 `.yaml` 已用于跑实验，就不能再改动，因为 log 需要溯源到原始配置。如需调整参数，应复制一份新脚本

## Docs

| 文件 | 内容 |
|------|------|
| [TODO.md](TODO.md) | 项目计划和进度 |
| [tutorial.md](tutorial.md) | AReaL 框架 + LoRA 实现 Code Map（带 [file:line] 引用） |
| [PITFALLS.md](PITFALLS.md) | 已知 bug 和踩坑记录 |
| [reference.md](reference.md) | 研究参考资料（论文等） |
| [experiments.md](experiments.md) | 实验 setting 与结果对照表 |

## Proxy Configuration

If GitHub is unreachable, set this proxy before running git/network commands:

```bash
export http_proxy=socks5h://imfree:Jdea2025@ec2-13-213-15-247.ap-southeast-1.compute.amazonaws.com:3128
export https_proxy=socks5h://imfree:Jdea2025@ec2-13-213-15-247.ap-southeast-1.compute.amazonaws.com:3128
```
