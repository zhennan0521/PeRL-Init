# CLAUDE.md

## Project

- LoRA Init 项目，计划和进度见 [TODO.md](TODO.md)
- AReaL submodule 在 `modules/AReaL`，开发在 `lora_init` 分支，main 保持与上游 `inclusionAI/AReaL` 同步
- PeRL submodule 在 `modules/PeRL`
- 训练脚本放在 PeRL-Init 仓库，不要放在 AReaL 子模块里
- **时刻做好版本管理**：改动及时 commit，保持各分支和子模块状态清晰，push 前确认分支正确

## Proxy Configuration

If GitHub is unreachable, set this proxy before running git/network commands:

```bash
export http_proxy=socks5h://imfree:Jdea2025@ec2-13-213-15-247.ap-southeast-1.compute.amazonaws.com:3128
export https_proxy=socks5h://imfree:Jdea2025@ec2-13-213-15-247.ap-southeast-1.compute.amazonaws.com:3128
```
