# Ctrl-World on one AMD Radeon Pro W7900

上游是 [Robert-gyj/Ctrl-World](https://github.com/Robert-gyj/Ctrl-World)（MIT，清华大学，2025）。这篇复现只走它 README 里的第一条主路径：在世界模型里回放 DROID 记录的动作。π0.5 闭环不在这次验证里。

权重、完整数据缓存和容器层留在跑实验的机器上。这里是移植后的代码，以及这次跑出来的日志和视频。

## 环境

- 镜像：`rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0`
- GPU：一张 AMD Radeon Pro W7900D（gfx1100，47.98 GB）
- 容器内只看见这一张卡：

```bash
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
```

`/dev/kfd` 会把整机的卡都暴露给容器，所以上面三个变量才是单卡边界。不要 `pip install torch`。PyPI 上的 torch 是 CUDA 轮子，装进环境后会挡住镜像里的 `2.11.0+rocm7.14.0`。

镜像的 Python 在 `/opt/venv`。其它依赖装在另一个虚拟环境里，用 `PYTHONPATH` 指过去。`scripts/run_replay_radeon.sh` 就是这样做的。依赖列表在 `requirements-rocm.txt`，里面没有 torch。

ROCm 镜像是 CPython 3.12，decord 没有对应的轮子。`radeon_video.py` 用 PyAV 读 mp4，接口仍是 `VideoReader.get_batch().asnumpy()`。

## 权重

| 用途 | 仓库 |
| --- | --- |
| 视频扩散底座 | `stabilityai/stable-video-diffusion-img2vid` |
| 文本编码器 | `openai/clip-vit-base-patch32` |
| Ctrl-World 权重 | `yjguo/Ctrl-World` 的 `checkpoint-10000.pt` |

示例轨迹用上游仓库自带的 `dataset_example/droid_subset`，不要下那 370 GB 的全量 DROID。

## 复现

容器里，工作目录是这份代码：

```bash
bash scripts/run_replay_radeon.sh
```

脚本对应的上游命令是：

```bash
python scripts/rollout_replay_traj.py \
  --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset \
  --svd_model_path <svd> \
  --clip_model_path <clip> \
  --ckpt_path <checkpoint-10000.pt> \
  --task_type replay
```

配置在 `config.py` 的 `task_type == "replay"`：三条验证轨迹，每条 12 次交互，每次 5 帧动作、50 步 Euler，精度 bfloat16。

## 验证

2026-10-25，单卡，进程退出码 0。

| 项 | 结果 |
| --- | --- |
| 设备 | AMD Radeon Pro W7900D，可见设备数 1 |
| PyTorch | 2.11.0+rocm7.14.0，HIP 7.14.60850 |
| 开始 | 12:15:56 UTC |
| 结束 | 12:24:57 UTC |
| 轨迹 | 899，18599，199，各 12 步 |
| 视频 | `results/videos/` 下 3 个 mp4 |

三条指令分别是 “Move the banana to the right”、“Pick the black thing from the chair and place it on the table”、“Move the green towel to the right of the lid”。日志摘要在 `results/replay_key.log`。

## 许可

上游许可证见 `LICENSE.txt`。本仓库的改动是让这条回放在 Radeon 上跑完，不改变扩散步数和网络结构。
