# Ctrl-World on one AMD Radeon Pro W7900

上游是 [Robert-gyj/Ctrl-World](https://github.com/Robert-gyj/Ctrl-World)（MIT，清华大学，2025）。这次复现走它 README 里的第一条主路径：在世界模型里回放 DROID 记录的动作。π0.5 闭环不在这次验证里。

仓库里是移植后的代码、这次跑出来的日志和视频，以及从零复现用的脚本。权重、依赖环境和容器层不进仓库。

## 环境

- 镜像：`rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0`
- GPU：一张 AMD Radeon Pro W7900D（gfx1100，47.98 GB）
- 磁盘：三份权重合计约 42 GB

容器挂了 `/dev/kfd` 和整组 `/dev/dri`。单卡由下面三个变量限定，进程里 `torch.cuda.device_count()` 是 1：

```bash
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
```

只挂 GPU 0 对应的那个渲染节点是更严格的隔离，这次没有这样做。

## 目录约定

宿主机上准备一个工作目录，挂到容器的 `/workspace`。脚本按这些路径找东西：

| 容器内路径 | 内容 |
| --- | --- |
| `/workspace/src` | 本仓库 |
| `/workspace/venv` | torch 以外的依赖 |
| `/workspace/weights/svd`、`clip`、`ctrl-world` | 三份权重 |
| `/workspace/results` | 回放日志和退出码 |

## 复现

```bash
# 宿主机
WORK=/path/to/workdir
git clone https://github.com/charyang-ai/ctrl-world "$WORK/src"
bash "$WORK/src/scripts/radeon_start_container.sh" "$WORK"

# 依赖、权重、回放都在容器里跑
docker exec radeon-ctrl-world bash /workspace/src/scripts/radeon_install_deps.sh
docker exec radeon-ctrl-world bash /workspace/src/scripts/radeon_download_weights.sh
docker exec radeon-ctrl-world bash /workspace/src/scripts/run_replay_radeon.sh
```

这次是经由清华 PyPI 镜像和 hf-mirror.com 下载的。要这样做，在 `docker exec` 后加 `-e PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` 或 `-e HF_ENDPOINT=https://hf-mirror.com`。直连 huggingface.co 时如果 SVD 仓库返回 401，先在网页上同意它的许可，再设置 `HF_TOKEN`。

回放对应的上游命令是：

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

配置在 `config.py` 的 `task_type == "replay"`：三条验证轨迹，每条 12 次交互，每次 5 帧动作、50 步 Euler，精度 bfloat16。示例轨迹用仓库自带的 `dataset_example/droid_subset`，不需要 370 GB 的全量 DROID。

| 用途 | Hugging Face 仓库 |
| --- | --- |
| 视频扩散底座 | `stabilityai/stable-video-diffusion-img2vid` |
| 文本编码器 | `openai/clip-vit-base-patch32` |
| Ctrl-World 权重 | `yjguo/Ctrl-World` 的 `checkpoint-10000.pt` |

### 依赖为什么这样装

镜像的 ROCm PyTorch 装在 `/opt/venv`。从它建出来的 `/workspace/venv` 以 `/usr/bin/python3.12` 为底，`--system-site-packages` 继承的是 Ubuntu 的系统目录，看不见 `/opt/venv` 里的 torch。`accelerate` 依赖 torch，pip 于是从 PyPI 装了 CUDA 版的 `torch 2.14.1+cu130` 和一串 `nvidia-*` 包。`radeon_install_deps.sh` 装完依赖后，把 `torch`、`triton`、`nvidia-*`、`cuda-*` 从这个环境里卸掉。回放用镜像的 `/opt/venv/bin/python`，再用 `PYTHONPATH` 指向 `/workspace/venv` 的 site-packages。

### 视频读取

`radeon_video.py` 用 PyAV 代替 decord 读 mp4，接口仍是 `VideoReader.get_batch().asnumpy()`。写它时我以为 decord 没有 CPython 3.12 的轮子，这个判断错了：decord 0.6.0 的 Linux 轮子标签是 `py3-none-manylinux2010_x86_64`，在这个镜像里能装能用。在 `droid_subset` 全部 12 个验证视频上，两者解出的帧逐像素相同，最大差值为 0。回放的输入因此和上游走 decord 时一致。想贴近上游，可以把 import 改回 decord。

### 检查点

`checkpoint-10000.pt` 里的 2525 处设备标记都是 `cuda:0`。在 ROCm 版 PyTorch 里，`cuda:0` 就是第一张 HIP 设备。代码里加的 `map_location="cpu"` 让 9.3 GB 的张量先落在内存里，再随模型一起搬上显卡。

## 验证

2026-10-25，单卡，进程退出码 0。

| 项 | 结果 |
| --- | --- |
| 设备 | AMD Radeon Pro W7900D，可见设备数 1 |
| PyTorch | 2.11.0+rocm7.14.0，HIP 7.14.60850 |
| 开始 | 12:15:56 UTC |
| 结束 | 12:24:57 UTC |
| 轨迹 | 899，18599，199，各 12 次交互 |
| 每次交互 | 约 14 秒（后两条轨迹各用 168 秒、169 秒） |
| 视频 | `results/videos/` 下 3 个 mp4 |

三条指令分别是 “Move the banana to the right”、“Pick the black thing from the chair and place it on the table”、“Move the green towel to the right of the lid”。日志摘要在 `results/replay_key.log`，对上游文件的改动在 `results/port.diff`。

## 许可

上游许可证见 `LICENSE.txt`。本仓库的改动是让这条回放在 Radeon 上跑完，扩散步数和网络结构都没有改。
