# Ctrl-World on one AMD Radeon Pro W7900

上游是 [Robert-gyj/Ctrl-World](https://github.com/Robert-gyj/Ctrl-World)（MIT，ICLR 2026）。这次复现走它 README 里的第一条主路径：在世界模型里回放 DROID 记录的动作。π0.5 闭环和训练不在这次验证里。

仓库里是移植后的代码、这次跑出来的日志和视频，以及从零复现用的脚本。权重、依赖环境和容器层不进仓库。

## 环境

- 镜像：`rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0`
- GPU：一张 AMD Radeon Pro W7900D（gfx1100，47.98 GB）
- 磁盘：三份权重合计约 42 GB

容器只挂 `/dev/kfd` 和 GPU 0 的渲染节点 `/dev/dri/renderD128`，不设任何 `*_VISIBLE_DEVICES`。容器里 `rocminfo` 只列出一个 gfx1100，`torch.cuda.device_count()` 是 1，记录在 `results/isolation.log`。

换卡时按 PCI 地址找渲染节点，不要按编号猜。这台机器的 `card0` 是主板 BMC 的 ASPEED 显示控制器，GPU 0 是 `card1`，渲染节点却是 `renderD128`：

```bash
rocm-smi --showbus
for r in /sys/class/drm/renderD*; do echo "$r $(basename "$(readlink -f "$r/device")")"; done
RENDER_NODE=/dev/dri/renderD129 bash scripts/radeon_start_container.sh "$WORK"
```

`run_replay_radeon.sh` 仍把 `HIP_VISIBLE_DEVICES`、`ROCR_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES` 设成 0。容器若挂了整组 `/dev/dri`，回放也只用一张卡。

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

# 回放视频的 PSNR
docker exec radeon-ctrl-world bash -c 'cd /workspace/src && \
  PYTHONPATH=/workspace/venv/lib/python3.12/site-packages /opt/venv/bin/python \
  scripts/radeon_replay_metrics.py synthetic_traj/Rollouts_replay/video/*.mp4'
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

镜像的 ROCm PyTorch 装在 `/opt/venv`。从它建出来的 `/workspace/venv` 以 `/usr/bin/python3.12` 为底，`--system-site-packages` 继承的是 Ubuntu 的系统目录，看不见 `/opt/venv` 里的 torch。`accelerate` 依赖 torch，pip 于是从 PyPI 装了 CUDA 版的 `torch 2.14.1` 和 `triton`、15 个 `nvidia-*`、3 个 `cuda-*` 包。`radeon_install_deps.sh` 装完依赖后把它们从这个环境里卸掉。回放用镜像的 `/opt/venv/bin/python`，再用 `PYTHONPATH` 指向 `/workspace/venv` 的 site-packages。

### 视频读取

`radeon_video.py` 用 PyAV 代替 decord 读 mp4，接口仍是 `VideoReader.get_batch().asnumpy()`。写它时我以为 decord 没有 CPython 3.12 的轮子，这个判断错了：decord 0.6.0 的 Linux 轮子标签是 `py3-none-manylinux2010_x86_64`，在这个镜像里能装能用。在 `droid_subset` 全部 12 个验证视频上，两者解出的帧逐像素相同，最大差值为 0。回放的输入因此和上游走 decord 时一致。想贴近上游，可以把 import 改回 decord。

### 检查点

`checkpoint-10000.pt` 里的 2525 处设备标记都是 `cuda:0`。在 ROCm 版 PyTorch 里，`cuda:0` 就是第一张 HIP 设备。代码里加的 `map_location="cpu"` 让 9.3 GB 的张量先落在内存里，再随模型一起搬上显卡。

## 验证

2026-10-25，单卡，容器只挂 `renderD128`，进程退出码 0。

| 项 | 结果 |
| --- | --- |
| 设备 | AMD Radeon Pro W7900D，容器内可见 1 张 GPU |
| PyTorch | 2.11.0+rocm7.14.0，HIP 7.14.60850 |
| 开始 / 结束 | 15:52:52 / 16:02:27 UTC，共 575 秒 |
| 轨迹 | 899、18599、199，各 12 次交互 |
| 每次交互 | 50 步去噪平均 12.37 秒；后两条轨迹各用 179 秒、169 秒，即 14–15 秒一次 |
| 显存峰值 | 30.41 GiB，`rocm-smi` 每 2 秒采样，含 PyTorch 缓存 |
| PSNR | 预测对真值 22.50 dB，重复首帧的基线 16.98 dB |

三条指令分别是 “Move the banana to the right”、“Pick the black thing from the chair and place it on the table”、“Move the green towel to the right of the lid”。视频在 `results/videos/`，每帧上半是解码后的真值，下半是预测，三个视角从左到右是两台外部相机和腕部相机。

`scripts/radeon_replay_metrics.py` 按这个布局切开视频算 PSNR，跳过模型拿来做条件的第 0 帧。基线一直重复第一帧真值，相当于一个预测“什么都不动”的世界模型。差距最大的是腕部相机：预测 17.2–20.0 dB，基线 11.0–12.5 dB。回放没有固定随机种子，这次和上一次的预测互相比是 20.8–23.0 dB，和预测对真值差不多，所以单条轨迹零点几 dB 的差别说明不了什么。逐条数字在 `results/metrics.txt`。

第一次运行挂的是整组 `/dev/dri`，靠三个可见设备变量限定单卡，结果在 `results/run1_all_dri/`：退出码 0，后两条轨迹各用 168 秒、169 秒，PSNR 22.30 dB 对基线 16.75 dB。

| 文件 | 内容 |
| --- | --- |
| `results/isolation.log` | 容器里的设备节点、`rocminfo` 和 torch 看到的 GPU |
| `results/replay_key.log` | 回放日志摘要 |
| `results/timing.txt` | 每次去噪的耗时和每条轨迹的用时 |
| `results/vram_peak.txt`、`vram_trace.csv` | 显存峰值和采样记录 |
| `results/metrics.txt` | 逐条轨迹、逐个视角的 PSNR |
| `results/port.diff` | 对上游文件的改动 |

## 许可

上游许可证见 `LICENSE.txt`。本仓库的改动是让这条回放在 Radeon 上跑完，扩散步数和网络结构都没有改。
