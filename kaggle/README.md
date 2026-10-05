> GPU 历史实测记录。当前源码冻结为 0.5.1；以下 2026-09 的结果不作为本版新成绩。克隆后先用自己的账号运行 `python -m scripts.build_kaggle_notebook --owner YOUR_USERNAME` 生成 Notebook，生成产物不提交 Git。

# 曦源 Kaggle GPU Notebook

私有 Notebook：https://www.kaggle.com/code/hunuunnn/xiyuan-stitch-gpu-mvp

本机无须安装 PyTorch 或 Diffusers。Notebook 在 Kaggle 免费 GPU 上运行完整配准、局部扩散重绘和融合，结果可以下载回本机。它是按需运行的实验，不是桌面端的常驻 API。

## 当前配置

- 私有 Notebook，启用 Internet 下载公开模型。
- 申请 NVIDIA Tesla T4（平台可能分配 T4 ×2）；代码只使用第一块 GPU。
- 单次运行最长 1800 秒，计算结束自动退出。
- 使用 Kaggle 自带的 PyTorch/CUDA，不覆盖其 GPU 驱动环境。
- Diffusers 0.35.1、Transformers 4.56.2、Accelerate 1.10.1。
- SD 1.5 Inpainting：`stable-diffusion-v1-5/stable-diffusion-inpainting`；每次运行记录实际模型 revision，指定 `variant='fp16'` 加载仓库中的 safetensors 权重。
- 512×512 Patch，20 步，固定 seed 2026，比较 strength 0.35 和 0.50。
- 样例为程序生成的砖墙，包含轻微局部错位和亮度变化。它用于验证真实推理链路，效果仍需真实照片评估。

## 已完成的 GPU 实测（2026-09-09）

Kaggle Notebook 第 4 版已完成，两组真实推理均通过。GPU 为 Tesla T4，可用显存 14.56 GiB，PyTorch 2.10.0+cu128 / CUDA 12.8；模型 revision 为 `8a4288a76071f7280aedbdb3253bdb9e9d5d84bb`，使用 fp16 safetensors。

| 案例 | 模型加载 | 推理 | 完整拼接流程 | 显存分配峰值 |
| --- | ---: | ---: | ---: | ---: |
| strength 0.35，首次运行 | 72.77 秒 | 4.86 秒 | 78.44 秒 | 3.18 GiB |
| strength 0.50，复用模型 | 接近 0 秒 | 2.45 秒 | 2.90 秒 | 3.18 GiB |

设置 20 步，两种 strength 对应实际去噪 7 / 10 步。以上不包含 Kaggle 排队、启动容器和安装依赖的时间；首次模型加载含权重下载。第二组完整流程仍包含配准和传统融合，不能视为固定服务延迟。

两组 Mask 内像素均发生变化，软 Mask 外最大像素差均为 0。图像结构保持可辨认，但接缝附近有平滑和纹理改动；这次只验通推理，尚未证明优于传统融合。源 Notebook 与远程私有配置快照在 `outputs/kaggle/submitted-v4`，下载报告和对比图在 `outputs/kaggle/v4`。第 3 版的早期成功记录保存在 `outputs/kaggle/v3`。

任务已经结束。本次账户查询显示 GPU 已用约 0.11 小时，剩余约 29.89 / 30 小时，平台显示刷新时间为 `2026-09-12T00:00:00`。额度和可分配 GPU 以之后的 `quota` 查询为准。

## 本机操作

已经提供操作脚本，在项目目录运行 `./scripts/Invoke-Kaggle.ps1 status` 或 `./scripts/Invoke-Kaggle.ps1 quota` 即可。`push` 会重新生成并运行私有 Notebook，`download` 会下载到按时间命名的新目录。脚本仅在当前进程读取凭据，退出后恢复环境变量；换电脑时用 `-Credentials '你的路径/kaggle.json'` 指定文件。

```powershell
.\scripts\Invoke-Kaggle.ps1 push      # 提交并运行
.\scripts\Invoke-Kaggle.ps1 status    # 等待显示 COMPLETE
.\scripts\Invoke-Kaggle.ps1 download  # 完成后下载结果
```

从项目根目录使用 PowerShell。API Key 保留在用户提供的位置，通过当前进程的 `KAGGLE_CONFIG_DIR` 读取；不复制到项目，也不上传至 Notebook。

```powershell
$env:KAGGLE_CONFIG_DIR = 'C:/private'

# 查看剩余免费额度
.\.venv\Scripts\kaggle.exe quota

# 修改算法源码后，重新打包并提交一个新版本
.\.venv\Scripts\python.exe -m scripts.build_kaggle_notebook --owner hunuunnn
.\.venv\Scripts\kaggle.exe kernels push -p kaggle --accelerator NvidiaTeslaT4 --timeout 1800

# 查看状态和日志
.\.venv\Scripts\kaggle.exe kernels status hunuunnn/xiyuan-stitch-gpu-mvp
.\.venv\Scripts\kaggle.exe kernels logs hunuunnn/xiyuan-stitch-gpu-mvp

# 完成后下载到一个新的结果目录，避免覆盖旧实验
.\.venv\Scripts\kaggle.exe kernels output hunuunnn/xiyuan-stitch-gpu-mvp -p outputs/kaggle/latest
```

`push` 会立即申请 GPU 并运行。修改 Notebook 后再提交，不要在旧任务运行期间连续重复提交。

2026-09-09 首轮实测：Kaggle 提供的 PyTorch 2.10.0+cu128 不支持 P100 的 sm_60 架构，出现 `no kernel image is available`。本配置因此改为 T4，并在安装依赖前执行一次实际 CUDA 张量运算来检查兼容性。

新电脑先创建 `.venv`，然后执行 `.\.venv\Scripts\python.exe -m pip install -e ".[gui,kaggle,dev]"`。Notebook 由脚本生成，建议编辑 `scripts/kaggle_smoke.py` 和项目源码后重建，避免下次生成覆盖手动修改。

## 输出

- `gpu.json`：分配到的 GPU、显存容量、PyTorch/CUDA 版本。
- `results/report.json`：真实执行状态、模型 revision、每个案例的耗时和显存峰值。
- `results/brick-strength-*/input_a.png` 和 `input_b.png`：测试输入。
- `traditional.png`、`mask.png`、`soft_mask.png`、`ai_result.png`、`comparison.jpg`：输入与生成结果的比较。
- `config.json`：每个案例的独立参数副本。

第一个案例的 `ai_seconds` 包含首次模型加载，第二个案例复用模型。不能把第一个案例的总时间当成稳定推理延迟。这里的显存峰值也不代表所有高清图片都能满足同一显存要求。

## 换成自己的图片

在 Kaggle 编辑页添加一个私有图片 Dataset，启用 GPU，然后顺序运行前四个准备单元。可新建一个代码单元调用已经解包的项目：

```python
from pathlib import Path
from xiyuan_mvp.pipeline import StitchPipeline
from xiyuan_mvp.config import load_config
from xiyuan_mvp.image_io import write_image

config = load_config()
config['inpainting']['device'] = 'cuda'
config['inpainting']['variant'] = 'fp16'
config['inpainting']['local_files_only'] = False
config['inpainting']['cache_dir'] = '/kaggle/temp/xiyuan-hf-cache'
config['inpainting']['prompt'] = 'seamless continuation of the surrounding texture'
pipeline = StitchPipeline(config)
result = pipeline.run(
    '/kaggle/input/YOUR_PRIVATE_DATASET/a.jpg',
    '/kaggle/input/YOUR_PRIVATE_DATASET/b.jpg',
    use_ai=True,
)
write_image('/kaggle/working/your_result.png', result.final_image)
print(result.metrics)
```

API Key 不需要放进 Notebook；公开模型下载不需要这个 Key。GPU 无法分配或免费额度耗尽时，先查看 Session options / quota，不启用付费云服务。
