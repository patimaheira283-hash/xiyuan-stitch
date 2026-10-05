# 基于扩散模型的图像无缝拼接 MVP 实施方案

## 1. MVP 目标

在普通消费级 GPU 电脑上，完成一个支持两张图片拼接的桌面程序。用户导入两张具有重叠区域的平面图片，程序自动完成初步对齐和接缝生成，用户可以手动修正待重绘区域，随后使用 Stable Diffusion Inpainting 对接缝局部进行修复，并导出高清结果。

第一版要证明三件事：

1. 两张图片能够稳定对齐并生成可用的初始拼接图。
2. 扩散模型能够改善砖墙、木纹、瓷砖等复杂纹理的接缝连续性。
3. 整个流程能够由非开发者通过 GUI 完成，并在 8GB 显存设备上运行。

第一版不追求训练新模型，也不追求一次处理任意数量图片。核心是做出一条可重复运行、可以人工修正、能够展示前后效果的完整流水线。

## 2. 第一版范围

### 必须实现

- 导入两张 JPG、PNG 图片。
- 图片缩放、平移和局部放大查看。
- 自动图像配准。
- 生成初始拼接图。
- 自动生成接缝待修复区域。
- 画笔添加、橡皮擦除和清空 Mask。
- 对接缝局部进行扩散模型重绘。
- 原图、传统融合结果、AI 结果三者对比。
- 支持 PNG、JPG 导出。
- 显示进度、错误信息、显存不足提示。
- 保存本次运行的参数和随机种子。

### 暂不实现

- 三张及以上图片的全自动拼接。
- 360 度全景和球面投影。
- 模型训练或微调。
- 自动图像描述和自动 Prompt 生成。
- 云端推理。
- 完整的 ComfyUI 嵌入。
- 移动端和网页端。

多图、全景和自动 Prompt 都会显著扩大问题范围，留到 MVP 验证有效之后再做。

## 3. 推荐产品流程

```text
导入图片 A、B
      ↓
低分辨率预览与特征匹配
      ↓
估计变换矩阵并 Warp
      ↓
生成传统初始拼接图
      ↓
自动计算接缝和 Mask
      ↓
用户手动调整 Mask
      ↓
裁剪接缝 Patch
      ↓
Stable Diffusion Inpainting
      ↓
颜色校正与软 Mask 融合
      ↓
高清结果导出
```

程序应始终保留原始图片和原始坐标。预览时可以缩小图片，真正导出时只对接缝附近的局部区域进行高质量处理，避免整张大图进入显存。

## 4. 技术路线

### 4.1 图像配准

MVP 采用两级配准策略：

1. 默认使用 OpenCV SIFT + Lowe ratio test + RANSAC，先保证流程稳定。
2. SIFT 匹配点不足或单应矩阵质量不合格时，再切换 LoFTR。

推荐复用：

- [OpenCV](https://github.com/opencv/opencv)：特征提取、Warp、图像处理和导出。
- [Kornia](https://github.com/kornia/kornia)：PyTorch 几何操作和 LoFTR 封装。
- [官方 LoFTR](https://github.com/zju3dv/LoFTR)：需要更高鲁棒性时作为备用实现。

配准阶段需要记录以下结果：匹配点数量、RANSAC 内点比例、单应矩阵、Warp 后重叠区域面积。如果内点比例过低或重叠区域过小，直接提示用户重新选择图片，不要把明显错误的结果送入扩散模型。

建议的初始判定条件：

- 有效匹配点不少于 12 个；
- RANSAC 内点不少于 8 个；
- 内点比例不低于 0.25；
- 变换后重叠区域占较小图片面积不低于 10%。

这些数值先作为工程默认值，后续根据测试集调整。

### 4.2 初始拼接和接缝生成

先把图片 B Warp 到图片 A 的坐标系中，得到两张图在同一画布上的重叠区域。MVP 不需要一开始实现复杂的全景优化，可以按以下方式生成初始结果：

1. 计算两张图的有效区域和重叠区域。
2. 在重叠区域中计算颜色差、梯度差和边缘强度。
3. 使用简单的最小代价路径确定初始接缝；实现困难时，先使用重叠区域中心线作为 fallback。
4. 将接缝路径膨胀 32 到 96 像素，生成扩散模型的 Mask。
5. 对 Mask 做高斯模糊，得到最终贴回原图时使用的软 Alpha Mask。

自动 Mask 只负责给出起点，GUI 必须允许用户增删区域。对于第一版，手动修正能力比自动接缝算法的复杂程度更重要。

### 4.3 局部扩散重绘

使用 [Hugging Face Diffusers](https://github.com/huggingface/diffusers) 调用 SD 1.5 Inpainting 模型。第一版先不强制加入 ControlNet，先把基础 Inpainting 跑通；ControlNet 作为第二个迭代加入。

推荐流程：

1. 根据 Mask 计算最小包围框。
2. 向外扩展 128 到 256 像素，确保模型能看到接缝两侧的上下文。
3. 将 Patch 缩放到 512×512；必要时支持 768×768 质量模式。
4. 将硬拼接结果作为输入图，将接缝带作为 Mask。
5. 使用一个保守的默认 Prompt，例如 `seamless continuation of the surrounding texture, preserve the original structure and lighting`。
6. 生成一张结果，允许用户修改随机种子后重试。
7. 将生成 Patch 缩放回原尺寸，只在软 Mask 范围内贴回。

建议初始参数：

| 参数 | 预览模式 | 质量模式 |
|---|---:|---:|
| Patch 尺寸 | 512×512 | 512×512 或 768×768 |
| Steps | 8 到 12 | 18 到 25 |
| Denoising strength | 0.35 到 0.50 | 0.45 到 0.65 |
| CFG | 3.5 到 5.0 | 4.0 到 6.0 |
| 候选数量 | 1 | 1 到 3 |

默认优先保留原图结构。Denoising strength 过高会让模型重新创造纹理，容易出现砖缝数量变化、木纹方向改变和局部物体变形。

### 4.4 ControlNet 和 LCM 的加入顺序

ControlNet 和 LCM 都不放在第一条可运行链路之前。

第二阶段加入 ControlNet Tile 或 Canny，用于比较以下三种结果：

- 普通 Inpainting；
- Inpainting + Canny；
- Inpainting + Tile。

以测试结果决定默认模式，而不是预先假设两种控制都一定有效。

第三阶段再加入 LCM-LoRA。LCM 只作为“快速预览”选项，质量模式继续使用普通采样，避免加速后细节质量下降却无法定位原因。

### 4.5 颜色校正和融合

生成结果贴回原图前，先在 Patch 边缘的未遮罩区域估计颜色差，在 LAB 空间对生成区域做轻量均值和标准差匹配。然后使用高斯模糊后的 Mask 做 Alpha 融合。

处理顺序：

```text
生成 Patch
 → LAB 颜色匹配
 → 限制亮度和色彩修正幅度
 → 软 Mask 融合
 → 可选的轻度局部锐化
```

颜色校正必须设置上限，避免为了消除色差而把整块生成区域染成异常颜色。

## 5. 软件架构

推荐目录结构如下：

```text
xiyuan_mvp/
├─ app.py
├─ ui/
│  ├─ main_window.py
│  ├─ image_canvas.py
│  ├─ mask_editor.py
│  └─ parameter_panel.py
├─ pipeline/
│  ├─ registration.py
│  ├─ warping.py
│  ├─ seam_mask.py
│  ├─ inpainting.py
│  ├─ blending.py
│  └─ pipeline.py
├─ models/
│  ├─ device.py
│  ├─ loftr_loader.py
│  └─ diffusion_loader.py
├─ workers/
│  └─ inference_worker.py
├─ utils/
│  ├─ image_io.py
│  ├─ coordinates.py
│  ├─ tiling.py
│  └─ run_log.py
├─ configs/
│  └─ default.yaml
├─ data/
│  ├─ raw/
│  └─ benchmark/
├─ outputs/
└─ requirements.txt
```

核心模块之间只传递数据对象，不要让 GUI 直接操作模型。建议定义一个统一的运行对象：

```python
class StitchResult:
    canvas_image: "np.ndarray"
    overlap_mask: "np.ndarray"
    seam_mask: "np.ndarray"
    result_image: "np.ndarray | None"
    homography: "np.ndarray"
    metrics: dict
```

推理线程通过信号返回进度、预览图和错误信息，主线程只负责更新界面。模型加载应在第一次运行时完成并缓存，不能每次点击“生成”都重新加载。

## 6. GUI 最小设计

主窗口分为三个区域：

```text
左侧：图片列表和参数
中间：可缩放画布
右侧：原图 / 初始拼接 / AI 结果预览
底部：进度、耗时、显存和错误提示
```

最少需要这些按钮：

- 导入图片 A；
- 导入图片 B；
- 自动配准；
- 生成 Mask；
- 画笔；
- 橡皮擦；
- 清空 Mask；
- 运行 AI 修复；
- 撤销本次结果；
- 导出结果。

参数面板只暴露真正需要调节的参数：模式、Mask 宽度、Denoising strength、Steps、Seed 和 Prompt。模型路径、缓存目录和高级显存参数放到设置页或配置文件中。

## 7. 开发里程碑

### 第 1 周：先跑通无 GUI 的基础链路

- 准备 30 组测试图片，覆盖砖墙、瓷砖、木纹、自然纹理和光照变化。
- 完成图片读取和统一坐标系。
- 完成 SIFT + RANSAC 配准。
- 完成 Warp 和传统融合。
- 输出每一步的中间图片，便于调试。

验收：命令行可以对一组图片生成初始拼接结果，并能保存匹配点和单应矩阵。

### 第 2 周：完成 Mask 和局部 Patch

- 完成重叠区域计算。
- 完成自动接缝或中心线 fallback。
- 完成 Mask 膨胀、裁剪和坐标映射。
- 完成 CLI 版 Patch 提取和贴回。

验收：给定一组图片和 Mask，可以准确提取 Patch，处理后能贴回原始高清画布，尺寸和坐标不漂移。

### 第 3 周：接入 Diffusers

- 下载并固定 SD 1.5 Inpainting 模型版本。
- 完成 FP16 推理。
- 完成 seed、steps、strength 和 prompt 配置。
- 完成 OOM 捕获和显存清理。
- 用 512×512 Patch 跑通端到端结果。

验收：至少 10 组测试图片能够稳定生成结果，失败时有明确错误提示，不会导致界面卡死。

### 第 4 周：完成 PyQt 界面

- 完成图片导入、画布缩放和平移。
- 完成自动配准按钮。
- 完成 Mask 画笔和橡皮擦。
- 完成后台推理线程和进度显示。
- 完成三结果对比和导出。

验收：不使用命令行，用户可以从导入图片到导出结果完整操作一次。

### 第 5 周：质量和性能优化

- 加入 LAB 颜色匹配。
- 调整软 Mask 和边缘融合。
- 对比普通 Inpainting、Canny 和 Tile。
- 加入 LCM 预览模式。
- 测试 8GB 显存设备的峰值和耗时。

验收：典型复杂纹理案例中，AI 结果比传统融合结果更连续，并且没有大面积颜色跳变。

### 第 6 周：整理、打包和结题材料

- 固定依赖版本。
- 增加日志、配置导入导出和异常处理。
- 准备演示案例和失败案例。
- 生成基线对比图和指标表。
- 打包成可运行版本或提供一键启动脚本。

## 8. 两人分工

### 郑子焌

- 测试数据集和目录规范；
- 传统算法基线和结果记录；
- Mask 编辑器和 PyQt 主界面；
- 推理进度、结果对比和导出；
- 用户操作说明和演示材料。

### 游航

- SIFT、LoFTR、RANSAC 配准；
- Warp、重叠区域和接缝算法；
- Diffusers Inpainting 推理；
- ControlNet、LCM 和显存优化；
- 算法参数和客观指标测试。

两人共同维护 `pipeline.py` 的输入输出协议，避免算法代码和界面代码互相嵌套。

## 9. 验收指标

MVP 不用追求论文级全面指标，先确保可运行、可比较、可复现。

### 功能指标

- 支持两张图片导入和高清导出；
- 自动配准失败时能提示并允许重新选择；
- Mask 可以手动添加和删除；
- 推理期间界面保持响应；
- 支持固定随机种子复现相同结果；
- 每次运行保存参数、耗时和错误日志。

### 性能目标

- 512×512 局部 Patch 在 8GB 显存设备上不发生 OOM；
- 预览模式目标耗时不超过 30 秒；
- 质量模式目标耗时不超过 90 秒；
- 模型只加载一次，重复生成不重复初始化。

### 效果目标

- 至少准备 30 组有明确重叠区域的测试图片；
- 至少覆盖三类复杂纹理；
- 与线性融合、泊松融合进行对比；
- 至少 70% 的典型案例中，人工盲评认为 AI 结果优于传统融合；
- 记录所有明显失败案例，不删除失败结果。

## 10. 主要风险和降级方案

### LoFTR 环境难以安装

先使用 SIFT + RANSAC 完成 MVP，LoFTR 作为增强模式。这样不会因为一个研究代码仓库的依赖问题阻塞整个项目。

### 扩散模型产生幻觉

降低 Denoising strength，缩小重绘区域，保留用户手动 Mask，并加入 Canny 或 Tile 控制。对于无法可靠修复的区域，允许用户退回传统融合结果。

### 8GB 显存 OOM

固定 512×512 Patch，使用 FP16、VAE slicing、attention slicing，并在每次推理结束后释放中间 Tensor。必要时关闭 ControlNet，或者提供 CPU offload 选项。

### 结果有色差

先做 LAB 颜色匹配，再做软 Mask 融合。颜色修正幅度设置上限，并在界面中保留原始结果供比较。

### 大图分块出现重复纹理

MVP 只处理一个接缝 Patch，不做整张图片的密集分块。后续确实需要大范围修复时，再设计重叠窗口和边界一致性策略。

### 模型下载和许可证问题

代码仓库许可证与模型权重许可证分别记录。模型不直接提交到 Git 仓库，程序通过本地模型目录或首次启动时的明确下载步骤加载，并在项目文档中保留来源和版本。

## 11. 第一阶段马上要做的事情

开工时不要先写完整 GUI，先完成一个命令行纵向切片：

```text
input_a.jpg + input_b.jpg
        ↓
registration
        ↓
warped_preview.png
        ↓
seam_mask.png
        ↓
inpaint_patch.png
        ↓
final_result.png
```

第一批代码只需要实现以下三个入口：

```python
register(image_a, image_b) -> RegistrationResult
generate_mask(registration_result) -> MaskResult
inpaint_and_blend(registration_result, mask_result, config) -> StitchResult
```

这条链路跑通后，再把它接进 PyQt。这样每个算法问题都可以脱离界面单独复现，后续调参和写实验报告也会容易很多。

## 12. MVP 最终交付物

- 一个可以启动的桌面程序；
- 一条可独立运行的命令行推理链路；
- 30 组测试数据及结果目录；
- 传统融合与 AI 融合的对比图；
- 参数和运行日志；
- 模型、代码和第三方许可证清单；
- 安装说明和用户操作说明；
- 结题报告所需的实验表格、失败案例和演示视频。

