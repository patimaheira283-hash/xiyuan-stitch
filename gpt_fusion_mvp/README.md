# GPT 直接融合 MVP

这个实验把两张原图直接交给云端模型融合。不会运行原项目的 LoFTR、RANSAC、Stable Diffusion、ControlNet、接缝掩码或颜色后处理。

## 启动

双击项目根目录的 **启动GPT融合实验.bat**，或运行：

```powershell
.\.venv\Scripts\python.exe -m gpt_fusion_mvp serve --open
```

浏览器打开 `http://127.0.0.1:18767`。选择素材后可以检查请求、开始融合、下载原始结果，并在结果与参考图之间滑动对照。也可以上传自己的左右两张图片。

每次点击“开始融合”都会调用 CC Switch 当前 Codex 服务商，可能产生用量费用；程序不会自动重试失败的生成请求。

临时给同学试用的入口使用独立的邀请网关与 Cloudflare Tunnel，默认 24 小时、合计 30 次生成额度。可双击 `关闭临时公网访问.bat` 提前关闭。详见 `docs/临时公网试用.md`。

## 实际链路

```text
本机读取 CC Switch 当前 Codex Provider 的地址和 key
  → POST /responses，主模型 gpt-6-astra
  → 两个 input_image + 融合提示词
  → image_generation 工具（请求 gpt-image-2.5-sunburst）
  → 接收流中的 image_generation_call.result
  → 保存原始图片，再用完整参考图评估
```

选择 `gpt-6-astra` 的依据是本次实测：同一服务商上的 `gpt-5.5` 纯文本成功，但携带图片返回 404；`gpt-6-astra` 看图和调用图像工具都成功。没有修改 CC Switch 或 Codex 的全局配置。

`gpt-image-2.5-sunburst` 是请求值。第三方可能改写模型、质量或尺寸；本次返回的图像工具没有声明模型名。因此不能独立确认实际图像模型，也不能把请求的 `high` 当作实际执行档位。界面显示工具回报的质量与实测尺寸。

原来的 Images API 通道仍可用 `--route images` 显式测试：它通过 openai-codex-image-skills 的 helper 执行。默认 Responses 通道按照用户明确指定的 Codex 接口实现。

## 素材库

`data/gpt-fusion-library/manifest.json` 现在包含 **611 对样例**：来自 GitHub 的 601 对真实照片，以及原来的 10 对受控裁剪样例。网页可按来源、场景关键词、真实拍摄/有参考图/待检查筛选，并分页浏览。

新增素材来自 SPW、LPC、REW、GES-50、OpenPano，共下载 752 张原始照片；25 对真实照片已加入“推荐先试”。详细来源、下载范围、去重记录和使用方法见 `docs/GitHub拼接测试素材库.md`。下载与整理没有调用生成模型。

原来的五类受控素材是砖墙、咖啡与木桌、火箭与天空、草地、文字与几何校验板。每种有原始曝光和右图增亮 18% 两个版本。

照片来自本机 scikit-image 随包素材，保留作者、来源网址、许可证据、原始尺寸和 SHA-256。文字板为项目程序绘制的测试图，以 CC0 提供。

受控裁剪每组包含 `left.png`、`right.png`、`reference.png`。左右视野各覆盖完整图约 62.5% 的宽度，相互重叠约 40% 的自身视野。照片保持原生分辨率，经居中裁剪使完整参考图接近 3:2。**完整参考图不发送给模型。** 真实照片的 `reference` 为 null，保留原始文件及单独的网页缩略图；提交时只修正 EXIF 方向并做 RGB/PNG 转换，不进行预对齐。真实素材使用不预设左右方向的提示词。

只有受控样例计算完整参考图指标。真实样例通过目视检查接缝、文字、物体数量和结构保真；素材数量不等于已完成的模型测试次数。

重新运行 `materials` 会保留已导入的数据集。使用本地保存的仓库版本清单，可重下载或重建索引：

```powershell
.\.venv\Scripts\python.exe -m gpt_fusion_mvp.github_materials download
.\.venv\Scripts\python.exe -m gpt_fusion_mvp.github_materials index
```

## 命令行

```powershell
.\.venv\Scripts\python.exe -m gpt_fusion_mvp doctor
.\.venv\Scripts\python.exe -m gpt_fusion_mvp materials
.\.venv\Scripts\python.exe -m gpt_fusion_mvp run --case coffee-clean --dry-run
.\.venv\Scripts\python.exe -m gpt_fusion_mvp run --case brick-exposure
.\.venv\Scripts\python.exe -m gpt_fusion_mvp run --case text-grid-clean
.\.venv\Scripts\python.exe -m gpt_fusion_mvp run --left "left.jpg" --right "right.jpg"
```

`--mainline-model` 调整 Responses 主模型，`--model` 调整请求的图像模型。默认使用当前 CC Switch Provider；可用 `--provider-id` 显式选择配置库中的另一项。密钥不会写进项目文件、日志、命令参数或网页。

如果迁移环境，使用 Python 3.11/3.12 并安装本目录 `requirements.txt`。现有项目环境已具备依赖。

## 结果和评估

每次运行在 `outputs/gpt-fusion/<运行ID>/` 保存输入快照、提示词、脱敏请求预览、`run.json`、原始结果和参考图。原始生图另外保存在当前 `CODEX_HOME/output/imagegen` 下（未设置时是用户目录的 `.codex/output/imagegen`）。

- SSIM：输出缩放到参考图尺寸后计算；通常越接近 1 越相似，但不是接缝质量分数，也不是成功率。
- MAE：RGB 平均绝对像素差，范围 0–255；越低越接近原图。
- 重叠区外 MAE：检查原本无需修缝的区域是否也有变化。
- 原始输出不因评估而被缩放、修缝或颜色校正。
- 服务商返回的 usage 原样留档；由于未提供明确图像账单，不据此编造单图价格。

本次结果见 `docs/GPT直接融合首轮测试.md`。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_gpt_fusion.py -q
```

测试覆盖只读配置选择、密钥不泄露、两张原图正确提交且参考图不泄露、SSE 流解析、重复完成事件去重、成功后断流仍保留图片、无图片不误报成功、路径穿越防护与 dry-run 不调用 API。
