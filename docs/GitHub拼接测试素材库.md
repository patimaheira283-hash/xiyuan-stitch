# GitHub 图像拼接测试素材库

整理日期：2026-09-09。用途：把两张真实照片直接交给 GPT，观察接缝、内容和几何结构是否保留。

## 已经可以使用

打开 **http://127.0.0.1:18767/**，或双击项目根目录的 `启动GPT融合实验.bat`。素材页已加入搜索、来源筛选、测试范围和分页。

- 新下载 **752 张原始照片**，整理成 **601 对真实拍摄样例**。
- 保留原来的 **10 对受控裁剪样例**，网页合计 **611 对**。
- 从五个来源中各挑 5 对，共 **25 对“推荐先试”**；网页还会显示咖啡这一对旧基准。
- 原始照片合计 823,397,065 字节（约 823 MB），包含来源备份、下载压缩包、缩略图和索引的素材目录约 0.94 GB。
- 原始文件按仓库目录保存，未裁剪、预对齐或修缝。网页使用较小的预览图，模型提交使用原始照片的 EXIF 方向修正及 RGB/PNG 转换版本。

这次完成的是素材准备和无费用请求检查；没有用这 601 对样例批量调用生成服务。

应用验证：现有 9 项测试通过；五个新来源共检查 20 个原图/缩略图 HTTP 请求；611 对索引路径全部有效；重建基础素材不丢失导入数据；已在网页验证来源筛选、搜索、分页和一次无参考图请求检查。记录保存在 `outputs/gpt-fusion/dataset-validation.json`。

## 下载来源与实际数量

| GitHub 来源 | 下载范围 | 原始照片 | 原始场景目录 | 去重后保留测试对 | 适合观察 |
|---|---|---:|---:|---:|---|
| [SPW / Single-perspective-warps](https://github.com/tlliao/Single-perspective-warps) | `TwoImage/Images/` 全部双图目录 | 86 | 43 | 43 | 建筑立面、直线、透视关系 |
| [LPC / Image-Stitching](https://github.com/dut-media-lab/Image-Stitching) | `Imgs/` 每个场景的原始双图，排除 `results/` | 108 | 54 | 54 | 近景与远景视差、室内架子、街景文字 |
| [REW / Robust_Elastic_Warping](https://github.com/gain2217/Robust_Elastic_Warping) | `two_views/images/` 全部双图目录 | 72 | 36 | 12 | 铁轨、栅栏、路口、室内柜体 |
| [GES-50 / GES-GSP-Stitching](https://github.com/flowerDuo/GES-GSP-Stitching) | `Dataset/` 全部 50 组原图及作者配对表 | 340 | 50 | 361 | 多视图、商场、建筑、花园、室内外结构 |
| [OpenPano](https://github.com/ppwwyyxx/OpenPano/releases/tag/0.1) | Release 0.1 的 `example-data.tgz` 完整输入照片 | 146 | 8 | 131 | 全景序列、人物不同姿态、航拍、近景花朵 |
| **合计** | | **752** | **191** | **601** | |

REW 的 36 对中有 24 对与先导入的来源像素完全相同，因此其新增数量是 12。GES 中也存在重复收录。191 是上游目录数，601 是输入对数，都不等于独立场景数。

检索入口是 [Image-Stitching-Dataset 汇总仓库](https://github.com/visionxiang/Image-Stitching-Dataset)。实际下载来自上表的作者仓库和 Release，没有把汇总页的论文展示图片当测试输入。

## 怎样得到这些配对

1. **作者原本提供两张图的目录**：直接组成一对，不拿该目录中的算法输出作输入或真值。
2. **GES 多图序列**：按作者 `STITCH-GRAPH.txt` 的连接关系配对；图片索引按文件名排序，保留对应文件名和连接表。上游代码使用目录枚举，未显式排序，因此还做了重叠合理性检查。
3. **OpenPano 序列**：按文件名相邻关系提出候选，再通过局部特征匹配检查共同视野。完整多图序列仍保存在原始目录，可用于以后测试多图融合。

匹配检查只用于整理素材，不会给 GPT 提供对齐图、匹配点、单应矩阵或接缝掩码。它也不是拼接质量评分。

## 校验与去重

- 752 张原图都已完整解码；GitHub 文件按固定提交下载，逐文件核对 Git blob SHA-1，并记录 SHA-256、尺寸和文件大小。
- 752 张文件对应 **692 张不同的解码图像**；不同仓库的相同原图仍保留各自来源副本。
- 共构造 641 个候选配对，移除 **33 个像素完全相同的重复配对**，输入先后反转也视为重复。
- **7 个 OpenPano 相邻候选**的共同视野证据不足，没有放入可直接选用的列表；原图与候选记录仍保留。
- **9 个作者配对**自动匹配支持不足，保留并标成“配对待目视检查”。低纹理、大视差等也可能使传统匹配失败，所以不直接删除。
- 另外标记 **7 个跨仓库近似重复候选**，可能是同场景图像的压缩或缩放版本，没有冒充完全去重；推荐集避开了这些候选。
- GES 的 `GES-05_PetrolBuilding-1` 只有 3 张图，作者表中却有索引 `2 → 3`。这条越界连接被排除，其余合法连接保留。没有修改原始配对表。

自动检查使用 ORB 特征和 RANSAC 验证是否有足够且分布合理的对应点，不能证明“物理上完全正确的配对”。推荐 25 对已查看缩略总览；剩余素材未逐张做人工精细标注。标签中的“室内线索”“植被线索”来自目录名，研究方向标签来自论文主题。

所有导入素材都标为 `upstream_unspecified_local_evaluation`：这是本地测试用途，没有声称它们属于论文官方训练集或测试集。以后若用于训练和评估，应按场景及重复关联分组，避免同场景泄漏。

## 推荐如何开始

先在“推荐先试”里挑几对，逐张看结果：

| 样例 | 主要看什么 |
|---|---|
| LPC / Potberry | 商店文字有没有改写，门窗和花盆有没有重复 |
| LPC / DFW_shelf | 近处架子的边线能否接上，是否弯曲或多出隔板 |
| REW / APAP-railtracks | 铁轨的数量、方向和延伸关系是否保持 |
| GES / WandaIndoor | 商场楼层、栏杆与远近结构是否连贯 |
| OpenPano / myself | 同一人物不同姿态如何处理，模型有没有擅自新增或删除内容 |
| OpenPano / uav | 航拍建筑、道路与地面纹理是否被重画 |

“仅检查请求”不会调用生成服务；“开始融合”才会产生用量。真实拍摄样例没有唯一正确的完整全景图，不计算参考图 SSIM。部分上游文件名叫 `reference`，在这些目录里只是第二张拍摄视图，不代表完整真值。

## 文件位置

素材根目录：`D:\aa曦源项目\data\gpt-fusion-library`

| 文件或目录 | 内容 |
|---|---|
| `manifest.json` | 网页/CLI 共用的 611 对样例索引、标签、提示词和来源 |
| `originals/` | 按来源保留的原始照片、GES 连接表、OpenPano 压缩包 |
| `previews/` | 只供网页使用的缩略图 |
| `smoke-suite.json` | 推荐 25 对的准确 case ID 与检查项目 |
| `recommended-contact-sheet.jpg` | 推荐 25 对的输入总览 |
| `provenance/images.json` | 原图文件、尺寸、文件校验值和解码像素校验值 |
| `provenance/sequences.json` | 完整序列及候选配对关系 |
| `provenance/pair-audit.json` | 重复关联、未采用候选、待检查依据、上游表格错误 |
| `provenance/*__*.json` | 仓库信息与固定提交的文件树快照 |
| `provenance/<作者__仓库>/` | 下载当时的 README、已有许可证和必要的索引规则依据 |

重复运行 `python -m gpt_fusion_mvp materials` 会保留已导入的数据集。重下载使用 `python -m gpt_fusion_mvp.github_materials download`；重建配对索引用 `python -m gpt_fusion_mvp.github_materials index`。两个命令都不调用图像生成服务。

## 出处与使用说明

SPW、LPC、GES 的 README 明确要求使用代码或数据时引用论文；REW 保留论文名称及仓库出处。这四个仓库未提供单独的数据许可证，不能据“GitHub 可下载”推定为 CC0 或可任意再分发。OpenPano 的 MIT 许可证针对仓库软件，示例照片没有单独声明。当前素材作为本地学习实验存放，相关 README 和许可证原文均已保存。

论文对应关系：SPW 为 *Single-Perspective Warps in Natural Image Stitching*；LPC 为 *Leveraging Line-point Consistence to Preserve Structures for Wide Parallax Image Stitching*；REW 为 *Parallax-Tolerant Image Stitching Based on Robust Elastic Warping*；GES 为 *Geometric Structure Preserving Warp for Natural Image Stitching*。GES README 还列出其从其他论文继承的素材出处。

## 找到但本轮未导入

- [UDIS-D](https://github.com/nie-lang/UnsupervisedDeepImageStitching)、[UDIS2](https://github.com/nie-lang/UDIS2)：完整数据从作者提供的 Google Drive/百度云下载；GitHub 内主要是代码和展示结果。本轮先整理了直接可下载的五套素材，没有把 UDIS 的对齐后图像或生成结果当作原始测试集。
- [ColorConsistency](https://github.com/MenghanXia/ColorConsistency)：仓库示例已通过拼接算法对齐，适合研究颜色一致性；本轮优先收集未经预对齐的拍摄照片。
