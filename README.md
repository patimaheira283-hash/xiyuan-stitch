# 曦源图像拼接 · 0.5.1

Windows 桌面图像拼接研究原型。支持 SIFT / LoFTR 配准、接缝画笔、羽化 / 泊松对照、清晰融合、RAFT 结构修复与原尺寸 PNG 导出；扩散、ControlNet 和 LCM 通过可选 GPU 实验运行。

## 当前冻结版本

**0.5.1** 是当前源码与软件交付基线，Git 标签为 `v0.5.1`。唯一版本定义在 `xiyuan_mvp/__init__.py`，wheel、窗口和验收记录读取同一版本。

[GitHub Releases](https://github.com/patimaheira283-hash/xiyuan-stitch/releases) 提供源码 ZIP、wheel、Windows CPU AI 免安装包、验收记录和 SHA256 校验清单。该仓库为私有仓库，需要相应账号权限。

本次冻结统一已有实现和发布证据；历史 0.3 / 0.4 / 0.5 实验成绩仍属于原快照，不作为 0.5.1 的新质量评估。真人外机验收、人工盲评和原生 8 GB 显卡验证仍待完成。

## 使用免安装版

解压 `xiyuan-windows-ai-0.5.1.zip`，双击 `XiyuanAI/XiyuanAI.exe`，保留整个文件夹。包内含 LoFTR / RAFT CPU 运行依赖和权重。

导入相邻重叠图片 → 自动配准与清晰融合 → 编辑接缝区域 → 选择本机结构修复 → 对比 → 导出 PNG。两图本机结构修复不需要 NVIDIA 显卡。

## 从源码运行

需要 Windows x64、Python 3.12：

```powershell
./scripts/Setup.ps1 -Neural
.venv/Scripts/python.exe -m xiyuan_mvp.gui
```

只运行 SIFT 和传统融合时执行 `./scripts/Setup.ps1`。源码环境的首次依赖安装、模型下载需要网络；离线使用请提前准备模型或使用免安装包。

```powershell
.venv/Scripts/python.exe -m xiyuan_mvp.cli --version
.venv/Scripts/python.exe -m xiyuan_mvp.cli data/demo_a.png data/demo_b.png --output-dir outputs/demo
```

每次使用新的输出目录。运行记录保存输入、参数、Mask、配准矩阵、模型信息、耗时与图像 SHA256。

## 复现测试

```powershell
./scripts/Setup.ps1 -Test
$env:QT_QPA_PLATFORM = 'offscreen'
.venv/Scripts/python.exe -m pytest -q
```

GitHub Actions 从干净检出安装测试依赖，运行同一套测试和 wheel 独立安装检查。单元测试中的模型替身只验证逻辑；真实 LoFTR / RAFT 由发布包离线验收单独检查。

## 文档与范围

- [操作说明](docs/delivery/使用说明.md)
- [复现与验收](docs/delivery/复现与验收.md)
- [0.5.1 发布说明](docs/releases/0.5.1.md)
- [冻结与发布流程](docs/releases/发布流程.md)
- [项目验收追踪](docs/项目交付验收追踪.md)
- [模型和第三方材料](docs/delivery/模型与第三方材料.md)
- [变更记录](CHANGELOG.md)

`gpt_fusion_mvp/` 是另行保留的云端融合实验代码，使用自己的依赖和账号配置，不包含在桌面 wheel 中。仓库不保存凭据、用户申请书、模型权重、大型照片库和历史运行输出。示例出处见 [examples/README.md](examples/README.md)。

真实照片的大视差、运动物体和弱纹理仍可能失败；保护策略会在没有确认收益时保留底图。代码测试通过不等于画质普遍优于传统算法。
