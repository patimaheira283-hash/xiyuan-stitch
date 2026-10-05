# 演示输入与出处

三个纹理案例来自 Poly Haven CC0 颜色贴图，来源和原文件哈希见 `data/neural-validation/sources.json`。这些输入是贴图裁剪与扰动，不是独立实拍场景。

`motion-run/` 使用 scikit-image 的 NASA astronaut 公共领域图片生成相反局部变形；来源：https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut 。其配准使用已知裁剪坐标，只用于局部结构修复验收。

在桌面打开 `motion-run/run.json` 后，可运行本机结构修复，比较默认保护与关闭保护的结果。示例中保存的旧运行环境是输入记录的历史信息；当前执行版本以新输出的验收记录为准。
