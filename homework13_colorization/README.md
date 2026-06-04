# 数字图像处理第 13 次作业：智能图像彩色化处理

## 一、项目背景

本项目用于完成数字图像处理课程第 13 次作业，主题为“智能图像彩色化处理”。图像彩色化是指根据灰度图像自动生成合理的彩色图像，它既涉及图像表示、颜色空间转换等传统图像处理内容，也与深度学习中的特征提取、分类预测和生成结果评价有关。

本次作业计划阅读的论文为：

Richard Zhang, Phillip Isola, Alexei A. Efros. *Colorful Image Colorization*. ECCV 2016.

该论文提出了一种基于深度学习的自动图像彩色化方法，将颜色预测问题建模为分类任务，并通过重新加权方式缓解颜色分布不均衡问题，是图像彩色化领域中具有代表性的工作之一。

## 二、作业目标

本项目的主要目标包括：

1. 阅读并理解一篇图像彩色化相关文献。
2. 整理论文的研究背景、方法思路、实验结果和个人理解。
3. 撰写一份结构清晰的阅读报告。
4. 选做部分：尝试下载并运行相关彩色化算法代码，阅读关键代码，展示实验结果。

## 三、文件结构

```text
homework13_colorization/
├── README.md
├── report.md
├── references.md
├── notes/
│   └── code_analysis.md
├── data/
│   └── input/
├── results/
└── third_party/
```

各目录和文件说明如下：

- `README.md`：项目说明文件，记录作业背景、目标、目录结构和后续计划。
- `report.md`：阅读报告正文，后续用于整理论文内容和个人分析。
- `references.md`：参考文献列表，用于记录论文、代码仓库和相关资料。
- `notes/code_analysis.md`：代码阅读笔记，后续用于记录选做实验中的关键代码分析。
- `data/input/`：存放后续实验使用的输入图像。
- `results/`：存放算法运行后的彩色化结果和对比图。
- `third_party/`：存放后续下载或整理的第三方算法代码。

## 四、后续运行代码的计划

后续如果进行选做实验，计划按以下步骤完成：

1. 查找与论文 *Colorful Image Colorization* 对应的公开代码或复现项目。
2. 将第三方代码放入 `third_party/` 目录，并记录代码来源。
3. 准备若干灰度或低饱和度测试图像，放入 `data/input/`。
4. 阅读模型加载、图像预处理、网络推理和结果保存等关键代码。
5. 运行彩色化程序，将输出图像保存到 `results/`。
6. 在 `notes/code_analysis.md` 中记录关键代码功能和运行过程。
7. 在 `report.md` 中简要展示实验结果，并结合论文方法进行分析。

目前已完成一次官方 demo 运行，实验输出见本文档“运行结果”部分。完整阅读报告仍在后续整理中。

## 五、最终提交物说明

最终提交时，项目预计包含以下内容：

- 阅读报告：`report.md`
- 参考文献列表：`references.md`
- 代码阅读笔记：`notes/code_analysis.md`
- 实验输入图像：`data/input/`
- 实验输出结果：`results/`
- 第三方代码或代码来源说明：`third_party/`

如果完成选做实验，将在报告中补充算法运行结果和必要的截图说明。

## 六、运行结果

本项目已使用 `richzhang/colorization` 官方 PyTorch demo 对自选灰度图像进行一次彩色化实验。主结果采用 ECCV 2016 论文对应的 ECCV16 模型输出。

### 1. 结果文件

实验结果保存在 `results/` 目录下，主要文件如下：

- 输入灰度图：`results/input_image_gray.png`
- 输出彩色图：`results/output_colorized.png`
- 输入输出对比图：`results/comparison.png`
- 终端运行成功截图：`results/terminal_success.png`

其中，`comparison.png` 将输入灰度图和输出彩色图并排显示，适合放入报告正文中进行结果展示。

### 2. 运行命令

复制输入图像到实验输入目录：

```powershell
Copy-Item -LiteralPath .\input_image.jpg -Destination .\homework13_colorization\data\input\input_image.jpg -Force
```

运行官方 demo：

```powershell
cd .\homework13_colorization\third_party\colorization
$env:MPLBACKEND='Agg'
.\.venv\Scripts\python.exe demo_release.py -i ..\..\data\input\input_image.jpg -o ..\..\results\input_image_colorized
```

保存 ECCV16 输出作为主彩色化结果：

```powershell
cd ..\..\..
Copy-Item -LiteralPath .\homework13_colorization\results\input_image_colorized_eccv16.png -Destination .\homework13_colorization\results\output_colorized.png -Force
```

生成灰度图和并排对比图：

```powershell
python -c "from pathlib import Path; from PIL import Image, ImageOps; root=Path('homework13_colorization'); inp=root/'data/input/input_image.jpg'; gray_path=root/'results/input_image_gray.png'; out_path=root/'results/output_colorized.png'; cmp_path=root/'results/comparison.png'; gray=ImageOps.grayscale(Image.open(inp)); gray.save(gray_path); left=gray.convert('RGB'); right=Image.open(out_path).convert('RGB'); target_h=900; fit=lambda im: im.resize((round(im.size[0]*target_h/im.size[1]), target_h)); left=fit(left); right=fit(right); canvas=Image.new('RGB',(left.width+right.width,target_h),(255,255,255)); canvas.paste(left,(0,0)); canvas.paste(right,(left.width,0)); canvas.save(cmp_path)"
```

### 3. 展示材料建议

报告中建议优先展示 `results/comparison.png`，因为它能直接对比输入灰度图和输出彩色图。随后可以展示 `results/terminal_success.png`，用于说明程序已经成功运行并生成结果。如果版面允许，也可以单独展示 `results/input_image_gray.png` 和 `results/output_colorized.png`。

官方 demo 同时生成了 `results/input_image_colorized_siggraph17.png`。该图可作为补充观察结果，但本次报告的主结果仍以 ECCV16 模型输出为准。
