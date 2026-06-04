# 代码阅读笔记：Colorful Image Colorization 官方 Demo

本文档主要阅读 `richzhang/colorization` 中与 demo 运行直接相关的代码。阅读重点不是逐行解释所有网络层，而是理解一张灰度图像如何经过预处理、模型推理和后处理，最终得到彩色化结果。

本次主要关注以下文件：

- `demo_release.py`：demo 入口，负责读入图像、加载模型、执行推理和保存结果。
- `colorizers/eccv16.py`：ECCV16 彩色化模型结构和预训练权重加载。
- `colorizers/base_color.py`：L 通道和 ab 通道的归一化与反归一化。
- `colorizers/util.py`：图像读取、Lab 色彩空间转换、预处理和后处理。

## 一、代码阅读范围

关键代码位置如下：

```text
demo_release.py
colorizers/eccv16.py
colorizers/base_color.py
colorizers/util.py
```

这几个文件基本覆盖了官方 demo 的完整流程。`demo_release.py` 像一个总控程序，负责把图像送进模型并保存输出；`eccv16.py` 定义真正进行彩色预测的神经网络；`base_color.py` 处理数值范围；`util.py` 负责图像格式转换。理解这几部分后，就可以知道 demo 为什么只输入灰度信息，却能输出彩色图像。

## 二、`demo_release.py` 的整体流程

关键代码片段:

```python
colorizer_eccv16 = eccv16(pretrained=True).eval()
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
out_img_eccv16 = postprocess_tens(
    tens_l_orig,
    colorizer_eccv16(tens_l_rs).cpu()
)
plt.imsave('%s_eccv16.png' % opt.save_prefix, out_img_eccv16)
```

这段代码可以看作五步。第一步加载 ECCV16 彩色化模型，并设置为测试模式。第二步读取输入图像。第三步把图像转换成模型需要的 L 通道张量。第四步把 L 通道送入模型，让模型预测颜色信息。第五步把结果保存为 PNG 图像。

这里的 `eval()` 表示模型进入推理状态，不再像训练时那样更新参数。`preprocess_img()` 和 `postprocess_tens()` 是整个 demo 中很关键的两个函数，前者负责把图像变成模型输入，后者负责把模型输出重新变成可显示的 RGB 彩色图。

## 三、为什么使用 Lab 色彩空间

关键代码片段：

```python
img_lab_orig = color.rgb2lab(img_rgb_orig)
img_l_orig = img_lab_orig[:, :, 0]
```

Lab 色彩空间的好处是把“亮度”和“颜色”分开。L 通道表示亮度，主要包含图像的明暗、轮廓和结构；a、b 通道表示颜色，主要描述颜色偏向和颜色强度。

图像彩色化的输入通常是一张灰度图，而灰度图本质上保留的是亮度信息。因此，用 Lab 空间来做彩色化比较自然：已知 L 通道，让模型去预测缺失的 a、b 通道。这样问题就变成了“根据图像结构和语义推测颜色”，而不是直接在 RGB 三个通道中同时猜亮度和颜色。

需要注意的是，灰度图不能唯一决定真实颜色。例如同样亮度的一块区域，可能是真实场景中的绿色树林，也可能是黄色草地或棕色岩石。因此彩色化不是精确恢复真实颜色，而是根据训练数据和图像内容生成一种合理的颜色结果。

## 四、图像预处理：从 RGB 图像得到 L 通道

关键代码片段：

```python
img_rgb_rs = resize_img(img_rgb_orig, HW=HW)
img_lab_orig = color.rgb2lab(img_rgb_orig)
img_lab_rs = color.rgb2lab(img_rgb_rs)

img_l_orig = img_lab_orig[:, :, 0]
img_l_rs = img_lab_rs[:, :, 0]

tens_orig_l = torch.Tensor(img_l_orig)[None, None, :, :]
tens_rs_l = torch.Tensor(img_l_rs)[None, None, :, :]
```

`preprocess_img()` 同时保留了两个 L 通道。`tens_l_orig` 来自原始尺寸图像，用于最后恢复到原图大小；`tens_l_rs` 来自缩放到 `256 x 256` 的图像，用于输入神经网络。

这里的 `[None, None, :, :]` 是在数组前面增加两个维度。转换后，模型输入可以理解为 `1 x 1 x 256 x 256`：第一个 `1` 表示一次输入一张图，第二个 `1` 表示只有一个 L 通道，后面的 `256 x 256` 是图像尺寸。

这一步说明 ECCV16 模型并不是直接输入 RGB 图像，而是只输入亮度通道。也就是说，模型看到的是图像的灰度结构，然后根据这些结构预测颜色。

## 五、预训练 ECCV16 模型的加载

关键代码片段：

```python
def eccv16(pretrained=True):
    model = ECCVGenerator()
    if pretrained:
        import torch.utils.model_zoo as model_zoo
        model.load_state_dict(model_zoo.load_url(
            'https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',
            map_location='cpu',
            check_hash=True
        ))
    return model
```

`ECCVGenerator()` 会先建立模型结构，但此时模型还没有学到有效的彩色化能力。真正让模型能够工作的，是后面加载的预训练权重文件。

`model_zoo.load_url()` 会从官方地址下载 `.pth` 权重文件。`load_state_dict()` 再把这些参数填入模型中。`map_location='cpu'` 表示即使没有 GPU，也可以在 CPU 上加载模型。`check_hash=True` 用于检查下载文件是否和预期一致，减少权重文件损坏或下载错误的风险。

因此，本次实验并没有重新训练模型，而是直接使用作者已经训练好的 ECCV16 模型进行推理。

## 六、模型输入、输出与 ab 通道预测

关键代码片段：

```python
def forward(self, input_l):
    conv1_2 = self.model1(self.normalize_l(input_l))
    ...
    conv8_3 = self.model8(conv7_3)
    out_reg = self.model_out(self.softmax(conv8_3))
    return self.unnormalize_ab(self.upsample4(out_reg))
```

`forward()` 是模型真正执行推理的函数。输入 `input_l` 是预处理得到的 L 通道，也就是灰度图的亮度信息。模型首先调用 `normalize_l()` 对 L 通道进行归一化，使数值范围更适合神经网络处理。

ECCV16 网络中间由多层卷积、ReLU 和归一化层组成。这些层的作用可以理解为逐步提取图像特征：浅层更关注边缘和局部纹理，深层能结合更大范围的结构信息。对于山、云、河流和树林这类图像，模型需要根据局部纹理和整体场景判断可能的颜色。

代码中的 `conv8_3` 有 313 个通道，对应论文方法中离散化后的颜色类别。也就是说，模型先不是直接输出两个连续颜色通道，而是先预测每个位置属于哪些颜色类别的概率分布。`softmax` 用来把这些类别响应转换成概率形式。

随后，`model_out` 把 313 个颜色类别的分布转换成 2 个通道，也就是最终需要的 a、b 颜色信息。最后 `upsample4` 将预测结果上采样到更大的空间尺寸，`unnormalize_ab()` 再把 ab 通道从归一化范围还原到 Lab 空间中的数值范围。

## 七、`base_color.py` 中的归一化处理

关键代码片段：

```python
self.l_cent = 50.
self.l_norm = 100.
self.ab_norm = 110.

def normalize_l(self, in_l):
    return (in_l - self.l_cent) / self.l_norm

def unnormalize_ab(self, in_ab):
    return in_ab * self.ab_norm
```

Lab 空间中的 L 通道大致表示亮度，范围通常在 0 到 100 之间。如果直接把这个范围输入网络，数值中心不在 0 附近。代码中先减去 50，再除以 100，使 L 通道大致落在较小范围内，这样更符合神经网络常见的输入习惯。

ab 通道表示颜色，数值范围也需要进行缩放。模型内部输出的是归一化后的颜色结果，最后通过 `unnormalize_ab()` 乘回 `ab_norm`，把它还原成后处理时可以使用的 Lab 颜色通道。

这一步本身不改变图像语义，但能让模型输入输出的数值更稳定。

## 八、后处理：把 L 和 ab 合成为 RGB 彩色图

关键代码片段：

```python
if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
    out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
else:
    out_ab_orig = out_ab

out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0, ...].transpose((1,2,0)))
```

模型是在 `256 x 256` 尺寸上预测颜色的，但原图可能比这个尺寸大。本次输入图像尺寸为 `3000 x 2402`，所以预测出的 ab 通道需要先通过双线性插值放大到原图尺寸。

随后，代码把原图尺寸的 L 通道和预测得到的 ab 通道拼接起来，重新组成一张 Lab 图像。这里保留原始 L 通道很重要，因为它能尽量保留原图的亮度、边缘和细节；模型主要负责补充颜色，而不是重新生成整张图像的明暗结构。

最后，`color.lab2rgb()` 把 Lab 图像转换回 RGB 图像。RGB 图像就是平时显示器和图片文件常用的彩色格式，因此可以直接用 `plt.imsave()` 保存。

## 九、本次实验中的理解

关键代码片段：

```python
img_bw = postprocess_tens(
    tens_l_orig,
    torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1)
)
out_img_eccv16 = postprocess_tens(
    tens_l_orig,
    colorizer_eccv16(tens_l_rs).cpu()
)
```

第一段生成的是输入灰度图的显示结果。它把 ab 通道设为 0，相当于只保留亮度，不加入颜色。第二段生成的是 ECCV16 的彩色化结果，它把模型预测出的 ab 通道和原始 L 通道合成起来。

从这个流程可以看出，彩色化任务的核心不是改变图像结构，而是根据 L 通道提供的结构信息预测颜色。对于本次山景图像，模型能在天空、云层、树林和河流区域生成一定颜色，使图像比灰度输入更接近自然彩色图。但颜色仍然是预测结果，不一定等于真实拍摄时的颜色。例如云层可能偏暖或偏紫，河流和树林也可能出现偏绿或偏黄的倾向。

官方 demo 同时运行了 ECCV16 和 SIGGRAPH17 两个模型。本次代码分析重点放在 ECCV16，因为它对应所选论文 *Colorful Image Colorization*。SIGGRAPH17 的输出可以作为补充对比。
