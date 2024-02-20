---
{"dg-publish":true,"permalink":"/200-Study/Computer Vision/Convolutions in Deep Learning/","noteIcon":""}
---


# Convolutions in Deep Learning

[A Comprehensive Introduction to Different Types of Convolutions in Deep Learning | by Kunlun Bai | Towards Data Science](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)

这篇文章总结了深度学习中常用的几种卷积，并给出详细解释。除了这篇文章，还有其他人关于这个主题的几篇好文章。请查看它们（列在参考资料中）。

## 1 卷积和交叉相关

卷积是一种广泛用于信号处理、图像处理和其他工程/科学领域的技术。在深度学习中，一种模型架构，卷积神经网络（CNN）就是以这种技术命名的。然而，深度学习中的卷积本质上是信号/图像处理中的互相关（Crosscorrelation）。这两种操作之间存在细微的区别。

在信号/图像处理中，卷积的定义如下：
$$
(f \ast g)(t) = \int_{- \infty}^{+ \infty}f(\tau)g(t - \tau)d \tau
$$

它被定义为一个函数反转和移位后两个函数的乘积的积分。以下可视化演示了这个过程。

![Convolution in signal processing](https://image-1256466424.cos.accelerate.myqcloud.com/202402172013141_2024_02_20_05.png)

在这里，函数 $g$ 就是一个滤波器，它被反转，然后沿着横轴移动。对于任一位置，我们可以计算 $f$ 和翻转的 $g$ 相交部分的面积，这个面积就是该位置处的卷积值。

交叉相关被称为两个函数之间的**滑动点积**或**滑动内积**。交叉相关中的滤波器没有被反转，而是直接进行滑动操作。$f$ 和 $g$ 相交部分的面积就是**交叉相关**。下图展示了卷积和交叉相关性之间的区别。

![Difference between convolution and cross-correlation in signal processing](https://image-1256466424.cos.accelerate.myqcloud.com/202402172200135_2024_02_22_48.png)

在深度学习中，卷积中的滤波器没有进行反转，主要进行元素乘法和加法。严格来说，这是交叉相关。但在深度学习中，人们习惯性地将其称为卷积。这没有问题，因为滤波器的权重是在训练过程中学到的。如果上例中的反转函数 $g$ 是原函数，那么训练后学习到的权重矩阵会进行反转。因此，在训练之前，无需像真正的卷积那样先反转滤波器。

## 2 深度学习中的卷积

卷积的目的是为了从输入中提取有用的特征。在图像处理中，可以选择许多不同的滤波器进行卷积。每种滤波器都有助于从输入图像中提取不同的方面或特征，例如水平特征、垂直特征、对角线边缘特征等。同样地，在卷积神经网络中，*通过使用滤波器卷积提取不同的特征，这些滤波器的权重是在训练过程中自动学习的*。然后将所有提取出来的特征“组合”起来以做出决策。

卷积有几个==优点==，如**权重共享和平移不变**。卷积还能**考虑像素的空间关系**。这对许多计算机视觉任务非常有帮助，因为这些任务通常涉及识别某些组件与其他组件具有特定空间关系的物体（例如，狗的身体通常与头部、四条腿和尾巴相连）。

### 2.1 单通道卷积

![Convolution for a single channel](https://image-1256466424.cos.accelerate.myqcloud.com/1_hKkrLnzObzGtn7oeV4QRmA_2024_02_22_44.gif)

在深度学习中，卷积就是逐元素的乘法和加法。对于一张只有一个通道的图像，卷积操作过程如上图所示。这个滤波器是一个 $3 \times 3$ 的矩阵，其中元素为 \[\[0, 1, 2], \[2, 2, 0\], \[0, 1, 2\]\]。该滤波器在输入数据上滑动，在每一个位置上进行逐元素的乘法和加法，获得一个结果值。因此，最终输出为一个 $3 \times 3$ 的矩阵。

> [!attention]
> 在这个例子中，$\text{stride} = 1$，$\text{pading} = 0$，这些概念将在下文的算术部分进行介绍。

### 2.2 多通道卷积

在许多应用中，我们处理的图像都具有多个通道。一个典型的例子是 RGB 图像。如下图所示，每个 RGB 通道表征了原始图像的不同方面。

![Different channels emphasize different aspects of the raw image](https://image-1256466424.cos.accelerate.myqcloud.com/202402172243908_2024_02_22_24.png)

另外一个多通道数据的例子是卷积神经网络中的层。一个卷积层通常由多个通道组成（一般有几百个通道）。每个通道描述了前面一层的不同特征。如何在不同深度的层之间进行转换？如何将深度为 $n$ 的层转换到深度为 $m$ 的下一层？

我们需要阐述几个术语：层 (Layer)、通道 (Channel)、特征图 (Feature map)、滤波器 (Filter) 和内核 (Kernel)。从层次的角度来看，层和滤波器的概念处于同一层次，而通道和内核则低一个层次。通道和特征图是一回事。一个层可以有多个通道（或特征图）：如果输入是 RGB 图像，则输入层有 3 个通道。“通道”通常用来描述“层”的结构。同样，“内核”也用来描述“滤波器”的结构。

![Difference between "layer"("filter") and "channel"("kernel")](https://image-1256466424.cos.accelerate.myqcloud.com/202402172254112_2024_02_22_23.png)

过滤器和内核之间的区别有点棘手。有时，这两个词被交替使用，可能会造成混淆。从本质上讲，这两个术语有着微妙的区别。“内核”是指权重的二维数组。而 “滤波器”是指由多个内核堆叠在一起的三维结构。对于二维滤波器，滤波器与内核相同。但对于三维滤波器和深度学习中的大多数卷积来说，滤波器是内核的集合。每个内核都是独一无二的，强调输入通道的不同方面。

根据这些概念，==多通道卷积的过程==如下。**每个内核应用于上一层的一个输入通道，生成一个输出通道。这是一个按内核分类的过程。我们对所有内核重复这一过程，生成多个通道。然后将每个通道相加，形成一个输出通道。**下面的插图应该能让我们更清楚地了解这一过程。

![The first step of 2D convolution for multi-channels: each of the kernels in the filter are applied to three channels in the input layer, separately.](https://image-1256466424.cos.accelerate.myqcloud.com/1_Emy_ai48XaOeGDgykLypPg_2024_02_23_57.gif)

然后将这三个通道相加（逐元素相加），形成一个单一通道（$3 \times 3 \times 1$）。该通道是输入层（$5 \times 5 \times 3$ 矩阵）与滤波器（$3 \times 3 \times 3$ 矩阵）卷积的结果。

我们可以把这一过程等同于在输入层中滑动三维滤波器矩阵。**请注意，输入层和滤波器具有相同的深度（通道数 = 内核数）。**三维滤波器只在图像的高度和宽度这两个方向上移动（这就是为什么这种操作被称为二维卷积，尽管三维滤波器是用来处理三维体积数据的）。在每个滑动位置，我们都要执行元素乘法和加法运算，最后得到一个数字。在下面的示例中，我们在水平方向的 5 个位置和垂直方向的 5 个位置进行滑动。总之，我们得到了一个输出通道。

![2D convolution](https://image-1256466424.cos.accelerate.myqcloud.com/202402172322321_2024_02_23_50.png)

现在我们可以看到如何在不同深度的层之间进行转换。假设输入层有 $D_{in}$ 个通道，而我们希望输出层有 $D_{out}$ 个通道。我们需要做的就是将 $D_{out}$ 个滤波器应用到输入层。每个滤波器都有 $D_{in}$ 个内核。每个滤波器提供一个输出通道。应用 $D_{out}$ 个滤波器后，我们就有了 $D_{out}$ 个通道，然后将它们堆叠在一起就形成了输出层。

![Standard 2D convolution](https://image-1256466424.cos.accelerate.myqcloud.com/202402172332824_2024_02_23_27.png)

## 3 3D 卷积

在上一节的最后一个图中，我们看到我们实际上是在对 3D 数据执行卷积。但通常情况下，我们仍然将该操作称为深度学习中的二维卷积。*它是对 3D 数据的 2D 卷积。过滤器深度与输入层深度相同。3D 滤波器仅在 2 个方向（图像的高度和宽度）上移动。这种操作的输出是 2D 图像（只有 1 个通道）。*

当然，有 3D 卷积。它们是二维卷积的推广。*在 3D 卷积中，滤波器深度小于输入层深度（内核大小<通道大小）。因此，3D 滤波器可以在所有 3 个方向（图像的高度、宽度、通道）上移动*。在每个位置，逐元素乘法和加法提供一个值。由于滤波器在 3D 空间中滑动，因此，输出值也排列在 3D 空间中，然后输出是 3D 数据。

![3D Convolution](https://image-1256466424.cos.accelerate.myqcloud.com/202402172350921_2024_02_23_20.png)

二维卷积编码二维空间中物体的空间关系，三维卷积可以描述三维空间中物体的空间关系。这种三维关系在某些应用中非常重要，例如在生物医学成像（如 CT 和 MRI）的三维分割/重建中，血管等物体在三维空间中蜿蜒曲折。

## 4 1x1 卷积

在上一节的 3D 卷积中谈到了深度方面的操作，现在我们来谈谈另一个有趣的操作，$1 \times 1$ 卷积。

你可能想要知道为什么 $1 \times 1$ 卷积很有用。我们是否只需将一个数字与输入层中的每个数字相乘？对于只有一个通道的层来说，这种操作是微不足道的。在这里，我们将每个元素乘以一个数字。

如果输入层有多个通道，情况就变得有趣了。下图说明了 $1 \times 1$ 卷积在输入层的维度为 $H \times W \times D$ 时的工作原理。输入层经过滤波器尺寸为 $1 \times 1 \times D$ 的 $1 \times 1$ 卷积后，输出通道的尺寸为 $H \times W \times 1$。如果我们应用 $N$ 个这样的 $1 \times 1$ 卷积，然后将结果串联起来，我们就可以得到一个维度为 $H \times W \times N$ 的输出层。

![1 x 1 convolution](https://image-1256466424.cos.accelerate.myqcloud.com/202402180016481_2024_02_00_16.png)

$1 \times 1$ 卷积最初是在 Network-in-network 论文中提出的。随后，它们在 Google Inception 论文中得到了广泛应用。$1 \times 1$ 卷积的几个优点：
- 降低维度，提高计算效率
- 有效的低维嵌入或特征池华
- 在卷积后再次应用非线性特性

前两个优点可以在上图中观察到。经过 $1 \times 1$ 卷积后，我们降低了深度方向的维度。假如原输入有 200 个通道，$1 \times 1$ 卷积会将这些通道 (特征) 嵌入到一个通道中。第三个优点是，在 $1 \times 1$ 卷积之后，可以添加非线性激活，如 ReLU。非线性可以让网络学习到更复杂的功能。

Google Inception 论文中描述了上述优点：

> [!cite]
> “One big problem with the above modules, at least in this naïve form, is that even a modest number of $5 \times 5$ convolutions can be prohibitively expensive on top of a convolutional layer with a large number of filters.
>
> " 上述模块的一个大问题是，即使是少量的 $5 \times 5$ 卷积，在具有大量滤波器的卷积层之上，其成本也会高得令人望而却步。
>
> This leads to the second idea of the proposed architecture: judiciously applying dimension reductions and projections wherever the computational requirements would increase too much otherwise. This is based on the success of embeddings: even low dimensional embeddings might contain a lot of information about a relatively large image patch… That is, $1 \times 1$ convolutions are used to compute reductions before the expensive $3 \times 3$ and $5 \times 5$ convolutions. Besides being used as reductions, they also include the use of rectified linear activation which makes them dual-purpose.”
>
> 这就引出了所提架构的第二个想法：在计算需求会增加太多的地方，明智地应用维度缩减和投影。这是基于嵌入式的成功经验：即使是低维嵌入式，也可能包含相对较大图像片段的大量信息……也就是说，在计算昂贵的 $3 \times 3$ 和 $5 \times 5$ 卷积之前，先用 $1 \times 1$ 卷积来计算降维。除了用作还原，它们还包括使用整流线性激活，这使得它们具有双重用途 "。

关于 $1 \times 1$ 卷积，Yann LeCun 提出了一个有趣的观点：“在卷积网络中，不存在‘全连接层’。只有具有 $1 \times 1$ 卷积核和全连接表的卷积层 "。

![Yann LeCun](https://image-1256466424.cos.accelerate.myqcloud.com/202402180029086_2024_02_00_14.png)

## 5 卷积算术

我们现在知道了如何理解卷积的深度。接下来，我们将讨论如何处理其他两个方向（高度和宽度）的卷积，以及重要的卷积运算。

首先要介绍一些术语：
- **内核大小 (kernel size)**：内核已在上一节中讨论过。内核大小定义了卷积的视场。
- **步长 (stride)**：它定义了内核在图像中滑动时的步长。步长为 1 表示内核在图像中逐个像素滑动。步长为 2 表示内核在图像中滑动时，每一步移动 2 个像素（即跳过 1 个像素）。我们可以使用步长（>=2）对图像进行降采样。
- **填充 (padding)**：填充定义了如何处理图像的边界。有填充的卷积（Tensorflow 中的“same”填充）将保持空间输出尺寸与输入图像相等，必要时在输入边界周围填充 0。另一方面，无填充卷积（Tensorflow 中的 " 有效 " 填充）只对输入图像的像素执行卷积，而不在输入边界周围添加 0，因此输出大小小于输入大小。

下面的插图描述的是使用内核大小为 3、步长为 1 和填充为 1 的二维卷积。

![](https://image-1256466424.cos.accelerate.myqcloud.com/1_d03OGSWsBqAKBTP2QSvi3g_2024_02_19_34.gif)

有一篇关于详细运算的优秀文章 [[1603.07285] A guide to convolution arithmetic for deep learning]( https://arxiv.org/abs/1603.07285 )。关于内核大小、步长和填充的不同组合的详细说明和示例，可以参考这篇文章。

一般情况下，对于大小为 $i$、内核大小为 $k$、填充为 $p$、步长为 $s$ 的输入图像，卷积的输出图像大小为 $o$：

$$
o = \lfloor \frac{i + 2p - s}{k} \rfloor + 1
$$

## 6 转置卷积 (反卷积)

在许多应用和许多网络架构中，我们经常需要进行与正常卷积相反方向的变换，即进行向上采样，例如生成高分辨率图像、将低维特征图映射到高维空间。在后面的例子中，语义分割首先在编码器中提取特征图，然后在解码器中恢复原始图像大小，这样就能对原始图像中的每个像素进行分类。

传统上，人们可以通过应用插值方案或手动创建规则来实现向上采样。而神经网络等现代架构则倾向于让网络本身自动学习适当的变换，而无需人工干预。为此，我们可以使用转置卷积。

**转置卷积**也被称为**反卷积、解卷积或文献中的分步卷积**。不过，值得注意的是，“解卷积”这个名称不太恰当，因为转置卷积并不是信号/图像处理中定义的真正的解卷积。从技术上讲，信号处理中的解卷积是反向卷积操作。这里的情况并非如此。因此，一些学者强烈反对将转置卷积称为解卷积。人们称其为解卷积主要是为了简单起见。稍后，我们将看到为什么将这种操作称为转置卷积是自然的，也是更恰当的。

转置卷积总是可以通过直接卷积来实现。例如，在下图中，我们使用 $3 \times 3$ 内核对 $2 \times 2$ 输入进行转置卷积，并使用单位步长填充了 $2 \times 2$ 的零边界。上采样输出的大小为 $4 \times 4$。

![](https://image-1256466424.cos.accelerate.myqcloud.com/1_KGrCz7aav02KoGuO6znO0w_2024_02_19_48.gif)

有趣的是，通过使用不同的填充和步长，我们可以将相同的 $2 \times 2$ 输入图像映射成不同大小的图像。下面，在相同的 $2 \times 2$ 输入（输入之间插入 1 个零）上应用转置卷积，使用单位步长填充 $2 \times 2$ 边框的零。现在的输出大小为 $5 \times 5$。

![](https://image-1256466424.cos.accelerate.myqcloud.com/1_Lpn4nag_KRMfGkx1k6bV-g_2024_02_19_35.gif)

在上述例子中观察转置卷积可以帮助我们建立一些直觉。但要推广它的应用，最好还是看看它是如何在计算机中通过矩阵乘法实现的。从这里，我们也能明白为什么“转置卷积”是一个合适的名称。

在卷积过程中，我们将 $C$ 定义为内核，$Large$ 定义为输入图像，$Small$ 定义为卷积后的输出图像。卷积（矩阵乘法）后，我们将大图像向下采样为小输出图像。矩阵乘法中卷积的实现过程如下：$C \times Large = Small$。

下面的示例展示了这种操作是如何进行的。它将输入转化为 $16 \times 1$ 矩阵，并将内核转化为 $4 \times 16$ 稀疏矩阵。然后在稀疏矩阵和扁平化输入之间进行矩阵乘法运算。然后，将得到的 $4 \times 1$ 矩阵转换回 $2 \times 2$ 输出。

![](https://image-1256466424.cos.accelerate.myqcloud.com/202402201943816_2024_02_19_56.png)

现在，如果我们将矩阵的转置 $C^{\top}$ 在等式两边进行多重运算，并利用矩阵与其转置矩阵相乘得到单位矩阵的性质，就可以得到如下公式 $C^{\top} \times Small = Large$，如下图所示。

![](https://image-1256466424.cos.accelerate.myqcloud.com/202402201950502_2024_02_19_39.png)

此处，**用于转置卷积的权重矩阵** $C^{\top}$ 不一定来自于原卷积矩阵 $C$ (通常不会如此恰巧)，但其形状和原卷积矩阵 $C$ 的转置相同。

转置卷积的一般运算可以阅读 [[1603.07285] A guide to convolution arithmetic for deep learning]( https://arxiv.org/abs/1603.07285 ) 获取更多内容。

### 6.1 棋盘伪影 (Checkerboard artifacts)

在使用转置卷积时，人们会发现一种令人不快的现象，那就是所谓的棋盘伪影。

![](https://image-1256466424.cos.accelerate.myqcloud.com/202402202016974_2024_02_20_26.png)

论文 [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/) 对这种行为进行了详尽的描述。请查看这篇文章了解更多详情。在此，我仅总结几个要点。

棋盘伪影的产生是由于转置卷积的 " 不均匀重叠 "。

在下图中，上层是输入层，下层是转置卷积后的输出层。在转置卷积过程中，一个较小的层被映射到一个较大的层上。

![](https://image-1256466424.cos.accelerate.myqcloud.com/202402202023567_2024_02_20_51.png)

在示例 (a) 中，步长为 1，滤波器大小为 2。如红色所示，输入的第一个像素映射到输出的第一个和第二个像素。如绿色所示，输入端的第二个像素映射到输出端的第二个和第三个像素。输出端的第二个像素同时接收来自输入端的第一个和第二个像素的信息。总的来说，输出中间部分的像素从输入端接收到的信息量是相同的。这里存在一个内核重叠的区域。在示例 (b) 中，当滤波器的大小增加到 3 时，接收到最多信息的中间部分就会缩小。但这并不是什么大问题，因为重叠部分仍然是均匀的。输出中心部分的像素从输入中接收到的信息量是相同的。

现在，对于下面的示例，我们更改步长为 2。在示例 (a) 中，滤波器大小为 2，输出上的所有像素从输入接收相同数量的信息。它们都从输入上的单个像素接收信息。这里没有转置卷积的重叠。

![](https://image-1256466424.cos.accelerate.myqcloud.com/202402202029647_2024_02_20_45.png)

如果我们将滤波器的大小改为 4 如示例 (b) ，那么均匀重叠的区域就会缩小。但是，我们仍然可以将输出的中心部分作为有效输出，每个像素都能从输入中接收到相同数量的信息。

不过，如果我们将滤波器大小分别改为 3 和 5 如示例 (c) 和 (d)，情况就变得有趣了。在这两种情况下，与相邻像素相比，输出结果中的每个像素接收到的信息量都不同。我们无法在输出上找到一个连续且均匀重叠的区域。

**当滤波器的大小不能被步长整除时，转置卷积就会产生不均匀的重叠。这种 " 不均匀重叠 " 会产生棋盘效果。事实上，不均匀重叠区域在二维图像中往往更加极端。在二维空间中，两个图案相乘，不均匀度就会被平方化。**

在应用转置卷积时，有两种方法可以减少这种伪影。首先，确保使用的滤波器尺寸能够整除步长，以避免重叠问题。其次，可以使用步长为 1 的转置卷积，这有助于减少棋盘伪影。不过，伪影仍有可能存在，这在许多最新的模型中都可以看到。

[Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/) 进一步提出了一种更好的上采样方法：先调整图像大小（使用近邻插值或双线性插值），然后再做卷积层。通过这种方法，作者避免了棋盘效果。您不妨在自己的应用中尝试一下。

## 7 空洞/扩张/膨胀卷积 （Dilated/Atrous Convolution）

扩张卷积也称空洞/膨胀卷积，其在论文 [[1412.7062] Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs]( https://arxiv.org/abs/1412.7062 ) 和论文 [[1511.07122] Multi-Scale Context Aggregation by Dilated Convolutions]( https://arxiv.org/abs/1511.07122 ) 中进行了介绍。

标准的离散卷积表达式：
$$
(f \ast k)(p) = \sum_{s+t=p}F(s)k(t)
$$

![The standard convolution](https://image-1256466424.cos.accelerate.myqcloud.com/202402202227489_2024_02_22_16.png)

扩张卷积表达式：
$$
(f \ast_l k)(p) = \sum_{s+lt=p}F(s)k(t)
$$

当 $l=1$ 时，扩张卷积就变成了标准卷积。

![The dilated convolution](https://image-1256466424.cos.accelerate.myqcloud.com/1_niGh2BkLuAUS2lkctkd3sA_2024_02_22_44.gif)

直观地说，扩张卷积通过在内核元素之间插入空间来“膨胀”内核。这个附加参数 $l$（扩张率）表示我们想要扩大内核的程度。实现方式可能会有所不同，但通常会在内核元素之间插入 $l-1$ 个空格。下图显示了 $l = 1, 2, 4$ 时的内核大小。

![Receptive field for the dilated convolution](https://image-1256466424.cos.accelerate.myqcloud.com/202402202225177_2024_02_22_33.png)

在图像中，$3 \times 3$ 红点表示卷积后的输出图像为 $3 \times 3$ 像素。虽然三种扩张卷积的输出维度相同，但模型观察到的感受野却大相径庭。当 $l =1$ 时，感受野为 $3 \times 3$。$l =2$ 时为 $7 \times 7$。有趣的是，与这些操作相关的参数数量基本相同，但能获得一个更大的感受野，而不需要增加额外的成本。正因为如此，**扩张卷积被用来在不增加内核大小的情况下廉价地增加输出单元的感受野，这在多个扩张卷积相继叠加时尤为有效。**

## 8 可分离卷积 （空间可分离卷积，深度卷积）

可分离卷积用于某些神经网络架构，如 MobileNet（链接）。我们可以在空间上（空间可分离卷积）或深度上（深度可分离卷积）进行可分离卷积。

### 8.1 空间可分离卷积 (Spatially Separable Convolution)

空间可分离卷积对图像的二维空间维度（即高度和宽度）进行操作。从概念上讲，空间可分离卷积将卷积分解为两个独立的操作。例如，将 $3 \times 3$ 的 Sobel 内核分解为 $3 \times 1$ 和 $1 \times 3$ 内核。

![](https://image-1256466424.cos.accelerate.myqcloud.com/202402202342022_2024_02_23_17.png)

在卷积中，$3 \times 3$ 内核直接与图像卷积。在空间可分离卷积中，$3 \times 1$ 内核首先与图像卷积。然后再应用 $1 \times 3$ 内核。这样就需要 6 个而不是 9 个参数来完成相同的操作。

此外，与卷积相比，空间可分离卷积所需的矩阵乘法更少。举个具体的例子，在一幅 $5 \times 5$ 的图像上使用 $3 \times 3$ 内核（$stride=1，padding=0$）进行卷积，需要在水平方向的 3 个位置（和垂直方向的 3 个位置）扫描内核。总共 9 个位置，如下图中的点所示。每个位置都要进行 9 次元素乘法运算。总的来说，就是 $9 \times 9 = 81$ 次乘法。

![Standard convolution with 1 channel](https://image-1256466424.cos.accelerate.myqcloud.com/202402202344857_2024_02_23_16.png)

另一方面，对于空间可分离卷积，我们首先在 $5 \times 5$ 图像上应用 3 x 1 滤波器。我们在水平方向的 5 个位置和垂直方向的 3 个位置扫描这样的内核。总共是 5 x 3=15 个位置，如下图中的点所示。在每个位置，都要进行 3 次元素乘法运算。即 15 x 3=45 次乘法。现在我们得到了一个 3 x 5 矩阵。现在用一个 1 x 3 内核对矩阵进行卷积，在水平方向和垂直方向各扫描 3 个位置。在这 9 个位置上，每个位置都要进行 3 次元素乘法运算。这一步骤需要 9 x 3=27 次乘法。因此，总体而言，空间可分离卷积需要 45 + 27 = 72 次乘法运算，少于卷积运算。

![Spatially separable convolution with 1 channel](https://image-1256466424.cos.accelerate.myqcloud.com/202402202345974_2024_02_23_32.png)

让我们把上面的例子概括一下。传统卷积需要 (N-2) x (N-2) x m x m 乘法。空间可分离卷积需要 N x (N-2) x m + (N-2) x (N-2) x m = (2N-2) x (N-2) x m 次乘法。空间可分离卷积与标准卷积的计算成本之比为：
$$
\frac{2}{m}+\frac{2}{m(N-2)}
$$

对于图像尺寸 N 大于滤波器尺寸（N >> m）的图层，这一比率变为 2 / m。这意味着在这种渐近情况下（N >> m），空间可分离卷积的计算成本是 3 x 3 滤波器标准卷积的 2/3。5 x 5 滤波器的计算成本是 2 / 5，7 x 7 滤波器的计算成本是 2 / 7，以此类推。

虽然空间可分离卷积节省了成本，但在深度学习中却很少使用。其中一个主要原因是，并非所有的内核都能分成两个更小的内核。如果我们用空间可分离卷积代替所有传统卷积，就会限制我们在训练过程中搜索所有可能的核。训练结果可能达不到最优。

### 8.2 深度可分离卷积 (Depthwise Separable Convolution)

现在，让我们来看看深度可分离卷积，它在深度学习中更为常用（例如在 MobileNet 和 Xception 中）。深度可分离卷积包括两个步骤：**深度卷积**和 1x1 卷积。

在介绍这些步骤之前，我们不妨重温一下前面章节中提到的二维卷积和 1 x 1 卷积。让我们快速回顾一下标准的二维卷积。举个具体例子，假设输入层的大小为 7 x 7 x 3（高度 x 宽度 x 通道），滤波器的大小为 3 x 3 x 3。

![Standard 2D convolution to create output with 1 layer, using 1 filter](https://image-1256466424.cos.accelerate.myqcloud.com/202402202358518_2024_02_23_35.png)

通常情况下，两个神经网络层之间会应用多个滤波器。假设这里有 128 个滤波器。在应用这 128 个二维卷积后，我们会得到 128 个 5 x 5 x 1 的输出映射。然后，我们将这些映射堆叠成一个 5 x 5 x 128 的单层。这样，我们就把输入层（7 x 7 x 3）转换成了输出层（5 x 5 x 128）。空间维度（即高度和宽度）被缩小，而深度则被扩展。

![Standard 2D convolution to create output with 128 layer, using 128 filters](https://image-1256466424.cos.accelerate.myqcloud.com/202402202359386_2024_02_23_51.png)

现在，让我们看看如何利用深度可分离卷积实现同样的变换。

第一步，我们对输入层进行深度卷积。在二维卷积中，我们没有使用大小为 3 x 3 x 3 的单一滤波器，而是分别使用了 3 个核。每个滤波器的大小为 3 x 3 x 1。每个核对输入层的一个通道进行卷积（仅一个通道，而不是所有通道！）。然后，我们将这些映射叠加在一起，创建一个 5 x 5 x 3 的图像。现在我们缩小了空间尺寸，但深度仍与之前相同。

![Depthwise separable convolution](https://image-1256466424.cos.accelerate.myqcloud.com/202402210001419_2024_02_00_13.png)

第二步，为了扩展深度，我们使用核大小为 1x1x3 的 1x1 卷积。将 5 x 5 x 3 输入图像与每个 1 x 1 x 3 内核卷积，可得到大小为 5 x 5 x 1 的图像。

![](https://image-1256466424.cos.accelerate.myqcloud.com/202402210004355_2024_02_00_08.png)

因此，在应用 128 个 1x1 卷积后，我们可以得到一个大小为 5 x 5 x 128 的输出层。

深度可分离卷积的整个过程如下图所示。

![The overall process of depthwise separable convolution](https://image-1256466424.cos.accelerate.myqcloud.com/202402210008425_2024_02_00_48.png)

那么，进行深度可分离卷积的优势是什么？效率！与二维卷积相比，深度可分卷积所需的操作要少得多。

让我们回顾一下二维卷积的计算成本。共有 128 个 3x3x3 内核，移动 5x5 次。即 128 x 3 x 3 x 3 x 5 x 5 = 86400 次乘法运算。

可分离卷积如何？在第一个深度卷积步骤中，有 3 个 3x3x1 的核，移动了 5x5 次。即 3x3x3x1x5x5 = 675 次乘法。在 1 x 1 卷积的第二步中，有 128 个 1x1x3 内核，移动了 5x5 次。即 128 x 1 x 1 x 3 x 5 x 5 = 9 600 次乘法。因此，总体而言，深度可分离卷积需要 675 + 9600 = 10,275 次乘法运算。这只是二维卷积成本的 12%！

那么，对于任意大小的图像，如果我们应用深度可分离卷积，能节省多少时间呢？让我们把上面的例子概括一下。现在，对于大小为 H x W x D 的输入图像，我们要使用大小为 h x h x D（其中 h 为偶数）的 Nc 个核进行二维卷积（stride=1，padding=0）。这将输入层（H x W x D）转换为输出层（H-h+1 x W-h+1 x Nc）。所需的总乘法次数为：
$$
N_c \times h \times h \times D \times (H-h+1) \times (W-h+1)
$$

另一方面，对于相同的变换，深度可分离卷积所需的乘法次数为：
$$
D \times h \times h \times 1 \times (H-h+1) \times (W-h+1) + N_c \times 1 \times 1 \times D \times (H-h+1) \times (W-h+1) = (h \times h + N_c) \times D \times (H-h+1) \times (W-h+1)
$$

现在，深度可分离卷积与二维卷积的乘法运算次数比为：
$$
\frac{1}{N_c}+\frac{1}{h^2}
$$

对于大多数现代架构而言，输出层通常有很多通道，例如几百甚至几千个通道。对于这样的层数（Nc >> h），上述表达式将简化为 $\frac{1}{h^2}$。这意味着，对于这个渐近表达式，如果使用 3 x 3 滤波器，二维卷积的乘法次数是深度可分离卷积的 9 倍，对于 5 x 5 滤波器，则是 25 倍。

使用深度可分离卷积有什么缺点吗？当然有。深度可分卷积减少了卷积中的参数数量。因此，对于一个小模型来说，如果用深度可分卷积代替二维卷积，模型产数量可能会大大降低。因此，模型可能会变得不够理想。不过，如果使用得当，深度可分卷积可以提高效率，而不会对模型性能造成重大损害。

## 9 扁平化卷积

扁平化卷积是在 [[1412.5474] Flattened Convolutional Neural Networks for Feedforward Acceleration](https://arxiv.org/abs/1412.5474) 一文中提出的。直观地说，其原理是应用滤波器分离。我们不再使用一个标准卷积滤波器将输入层映射到输出层，而是将这个标准滤波器分离成 3 个一维滤波器。这种想法与上文所述的空间可分离卷积类似，即一个空间滤波器由两个 1 级滤波器近似。

![](https://image-1256466424.cos.accelerate.myqcloud.com/202402210034317_2024_02_00_04.png)

需要注意的是，如果标准卷积滤波器是秩为 1 的滤波器，那么这种滤波器总是可以分离成三个一维滤波器的互积。但这是一个很苛刻的条件，在实际应用中，标准滤波器的秩要大于 1。正如论文中指出的：“随着分类问题难度的增加，解决问题所需的前导成分数量也越多……深度网络中的学习滤波器具有分布式特征值，直接对滤波器进行分离会导致严重的信息损失”。

为了缓解这一问题，论文限制了感受野中的连接，使模型在训练时可以学习一维分离滤波器。论文称，通过使用扁平化网络（由三维空间所有方向上的连续一维滤波器序列组成）进行训练，可以获得与标准卷积网络相当的性能，而且由于学习参数的大幅减少，计算成本也大大降低。

## 10 分组卷积

分组卷积是在 2012 年的 [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 中提出的。实现它的主要原因是为了在内存有限（每个 GPU 1.5 GB 内存）的情况下，通过两个 GPU 进行网络训练。下面的 AlexNet 在大部分层显示了两条独立的卷积路径。这是在两个 GPU 上进行模型并行化（当然，如果有更多 GPU 可用，也可以进行多 GPU 并行化）。

![](https://image-1256466424.cos.accelerate.myqcloud.com/202402210047048_2024_02_00_02.png)

下面我们将介绍分组卷积的工作原理。首先，传统的二维卷积按以下步骤进行。在这个例子中，通过应用 128 个滤波器（每个滤波器的大小为 3 x 3 x 3），将大小为（7 x 7 x 3）的输入层转换为大小为（5 x 5 x 128）的输出层。或者在一般情况下，通过应用 Dout 个内核（每个内核的大小为 h x w x Din），将大小为（Hin x Win x Din）的输入层转换为大小为（Hout x Wout x Dout）的输出层。

![Standard 2D convolution](https://image-1256466424.cos.accelerate.myqcloud.com/202402210048427_2024_02_00_24.png)

在分组卷积中，滤波器被分成不同的组。每组负责一定深度的传统二维卷积。下面的例子可以更清楚地说明这一点。

![Grouped convolution with 2 filter groups](https://image-1256466424.cos.accelerate.myqcloud.com/202402210049219_2024_02_00_18.png)

上图是使用 2 个滤波器组进行分组卷积的示意图。在每个滤波器组中，每个滤波器的深度仅为标准二维卷积深度的一半。每个滤波器组包含 Dout /2 个滤波器。第一个滤波器组（红色）与输入层的前半部分（[:, :, 0:Din/2]）卷积，而第二个滤波器组（蓝色）与输入层的后半部分（[:, :, Din/2:Din]）卷积。因此，每个滤波器组都会产生 Dout/2 个通道。总的来说，两个滤波器组会产生 2 x Dout/2 = Dout 个通道。然后，我们将这些通道与 Dout 通道一起堆叠到输出层。

### 10.1 分组卷积与深度卷积

你可能已经注意到分组卷积和深度可分离卷积中使用的深度卷积之间的联系和区别。如果滤波器组数与输入层通道数相同，则每个滤波器的深度为 Din / Din = 1。这与深度卷积的滤波器深度相同。

另一方面，每个滤波器组现在都包含 Dout / Din 滤波器。总的来说，输出层的深度为 Dout。这与深度卷积不同，深度卷积不改变层深度。在深度可分离卷积中，层深度将通过 1x1 卷积得到扩展。

分组卷积的优点：

第一个优势是训练效率高。由于卷积被分为几条路径，因此每条路径都可以由不同的 GPU 分别处理。这一程序允许在多个 GPU 上以并行方式进行模型训练。与使用一个 GPU 对所有图像进行训练相比，在多个 GPU 上进行这种模型并行化处理，每一步都能向网络输入更多图像。**模型并行化被认为优于数据并行化**。后一种方法是将数据集分成若干批次，然后在每个批次上进行训练。然而，当批次规模变得太小时，我们基本上是在进行随机梯度下降，而不是批次梯度下降。这将导致收敛速度更慢，有时甚至更难拟合。

分组卷积对于训练深度神经网络非常重要，如下图所示的 ResNeXt 中的情况。

![](https://image-1256466424.cos.accelerate.myqcloud.com/202402210121676_2024_02_01_01.png)

第二个优点是模型更高效，即模型参数随着滤波器组数的增加而减少。在前面的例子中，2D 卷积中的滤波器参数为 h x w x Din x Dout。在有 2 个滤波器组的分组卷积中，滤波器的参数为（h x w x Din/2 x Dout/2）x 2。参数数量减少了一半。

第三个优势有点出人意料。分组卷积可能比二维卷积提供更好的模型。另一篇精彩的博客 [A Tutorial on Filter Groups (Grouped Convolution)](https://blog.yani.io/filter-group-tutorial/) 对此进行了解释。以下是简要摘要。

原因与稀疏滤波器的关系有关。下图是相邻层滤波器之间的相关性。这种关系是稀疏的。

![The correlation matrix between filters of adjacent layers in a Network-in-Network model trained on CIFAR10. Pairs of highly correlated filters are brighter, while lower correlated filters are darker.](https://image-1256466424.cos.accelerate.myqcloud.com/202402210126294_2024_02_01_18.png)

分组卷积的相关图如何？

![The correlations between filters of adjacent layers in a Network-in-Network model trained on CIFAR10, when trained with 1, 2, 4, 8 and 16 filter groups](https://image-1256466424.cos.accelerate.myqcloud.com/1_lAND4yAVjjQBR-DFmZBh8Q_2024_02_01_15.gif)

上图是使用 1、2、4、8 和 16 个滤波器组训练模型时，相邻层滤波器之间的相关性。文章给出了一个理由：“滤波器组的作用是在通道维度上以块对角结构稀疏性进行学习……在具有滤波器组的网络中，具有高相关性的滤波器以更有条理的方式进行学习。实际上，无需学习的滤波器关系不再被参数化。通过这种突出的方式减少网络中的参数数量，就不那么容易过度拟合了，因此类似正则化的效果可以让优化器学习到更准确、更高效的深度网络。”

![](https://image-1256466424.cos.accelerate.myqcloud.com/202402210131567_2024_02_01_22.png)

此外，每个滤波器组都能学习到独特的数据表示。AlexNet 的作者注意到，滤波器组似乎将学习到的滤波器分为两个不同的组，即黑白滤波器和彩色滤波器。

## 11 洗牌分组卷积

Magvii 公司（Face++）的 [[1707.01083] ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices]( https://arxiv.org/abs/1707.01083 ) 是一种计算效率高的卷积架构，专为计算能力非常有限（如 10-150 MFLOPs）的移动设备而设计。

洗牌分组卷积背后的理念与分组卷积（例如在 MobileNet 和 ResNeXt 中使用）和深度可分离卷积（在 Xception 中使用）背后的理念有关。

总的来说，洗牌分组卷积包括分组卷积和通道洗牌。

在分组卷积部分，我们知道滤波器被分成不同的组。每组负责一定深度的传统二维卷积。总的操作量大大减少。例如，在下图中，我们有 3 个滤波器组。第一组滤波器对输入层中的红色部分进行卷积。同样，第二和第三滤波器组分别对输入层中的绿色和蓝色部分进行卷积。每个滤波器组的内核深度仅为输入层总通道数的 1/3。在本示例中，在第一次分组卷积 GConv1 之后，输入层被映射到中间特征图。然后通过第二个分组卷积 GConv2 将该特征图映射到输出层。

![](https://image-1256466424.cos.accelerate.myqcloud.com/202402210152862_2024_02_01_20.png)

分组卷积的计算效率很高。但问题在于，每个滤波器组只能处理前几层固定部分传递下来的信息。例如，在上图中，第一滤波器组（红色）只处理从前 1/3 输入通道传递下来的信息。蓝色滤波器组（蓝色）只处理从最后 1/3 输入通道传递下来的信息。因此，每个过滤组只能学习几个特定的特征。这一特性阻碍了通道组之间的信息流动，并在训练过程中削弱了表征。为了克服这个问题，我们采用了通道洗牌的方法。

通道洗牌的原理是将来自不同滤波器组的信息混合在一起。在下图中，我们用 3 个滤波器组进行第一次分组卷积 GConv1 后得到了特征图。在将该特征图输入第二个分组卷积之前，我们首先将每个组中的通道分成几个子组。然后将这些子组混合起来。

![](https://image-1256466424.cos.accelerate.myqcloud.com/202402210153693_2024_02_01_15.png)

洗牌后，我们像往常一样继续执行第二次分组卷积 GConv2。但现在，由于洗牌层中的信息已经混合，我们基本上将 GConv2 中的每个组与特征图层（或输入层）中的不同子组进行馈送。因此，我们允许信息在通道组之间流动，并加强了表征。

## 12 逐点分组卷积

ShuffleNet 论文（链接）还引入了点分组卷积。通常情况下，分组卷积（如 MobileNet（链接）或 ResNeXt（链接）中的分组卷积）是对 3x3 空间卷积进行分组操作，而不是对 1 x 1 卷积进行分组操作。

ShuffleNet 论文认为，1 x 1 卷积的计算成本也很高。论文建议将分组卷积也应用于 1 x 1 卷积。点分组卷积，顾名思义，就是对 1 x 1 卷积进行分组操作。其操作与分组卷积完全相同，只有一处修改 -- 在 1x1 滤波器上执行，而不是 NxN 滤波器（N>1）。

在 ShuffleNet 论文中，作者利用了我们所学到的三种卷积：(1) Shuffled 分组卷积；(2) pointwise 分组卷积；(3) depthwise 可分离卷积。这种架构设计在保持准确性的同时，大大降低了计算成本。例如，在实际移动设备上，ShuffleNet 和 AlexNet 的分类误差相当。然而，计算成本却从 AlexNet 的 720 MFLOPs 大幅降低到 ShuffleNet 的 40-140 MFLOPs。凭借相对较小的计算成本和良好的模型性能，ShuffleNet 在面向移动设备的卷积神经网络领域获得了青睐。

## 13 参考

### 13.1 博客和文章

- “An Introduction to different Types of Convolutions in Deep Learning” ([Link](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d))
- “Review: DilatedNet — Dilated Convolution (Semantic Segmentation)” ([Link](https://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5))
- “ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices” ([Link](https://medium.com/syncedreview/shufflenet-an-extremely-efficient-convolutional-neural-network-for-mobile-devices-72c6f5b01651))
- “Separable convolutions “A Basic Introduction to Separable Convolutions” ([Link](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728))
- Inception network “A Simple Guide to the Versions of the Inception Network” ([Link](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202))
- “A Tutorial on Filter Groups (Grouped Convolution)” ([Link](https://blog.yani.io/filter-group-tutorial/))
- “Convolution arithmetic animation” ([Link](https://github.com/vdumoulin/conv_arithmetic))
- “Up-sampling with Transposed Convolution” ([Link](https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0))
- “Intuitively Understanding Convolutions for Deep Learning” ([Link](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1))

### 13.2 论文

- Network in Network ([Link](https://arxiv.org/abs/1312.4400))
- Multi-Scale Context Aggregation by Dilated Convolutions ([Link](https://arxiv.org/abs/1511.07122))
- Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs ([Link](https://arxiv.org/abs/1412.7062))
- ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices ([Link](https://arxiv.org/abs/1707.01083))
- A guide to convolution arithmetic for deep learning ([Link](https://arxiv.org/abs/1603.07285))
- Going deeper with convolutions ([Link](https://arxiv.org/abs/1409.4842))
- Rethinking the Inception Architecture for Computer Vision ([Link](https://arxiv.org/pdf/1512.00567v3.pdf))
- Flattened convolutional neural networks for feedforward acceleration ([Link](https://arxiv.org/abs/1412.5474))
- Xception: Deep Learning with Depthwise Separable Convolutions ([Link](https://arxiv.org/abs/1610.02357))
- MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications ([Link](https://arxiv.org/abs/1704.04861))
- Deconvolution and Checkerboard Artifacts ([Link](https://distill.pub/2016/deconv-checkerboard/))
- ResNeXt: Aggregated Residual Transformations for Deep Neural Networks ([Link](https://arxiv.org/abs/1611.05431))
