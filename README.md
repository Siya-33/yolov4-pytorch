# 课设简介

fork了[bubbing的项目](https://github.com/bubbliiiing/yolov4-pytorch)

使用YoloV4对voc数据集进行目标检测，总结实验原理

# 关键代码

见[github地址](https://github.com/Siya-33/yolov4-pytorch)或[实验原理](#jump1)部分

# 实验设置

## 运行环境

vs2019+cuda11.0+torch1.7.0+cudnn8.0.5


> pip install torch==1.7.0+cu110 torchvision==0.8.0+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html


并配置相关的环境变量

上述安装失败，则去[pytorch.org](https://download.pytorch.org/whl/torch_stable.html)下载whl文件安装

> pip install torch-1.7.0+cu110-cp38-cp38-win_amd64.whl

## 数据下载

VOC2007数据集

[预训练权重文件](https://pan.baidu.com/s/19YLQSxqMMv12eV_IfuNFBw?pwd=ksks)

## 数据集介绍

使用VOC数据集

> VOC2007
> - Annotations
> - ImageSets
> - JPEGImages

Annotations存放标注，具体内容位于xml文件内的object

ImageSets下划分训练集、验证集和测试集

JPEGImages存放图片

**voc__annotation.py**进行对数据集的划分，指定训练集与测试集、验证集的比例(9:1)，并输出图片路径

## 参数设置

**train.py**

classes_path 设置对应数据集的类别文件
model_path 权重文件

Freeze_batch_size 冻结训练参数 可以较大
Unfreeze_batch_size 解冻训练参数 建议调小

train_annotation_path 
val_annotation_path   获得图片的路径和标签

## 训练及预测

将数据集放入对应的文件夹下，修改model_path，如要训练自己的数据集需要修改classes_path和先验框文件

运行**train.py**，此处使用预训练权重进行训练，训练所得的权重文件保存于log文件夹下

运行**predict.py**，输入图片路径，进行单张图片目标检测。也可以修改其mode参数，改为使用摄像头进行目标检测

# 实验原理<span id = "jump1"> </span><span id = "jump1"> </span>
## 模型概述

<img src="/md_image/net.PNG" alt="net" style="zoom:67%;" />

模型主要由三大部分组成

- 主干特征提取网络 CSPDarkNet53

- 特征金字塔 SPP PANet（加强特征提取网络）

- 分类回归层  yolohead


基本流程如下

输入一张(416,416,3)的图像经过一层卷积，并用Mish激活。之后通过一系列残差网络结构Resblock_body压缩高宽，扩张通道数，获得更高的语义信息，选取最后的三个有效特征层用于之后的操作。首先对最后一个特征层做三次卷积，用4个不同大小的池化核进行最大池化，将结果堆叠并做三次卷积。进入PANet，做上采样，和之前的第二个特征层做数据融合，继续上采样做融合。再次进行两次下采样，将三层提取到的特征输入Yolo Head进行预测

### 主干提取网络backbone

**CSPdarknet.py**

激活函数采用Mish

$						Mish = x \times tanh(ln(1+e^x))$

<img src="/md_image/Mish.PNG" alt="Mish" style="zoom:50%;" />

Resblock_body  一系列残差网络构成的大卷积块

结构图如下

<img src="/md_image/Resblock_body.PNG" alt="Resblock_body" style="zoom:50%;" />

残差块堆叠分成了两部分，一部分做常规n次的堆叠，另一部分直接连接到输出，分别对应conv0和conv1

```python
self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)
self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)
self.blocks_conv = nn.Sequential(
    *[Resblock(out_channels//2) for _ in range(num_blocks)],
    BasicConv(out_channels//2, out_channels//2, 1)
)
self.concat_conv = BasicConv(out_channels, out_channels, 1)
```

通过5个Resblock_body得到三个有效特征层

### 特征金字塔

**yolo.py**

#### SPP

利用不同大小的池化核进行池化，再堆叠起来作为输出

由于padding设置为pool_size的1/2，输出尺寸一致，所以直接堆叠

```python
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)
        return features
```

#### PANet

将输入数据做上采样、堆叠、卷积、下采样等一系列操作，模型简述中已有，不再赘述。值得一提的是每次对堆叠完的特征做5次卷积，它是以1×1和3×3的卷积核交替进行的，有助于减少参数量和提取特征

```python
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m
```

最后得到三个更有语义信息的特征层

### 分类回归层

**yolo.py**

同yolov3，做两次卷积得到预测结果

<img src="/md_image/yolo_head.PNG" alt="yolo_head" style="zoom:50%;" />

```python
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m
```

3个yolo head输出的通道数均为75

以 (75,13,13) 为例(pytorch中通道数在前)，其将图像划分为13×13的网格，有3个预置的先验框，判断物体是否在先验框内，并对先验框调整作为预测框

**75=3×25  25=20+1+4**

于是之前的数据就可以理解了，3个先验框 ，voc数据集20个类，1判断内部是否有物体，4指预测框的参数(x,y,h,w)

## 先验框解码和调整

**utils_bbox.py**

以 (batch_size,75,13,13)为例 416/13=32 

将3个先验框映射到13×13的特征层，每个网格对应原图32个像素

之后将每个网格点加上x和y得到中心，并计算出框的高和宽，这样四个参数就确定了预测框的位置

```
pred_boxes          = FloatTensor(prediction[..., :4].shape)
pred_boxes[..., 0]  = x.data + grid_x
pred_boxes[..., 1]  = y.data + grid_y
pred_boxes[..., 2]  = torch.exp(w.data) * anchor_w
pred_boxes[..., 3]  = torch.exp(h.data) * anchor_h
```

其中x用了sigmoid将偏移限制在0~1中，即只能向右下角偏移

得到预测框的位置后，还会进行置信度排序和非极大抑制的操作

## 预测


输入图像转换为RGB，加灰条防止失真进行resize

计算预测框，进行非极大抑制（取出每一种类得分最大的框，计算与其他的框的交并比，大于阈值则剔除）

得到预测框的种类、坐标、得分，把它们绘制在图上

![predict_i](/md_image/predict_i.PNG)

### LOSS

**yolo_training.py**

loss由三部分组成

- 对正样本计算CIOU回归损失
- 先验框内部是否包含物体的交叉熵损失
- 种类置信度损失
## 数据集介绍

使用VOC数据集

VOC2007
- Annotations
- ImageSets
- JPEGImages

Annotations存放标注，具体内容位于xml文件内的object

ImageSets下划分训练集、验证集和测试集

JPEGImages存放图片

voc__annotation.py进行对数据集的划分，指定训练集与测试集、验证集的比例(9:1)，并输出图片路径

## 参数设置

**train.py**

classes_path 设置对应数据集的类别文件
model_path 权重文件


Freeze_batch_size 冻结训练参数 可以较大

Unfreeze_batch_size 解冻训练参数 建议调小

train_annotation_path  

val_annotation_path   获得图片的路径和标签

运行**train.py**，此处使用预训练权重进行训练

训练所得的权重文件保存于log文件夹下

运行**predict.py**，输入图片路径，进行目标检测。也可以修改其mode参数，改为使用摄像头进行目标检测

# 评估

**get_map.py**

map指标和对数平均误检率如下

<center class="half">
	<img src="/md_image/mAP.png" alt="mAP" style="zoom:72%;" />
	<img src="/md_image/lamr.png" alt="lamr" style="zoom:72%;" />
</center>

以bottle为例
<figure class="half">
	<img src="E:\Homework\CV\报告\image\bottle_AP.png" alt="bottle_AP" style="zoom:72%;" />
	<img src="E:\Homework\CV\报告\image\bottle_F1.png" alt="bottle_F1" style="zoom:72%;" />
	<img src="E:\Homework\CV\报告\image\bottle_prec.png" alt="bottle_prec" style="zoom:72%;" />
	<img src="E:\Homework\CV\报告\image\bottle_re.png" alt="bottle_re" style="zoom:72%;" />
</figure>
# 对实验结果的原理性分析

由上述评估结果可见，yolov4对小目标的检测存在缺陷，漏检率高。因为小目标往往分辨率低、体积小，网络不断提取高层的特征语义过程中，感受野增大，忽略了小目标。也有部分是因为训练集样本不平衡，因此可以通过训练数据数据增强、更改预测框的调整算法、自适应先验框等方法来改善，这部分在结论与总结中有提到。或者是加入注意力机制，yolov5针对小目标检测有很多改进的方面。


# 结论与总结

相比于yolov3，yolov4有相当多的改进点

## CIoU

<img src="/md_image/CIoU.PNG" alt="CIoU" style="zoom:50%;" />
$$
CIoU=I o U-\frac{\rho^{2}\left(b, b^{g t}\right)}{c^{2}}-\alpha v
$$
$b$和$b^{gt}$分别代表了预测框和真实框的中心点，$\rho^2$代表的是计算两个中心点间的欧式距离。 ![[公式]](https://www.zhihu.com/equation?tex=c) 代表的是能够同时包含预测框和真实框的最小闭包区域的对角线距离

其中$\alpha$是权重函数，$v$度量两个框宽高比的相似性，使得宽高比趋向于一致
$$
\alpha=\frac{v}{1-IoU+v}\\\\
v=\frac{4}{\pi^{2}}\left(\arctan \frac{w^{g t}}{h^{g t}}-\arctan \frac{w}{h}\right)^{2}\\\\
Loss=1-CIoU
$$
相比于只计算IoU，这种方法考虑了目标与anchor之间的距离，重叠率、尺度以及宽高比。在预测框和真实框不重叠的情况下，或者是水平垂直的情况下都能快速地收敛，不会出现像IoU一样发散的情况。总的来说就是使收敛更快更准确了

## Eliminate grid sensitivity

原本计算预测框中心点的位置是通过左上角的网格点加上x,y上的偏移量得到的，公式如下
$$
b_x=\sigma(x\_offset)+c_x\\
b_y=\sigma(y\_offset)+c_y
$$
但是如果目标中心点靠近左上角，就较难预测。于是引入了一个缩放系数并设置为2,偏移范围扩张到了-0.5~1.5
$$
b_x=\sigma(2 ⋅ x\_offset-0.5)+c_x\\
b_y=\sigma(2 ⋅ y\_offset-0.5)+c_y
$$
## Mosaic数据增强

在数据预处理时将四张图片进行翻转、缩放等操作拼成一张图片，提高学习样本的多样性，且一次计算能够处理四张图片

image待处理

# 后记

**实验过程中遇到的部分琐碎问题总结**

> 关于OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.错误

在代码开头加上

1. import os

2. os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

> gbk编码错误

标签和路径避免中文，打开文件过程中将encoding设置为utf-8
