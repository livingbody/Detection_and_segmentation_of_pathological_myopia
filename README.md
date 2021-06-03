# 飞桨常规赛：PALM眼底彩照视盘探测与分割 5月第2名方案
# 一、PaddleX助力【常规赛：PALM眼底彩照视盘探测与分割】

**github 地址： [https://github.com/livingbody/Detection_and_segmentation_of_pathological_myopia](https://github.com/livingbody/Detection_and_segmentation_of_pathological_myopia)**

**aistudio地址：[https://aistudio.baidu.com/aistudio/projectdetail/2027061](https://aistudio.baidu.com/aistudio/projectdetail/2027061)**

## 1.内容介绍

本文采用Paddlex傻瓜式操作，一键获得常规赛第二名，仅供大家参考！

![](https://ai-studio-static-online.cdn.bcebos.com/3f6c99fe5da94413a8a5ba6e19fd6f24b314d25a1b6f4bc1879bc51ba8b0f31b)

## 2.赛题介绍
比赛地址：[https://aistudio.baidu.com/aistudio/competition/detail/87](https://aistudio.baidu.com/aistudio/competition/detail/87)


**赛题简述**
	
    PALM眼底视盘检测与分割常规赛的重点是研究和发展与患者眼底照片结构分割相关的算法。该常规赛的目标是评估和比较在一个常见的视网膜眼底图像数据集上分割视盘的自动算法。该任务目的是对眼底图像的视盘进行检测，若存在视盘结构，需从眼底图像中分割出视盘区域；若无视盘结构，分割结果直接置全背景。

![](https://ai-studio-static-online.cdn.bcebos.com/938ab4fac88e44969e61f8f10181ca1366c53fbc3d6147ff80487b03c964543e)


**数据基本标签**

	标签为 0 代表视盘（黑色区域）；标签为 255 代表其他（白色区域）。


**训练数据集**

文件名称：Train

Train文件夹里有fundus_images文件夹和Disc_Masks文件夹。

* fundus_images文件夹内包含800张眼底彩照，分辨率为1444×1444，或2124×2056。命名形如H0001.jpg、N0001.jpg、P0001.jpg和V0001.jpg。

* Disc_Masks文件夹内包含fundus_images里眼底彩照的视盘分割金标准，大小与对应的眼底彩照一致。命名前缀和对应的fundus_images文件夹里的图像命名一致，后缀为bmp。

**测试数据集**

文件名称：PALM-Testing400-Images

* 包含400张眼底彩照，命名形如T0001.jpg。

## 3.思路办法
分割模型可用的方法较多，分别为：
* 传统模型手写法
* 使用paddlex快速法
* 使用paddleseg端到端解决法

在此选择paddlex来解决

## 4.重点难点
* 分割的label的图像mask数据转换（此次涉及背景255----1转换）
* 原图与label图大小检测，事实证明此次许多原图和label图大小不一致，直接弃用
* 预测结果图像格式转换为需要的提交格式（1--255转换）
* 炼丹参数设置，设置不对容易炸，且精度不高

## 5.处理重启不释放显存技巧
```
killall -9 python
```

## 6.快速显示aistudio图片技巧
```
%cd ~
from PIL import Image

# 读取图片
png_img = Image.open('dataset/MyDataset/JPEGImages/H0003.jpg')
png_img  # 展示图片
```
## 7.标签背景转换技巧
```
import PIL.Image as Image
from tqdm import tqdm
import numpy as np
import os

!rm 'dataset/MyDataset/Annotations/.DS_Store'

print('————开始数据预处理转换————')
print('转换说明:')
print('\t 1. 默认标签为255与0，为了训练方便，将255转换为1，变成2分类问题')
print('\t 2. 新标签0与1，预测结束进行后处理即可得到赛题需要的结果')


pretrans_img_path = 'dataset/MyDataset/Annotations'
for _, _, files in os.walk(pretrans_img_path):
    for f in tqdm(files):
        img = Image.open(os.path.join(pretrans_img_path, f))
        img = np.asarray(img).copy()
        img[img == 255] = 1
        img = Image.fromarray(img)
        img.save(os.path.join(pretrans_img_path, f))
```

```
import PIL.Image as Image
from tqdm import tqdm
import numpy as np
import os
import time

print('————开始提交结果前的后处理————')

!mkdir result1
print("新建result1保存处理后结果文件夹")
start = time.time()
# 预测结果路径
open_root = 'result'
# 转换处理后保存路径
save_root = 'result1'
for _, _, files in os.walk(open_root):
    for f in tqdm(files):
        img = Image.open(os.path.join(open_root, f))
        img = np.asarray(img).copy()
        # 北京标签改回255
        img[img == 1] = 255
        img = Image.fromarray(img)
        img.save(os.path.join(save_root, f))

print('后处理完成(cost: {0} s)！'.format(time.time()-start))
```



# 二、比赛数据集情况
PALM-Testing400-Images : 测试数据集文件夹

Train : 训练数据集文件夹

* Disc_Masks   ; 标注图片
* fundus_image  : 原始图片

> 注意没有验证数据集，这里提供一个简单的划分程序，划分比例为0.7

## 1.解压缩


```python
# 解压数据集到PaddleSeg目录下的data文件夹
%cd ~
!unzip -oq /home/aistudio/data/data85136/常规赛：PALM眼底彩照视盘探测与分割.zip -d dataset
```

## 2.格式化存储数据


```python
# 查看数据集文件的树形结构
!rm dataset/__MACOSX -rf
!mv dataset/常规赛：PALM眼底彩照视盘探测与分割 dataset/data
!tree -d dataset/data
```


```python
!mv dataset/data/Train dataset/MyDataset
!mv dataset/MyDataset/fundus_image dataset/MyDataset/JPEGImages
!mv dataset/MyDataset/Disc_Masks dataset/MyDataset/Annotations
```


```python
!tree -d dataset/MyDataset
```

## 3.数据查看

通过PIL的Image读取图片查看以下原数据与Label标注情况


```python
%cd ~
from PIL import Image

# 读取图片
png_img = Image.open('dataset/MyDataset/JPEGImages/H0003.jpg')
png_img  # 展示图片
```


```python
bmp_img = Image.open('dataset/MyDataset/Annotations/H0003.bmp')
bmp_img   # 展示图片
```


```python
# 比对大小
print(png_img.size)
print(bmp_img.size)
```

## 4.背景255转1

可以看出，白色部分全是255，黑色为有效标注区域(0值)


```python
import PIL.Image as Image
from tqdm import tqdm
import numpy as np
import os

!rm 'dataset/MyDataset/Annotations/.DS_Store'

print('————开始数据预处理转换————')
print('转换说明:')
print('\t 1. 默认标签为255与0，为了训练方便，将255转换为1，变成2分类问题')
print('\t 2. 新标签0与1，预测结束进行后处理即可得到赛题需要的结果')


pretrans_img_path = 'dataset/MyDataset/Annotations'
for _, _, files in os.walk(pretrans_img_path):
    for f in tqdm(files):
        img = Image.open(os.path.join(pretrans_img_path, f))
        img = np.asarray(img).copy()
        img[img == 255] = 1
        img = Image.fromarray(img)
        img.save(os.path.join(pretrans_img_path, f))
```

## 5.批量重命名后缀名


```python
%cd dataset/MyDataset/Annotations
!rename 's/\.bmp/\.png/' ./*
%cd ~
```

## 6.原图与label图尺寸检查
不合适的直接删掉


```python
import PIL.Image as Image
from tqdm import tqdm
import numpy as np
import os


pretrans_img_path = 'dataset/MyDataset/'
for _, _, files in os.walk(pretrans_img_path+'JPEGImages'):
    for f in tqdm(files):
        img1 = Image.open(os.path.join(pretrans_img_path,'JPEGImages', f))
        
        img2 = Image.open(os.path.join(pretrans_img_path,'Annotations', f.split('.')[0]+'.png'))

        if img1.width!=img2.width:
            print(f)
            os.remove(os.path.join(pretrans_img_path,'Annotations', f.split('.')[0]+'.png'))
            os.remove(os.path.join(pretrans_img_path,'JPEGImages', f))
```

# 三、划分数据集与数据预处置

## 1.paddlex安装


```python
!pip install paddlex
```

## 2.数据集划分
* test就不要了，省下来的都用于训练
* 数据存储格式前面已经处理了

移除原数据，减小项目空间，减少下一次进入和退出保存时花的时间


```python
!paddlex --split_dataset --format Seg --dataset_dir dataset/MyDataset --val_value 0.1
```

# 四、数据集设置


## 1.环境设置


```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddlex as pdx
from paddlex.seg import transforms
```

## 2.transform设置


```python
# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/seg_transforms.html
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.ResizeRangeScaling(),
    transforms.RandomPaddingCrop(crop_size=512), transforms.Normalize()
])

eval_transforms = transforms.Compose([
    transforms.ResizeByLong(long_size=512),
    transforms.Padding(target_size=512), transforms.Normalize()
])
```

## 3.dataset设置


```python
# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-segdataset
train_dataset = pdx.datasets.SegDataset(
    data_dir='dataset/MyDataset',
    file_list='dataset/MyDataset/train_list.txt',
    label_list='dataset/MyDataset/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.SegDataset(
    data_dir='dataset/MyDataset',
    file_list='dataset/MyDataset/val_list.txt',
    label_list='dataset/MyDataset/labels.txt',
    transforms=eval_transforms)
```

# 五、开始构建比赛模型

## 1.导入需要的库


```python
# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
num_classes = len(train_dataset.labels)
print(num_classes)
```

## 2.创建模型


```python
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#paddlex-seg-deeplabv3p
# model = pdx.seg.DeepLabv3p(
#     num_classes=num_classes,
#     backbone='MobileNetV3_large_x1_0_ssld',
#     pooling_crop_size=(512, 512))

# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#paddlex-seg-deeplabv3p
model = pdx.seg.DeepLabv3p(
    num_classes=num_classes,
    backbone='MobileNetV3_large_x1_0_ssld',
    pooling_crop_size=(512, 512))
```

## 3.开始训练

### 3.1 训练


```python
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#train
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=100,
    train_dataset=train_dataset,
    train_batch_size=110,
    eval_dataset=eval_dataset,
    learning_rate=0.01,
    save_dir='output/deeplabv3p_mobilenetv3_large_ssld',
    # resume_checkpoint ='output/hrnet/epoch_16',
    use_vdl=True)
```

### 3.2 训练图

#### vdl图
![](https://ai-studio-static-online.cdn.bcebos.com/d14d8bda6b8c4a56abd86887a6f2ad278886eaf5095349119b6352014d6fd0ec)
#### cpu内存消耗图
![](https://ai-studio-static-online.cdn.bcebos.com/772fba399a984c3f893f8443dbeebc10443a19944e044a6ca9a64e75640e4348)
#### gpu显存消耗图
![](https://ai-studio-static-online.cdn.bcebos.com/0b957bba010a491ba126659abb2253efafc7329b140748e982fc3b8b5f32ed4e)


## 4.开始预测

预测的配置略微不同，需要读取`test_list.txt`中的文件进入list中，然后传入list以及Image_dir进行预测

> 前面的训练与验证是通过给dir，自动搜寻，这里不一样，要注意一下哦

### 4.1获取预测数据列表


```python
base_dir='dataset/data/PALM-Testing400-Images'
test_list=[]
for f in os.listdir(base_dir):
    test_list.append(f+'\n')
print(len(test_list))
# 写入文件
with open('test_list.txt', 'w') as f:
    f.writelines(test_list)
```


```python
test_list = []
test_root = 'dataset/data/PALM-Testing400-Images'      # 之前划分数据图像保存的根路径
with open('test_list.txt') as f: 
    for i in f.readlines():
        test_list.append(os.path.join(test_root, i[:-1]))   # 逐行写入，-1是为了去掉 \n
print(test_list[0])
```

### 4.2 循环预测


```python
import paddlex as pdx
model = pdx.load_model('output/deeplabv3p_mobilenetv3_large_ssld/best_model')

from tqdm import tqdm
import cv2

# 预测结果保存路径
!mkdir result
print("新建result保存结果文件夹")

out_base = 'result/'
# 之前划分数据图像保存的根路径
test_root = 'dataset/data/PALM-Testing400-Images/'      
# 如果不存在result目录，则新建
if not os.path.exists(test_root):
    os.makedirs(out_base)


for im in tqdm(os.listdir(test_root)):
    if not im.endswith('.jpg'):
        continue
    pt = test_root + im
    
    # 预测
    result = model.predict(pt)
    # 另存
    cv2.imwrite(out_base+im.replace('jpg', 'png'), result['label_map'])    
```

## 5.后处理并生成提交文件


### 5.1将背景1处理为255


```python
import PIL.Image as Image
from tqdm import tqdm
import numpy as np
import os
import time

print('————开始提交结果前的后处理————')

!mkdir result1
print("新建result1保存处理后结果文件夹")
start = time.time()
# 预测结果路径
open_root = 'result'
# 转换处理后保存路径
save_root = 'result1'
for _, _, files in os.walk(open_root):
    for f in tqdm(files):
        img = Image.open(os.path.join(open_root, f))
        img = np.asarray(img).copy()
        # 北京标签改回255
        img[img == 1] = 255
        img = Image.fromarray(img)
        img.save(os.path.join(save_root, f))

print('后处理完成(cost: {0} s)！'.format(time.time()-start))
```

### 5.2重命名文件夹并打压缩


```python
# 重命名
!mv result1 Disc_Segmentation
```


```python
# 生成提交文件
!zip -r Disc_Segmentation.zip Disc_Segmentation/
```

# 六、提交结果
这块一定要注意提交的东西：平时测试提交的结果文件，后续审查需要提交代码、模型文件，缺一不可。所以每次训练感觉好的模型、代码及时保存。

## 1.上传生成的压缩包即可提交
![](https://ai-studio-static-online.cdn.bcebos.com/f9328734f8934ea793e828b20bd097929ec669cf1c88413e89a7c3e4cb0d4b89)

## 2.打包最佳模型

```
zip -r  best_model.zip ./best_model/
```

![](https://ai-studio-static-online.cdn.bcebos.com/241e99f153c7497b9af3e3eabdeb8d6a7b41c8ffa2d248c989835872d0bbb259)

## 3.打包notebook
```
zip ipynb.zip 1970708.ipynb 
```

![](https://ai-studio-static-online.cdn.bcebos.com/becb365da0514c54858718048b8b0f493acb0a4adcb94d27bbfb624f13c8b7e5)




# 七、注意事项
## 1.这种题要多看看，有些分割的mask尺寸就是不对，坑！！！
这个原因我删除了十几条训练数据。
## 2.分割数据一定要转格式
格式一定要正确，不正确的话很容易出问题，所以训练前数据处理，预测后数据处理很重要。
## 3.其它一些清理步骤选择性使用即可
在使用过程中会产生大量中间数据，请谨慎使用。

# 八、附件
## 1.结果文件
[Disc_Segmentation.zip](Disc_Segmentation.zip)
## 2.运行代码
[javaroom.ipynb](javaroom.ipynb)

## 3.模型文件
[epoch_98_ok.zip](epoch_98_ok.zip)