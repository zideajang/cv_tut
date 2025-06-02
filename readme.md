哈喽！是不是觉得最近几年，各种酷炫的深度学习模型，特别是 Transformer，在计算机视觉领域简直是“横扫千军”？从图像识别到目标检测，再到图像生成，Transformer 的身影无处不在，大有“一统江湖”的架势。这不禁让人思考：那我们以前学过的那些**传统机器学习算法**，是不是就此“退休”了呢？或者说，它们还有没有“用武之地”？

对于刚刚踏入计算机视觉大门的朋友们来说，这个问题尤其重要。毕竟，现在大家一上来可能就直接“拥抱”深度学习，但了解一下深度学习出现之前，计算机视觉都有哪些**经典任务**，以及当时**行之有效的算法**，绝对能让你对这个领域有更深刻的理解。

---

### 为什么还要看传统算法？

你可能会问：“现在深度学习这么厉害，我还学那些‘老古董’干嘛？” 别急，这里面可是大有学问！深度学习的很多技巧和创新点，并不是凭空冒出来的，它们往往**借鉴了前人的经验和成果**。就像盖高楼，地基打得牢，上面的建筑才能更稳固。

举个例子，深度学习中很多**特征提取**的思想，都能在传统算法中找到影子。再比如，一些**损失函数**的设计，也可能受到了传统优化方法的启发。所以，了解传统算法，能帮助我们更好地理解深度学习模型的**设计理念**和**演进脉络**。

---

### 计算机视觉的经典任务与传统算法

在深度学习“霸屏”之前，计算机视觉领域也有一系列经典的“挑战”，以及针对这些挑战提出的巧妙算法。让我们一起来回顾一下：

---

好的，这份大纲已经很不错了！它覆盖了计算机视觉的基础操作、传统算法的关键应用以及一些具体任务。为了让它更合理、更详细，并且贴合之前轻松自然的风格，我们可以将它组织成一个更具逻辑性和学习路径的“计算机视觉探索之旅”大纲。

我们将围绕**图像的本质**、**图像的基本操作**、**传统算法的“兵器库”** 以及 **经典任务的挑战与应对** 这几大主题展开。

---

## 计算机视觉的传统算法与机器学习应用：一次深度探索

### 引言：从像素到“看懂世界”

* **什么是计算机视觉？** 让计算机像人一样“看”和“理解”图像与视频。
* **为什么学习传统算法？** 它们是深度学习的基石，很多核心思想源于此；在特定场景仍有优势；帮助我们理解计算机视觉的演进。
* [**搭建环境**](./doc/cv_env_requirement.md)
---

### 第一章：图像的本质与基本操作

这部分是计算机视觉的“内功心法”，理解图像是如何被计算机“感知”和“处理”的至关重要。

#### 1.1 图像数据结构：图片在计算机里长啥样？

* **像素 (Pixel)：** 图像的最小单位，就像构成图片的“小砖块”。
    * **灰度图像：** 只有亮度信息，每个像素一个值 (0-255)。
    * **彩色图像 (RGB)：** 红、绿、蓝三通道，每个像素三个值。
    * **图像矩阵表示：** 如何将图像想象成一个充满数字的矩阵。
* **图像通道与深度：** 除了 RGB，还有哪些通道？什么是图像深度？
* **图像分辨率：** 图像的大小，决定了图像的细节程度。

#### 1.2 图片基本操作：玩转图像的“基础技能”

* **读取图片文件：** 如何把硬盘里的图片加载到程序中？
    * 常用库介绍（例如：OpenCV 的 `imread()`）。
* **显示图片：** 让计算机“看”到的图像呈现在屏幕上。
    * 常用库介绍（例如：OpenCV 的 `imshow()`）。
* **图片存储：** 如何把处理好的图片保存起来？
    * 不同格式的存储（例如：`imwrite()`）。
* **获取和修改像素值：** 直接操作图像的“细胞”。
* **图像属性查询：** 宽高、通道数、数据类型等。

---

### 第二章：图像几何变换与像素操作

掌握这些技巧，就能让图像“动”起来，或者改变它的“形态”。

#### 2.1 图像几何变换：让图片“变形金刚”

* **平移 (Translation)：** 图像的“搬家”操作，指定方向和距离。
    * 原理：仿射变换矩阵的构建。
* **旋转 (Rotation)：** 图像的“转圈”操作，指定角度和中心点。
    * 原理：旋转矩阵的构建，旋转中心的选择。
* **缩放操作 (Scaling)：** 图像的“放大镜”和“缩小镜”，指定缩放比例。
    * 原理：插值算法（最近邻、双线性、双三次）。
* **翻转 (Flipping)：** 图像的“照镜子”，水平或垂直翻转。
* **仿射变换 (Affine Transformation)：** 平移、旋转、缩放、剪切等操作的组合。
    * 原理：2x3 变换矩阵。
* **透视变换 (Perspective Transformation)：** 模拟三维投影，让图像更具立体感。
    * 原理：3x3 变换矩阵。

#### 2.2 图像格式转换与像素级特效

* **图像格式转换：** 不同图片格式的相互转化。
    * BMP, JPEG, PNG 等格式的特点与应用场景。
* **色彩空间转换：** RGB 到灰度，RGB 到 HSV 等，为什么需要这些转换？
    * HSV 空间的优势（颜色、饱和度、亮度分离）。
* **图像特效：** 给图片添加一些“艺术效果”。
    * **马赛克 (Mosaic)：** 图像的“模糊”艺术，如何实现？
    * **图片融合与叠加：** 将多张图片巧妙地结合在一起。
    * **图像锐化与模糊：** 通过卷积操作改变图像的清晰度。

---

### 第三章：传统机器学习算法在计算机视觉中的应用

这是传统计算机视觉的“核心竞争力”，理解它们如何从图像中提取“有意义”的信息。

#### 3.1 图像滤波与增强：让图像更“清晰”

* **高斯模糊 (Gaussian Blur)：** 经典的平滑滤波，去除噪声。
* **中值滤波 (Median Filter)：** 对椒盐噪声特别有效。
* **边缘检测 (Edge Detection)：** 找出图像中的“轮廓线”。
    * **Sobel、Prewitt、Roberts 算子：** 基础的边缘检测。
    * **Canny 边缘检测：** 经典的边缘检测算法，效果更优。
* **直方图均衡化：** 增强图像对比度。

#### 3.2 图像特征点检测：图像的“指纹”

* **角点检测 (Corner Detection)：** 图像中局部变化剧烈的点，就像图像的“路标”。
    * **Harris 角点检测：** 经典的角点检测算法，原理与应用。
* **尺度不变特征变换 (SIFT)：** 具有尺度和旋转不变性的局部特征点，被称为图像的“瑞士军刀”。
    * 原理：高斯差分、关键点定位、方向分配、描述符生成。
    * 应用：图像匹配、物体识别。
* **加速稳健特征 (SURF)：** SIFT 的加速版。
* **ORB (Oriented FAST and Rotated BRIEF)：** 结合了 FAST 和 BRIEF 的特征点检测与描述算法，速度更快。
* **特征点匹配：** 如何在不同图片中找到相同的特征点？
    * 暴力匹配、FLANN 匹配。

#### 3.3 特征描述符与分类器：从“特征”到“识别”

* **HOG (Histograms of Oriented Gradients)：** 描述图像局部形状的特征，常用于行人检测。
* **LBP (Local Binary Patterns)：** 描述图像纹理的特征，常用于人脸识别和纹理分类。
* **传统分类器：**
    * **支持向量机 (SVM - Support Vector Machine)：** 找到最优超平面进行分类。
    * **K 近邻 (K-NN - K-Nearest Neighbors)：** “物以类聚，人以群分”。
    * **Adaboost：** 集成学习的经典，常用于人脸检测。

---

### 第四章：传统计算机视觉经典任务与挑战

这部分将传统算法与具体任务结合起来，展示它们在解决实际问题时的思路。

#### 4.1 识图 (Image Recognition / Object Recognition)：识别图片中有啥？

* **基于特征匹配的识图：** 利用 SIFT、SURF 等特征匹配技术识别特定物体。
    * 案例：识别特定品牌的 logo。
* **基于分类器的识图：** 结合手工特征和分类器实现图像分类。
    * 传统图像分类流程：特征提取 -> 分类器训练 -> 分类。

#### 4.2 文字识别 (OCR - Optical Character Recognition)：让计算机“读懂”文字

* **传统 OCR 流程：**
    * 图像预处理：去噪、二值化。
    * 字符分割：将文字区域分割成单个字符。
    * 特征提取：提取单个字符的特征。
    * 分类器识别：识别字符。
* **Tesseract OCR：** 著名的开源 OCR 引擎，其早期版本就大量使用了传统方法。

#### 4.3 车牌识别 (License Plate Recognition)：路上的“火眼金睛”

* **传统车牌识别流程：**
    * **车牌定位 (license\_plate\_location)：** 在图像中找到车牌区域（颜色、形状、边缘特征）。
    * **车牌倾斜校正：** 矫正倾斜的车牌。
    * **字符分割：** 将车牌上的字符分开。
    * **字符识别 (license\_plate\_recognition)：** 识别每个字符。
* **传统算法在车牌识别中的应用：** 边缘检测、形态学操作、连通域分析等。

#### 4.4 人脸识别 (Face Recognition)：我是谁？

* **人脸检测 (Face Detection)：** 找出图片中的人脸。
    * **Viola-Jones 算法：** 早期经典的人脸检测算法，结合 Haar 特征和 Adaboost 级联分类器。
* **人脸对齐 (Face Alignment)：** 将人脸标准化，方便后续识别。
* **人脸特征提取：** 提取人脸的独特特征。
    * **PCA (Principal Component Analysis)：** 主成分分析，用于降维和特征提取（例如：特征脸 Eigenfaces）。
    * **LDA (Linear Discriminant Analysis)：** 线性判别分析，用于最大化类间距离。
* **人脸识别：** 基于提取的特征进行比对和识别。

---

### 第五章：传统算法的“余晖”与未来展望

* **传统算法的优势与局限性：**
    * 优势：小数据量、计算资源有限、可解释性强。
    * 局限性：特征设计困难、鲁棒性差、泛化能力弱。
* **传统算法与深度学习的融合：**
    * 作为深度学习的预处理/后处理。
    * 传统算法思想在深度学习中的体现（例如：卷积、注意力机制）。
* **总结：** 为什么了解传统算法对计算机视觉学习者至关重要。

---

### 拓展思考：

* 在你看来，哪些传统算法的思想在当今的深度学习模型中依然发挥着作用？
* 未来，传统算法和深度学习会在计算机视觉领域如何更好地协同工作？

希望这份更详细的大纲能帮助你更好地理解计算机视觉的传统算法世界！

### 传统算法的“余晖”与新应用

你可能会想，既然深度学习这么强，那传统算法是不是就彻底没用了？其实不然！在某些特定场景下，传统算法依然有着自己的优势：

* **数据量小：** 深度学习模型通常需要大量数据才能训练出好的效果，如果数据量非常小，传统算法可能表现得更好，或者作为辅助手段。
* **计算资源有限：** 深度学习模型通常计算量大，对硬件要求高。在一些资源受限的嵌入式设备上，轻量级的传统算法可能更具优势。
* **可解释性强：** 很多传统算法的原理相对直观，更容易理解其决策过程，这在一些需要高可解释性的场景（如医疗、金融）中非常重要。
* **特定任务的先验知识：** 在某些特定任务中，传统算法可以融入领域专家知识，从而取得更好的效果。比如，在一些工业检测中，针对特定缺陷，可能通过传统图像处理方法更容易识别。
* **作为深度学习的“预处理”或“后处理”：** 传统图像处理技术常常被用作深度学习模型的**预处理**（如图像增强、去噪）或**后处理**（如分割结果的平滑）。

---

### 总结一下

虽然 Transformer 和各种深度学习模型在计算机视觉领域大放异彩，但了解**传统机器学习算法**的重要性不容忽视。它们不仅为深度学习提供了宝贵的**思想源泉**，在某些特定场景下也依然具有**独特的应用价值**。对于计算机视觉的入门者来说，花点时间了解这些“前辈”算法，绝对能让你对整个领域有更全面、更深刻的理解。毕竟，站在巨人的肩膀上，才能看得更远嘛！

你觉得在未来的计算机视觉领域，传统算法和深度学习模型会是怎样的关系呢？是深度学习完全取代传统算法，还是会走向更深度的融合？





```
#coding=utf-8
import cv2
import numpy as np
```
为了能够在文件中写中文注释我们需要声明``` #coding=utf-8 ```，而且当然 numpy 也是不可缺少一个库

6. 创建一个模板文件
```python
#coding=utf-8
import cv2
import numpy as np
import argparse

class App():
    def run(self,img_name):
        dir = 'images/'
        img = cv2.imread( dir +  img_name + '.jpg')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        while(1):
            cv2.imshow("img",gray)
            key = cv2.waitKey(1)
            if key == 27:
                break
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("imagename", help="display image name", type=str)
    args = parser.parse_args()
    
    # 
    App().run(args.imagename)
    cv2.destroyAllWindows()
```

## 基础

1. 基础知识
    - 
    - 图像数据结构
2. 图片基本操作
    - 读取图片文件
    - 显示图片
    - 图片存储
3. 图像几何变换
    - 平移
    - 旋转
    - 缩放操作
4. 图像格式转换

5. 图像特效
    - 马赛克   
6. 图像特征点检测
    - harris
    - SIFT
7. 识图
    - 文字识别
    - 车牌识别(license_plate_)
    - 人脸识别

## 高级
1. 增强现实
2. 重建三维场景
    - 针孔照相机模型
    - 图像到图像的映射
3. 机器学习

## 1.基础知识
### 1.1 色彩空间
- **RGB**
我们看到色彩是由 RGB(R 代表红色 G 代表绿色 B代 表蓝色) 混合称为各种各样丰富色彩，所以我们用 3 通道(RGB 通道的值)分别代表某一点像素的值。这是一种图像表示方式
- **GRAY**
GRAY 就是只有灰度值一个 channel，来表示图片信息。这是我们对图片分析提取信息主要一种数据。我们根据灰度的变化来进行角点检测，边缘检测和模糊处理。
- **HSV**
在 HSV 中，H 参数表示色彩信息，即所处的光谱颜色的位置。该参数用一角度量来表示，红、绿、蓝分别纯度 S 为一比例值，范围从 0 到 1，表示成所选颜色的纯度和该颜色最大的纯度之间的比率。S= 0 时，只有灰度。相隔120度。互补色分别相差180度。
![hsv](https://upload-images.jianshu.io/upload_images/8207483-2f4e37d348aa8242.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![svh01](https://upload-images.jianshu.io/upload_images/8207483-908dca49506848d2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们通过在空间 3 个维度来表示 SHV
- LAB
在Lab颜色空间中，一种颜色由L（明度）、a颜色、b颜色三种参数表征．在一幅图像中，每一个像素有对应的Lab值．一幅图像就有对应的L通道、a通道和b通道．在Lab中，明度和颜色是分开的， L通道没有颜色，a通道和b通道只有颜色．不像在RGB颜色空间中，R通道、G通道、B通道每一个既包含有明度又包含有颜色． L取值为0--100(纯黑--纯白)、a取值为+127--128(洋红--绿)、b取值为+127--128(黄--蓝)．正为暖色，负为冷色．
#### 色彩空间转换
在 opencv 中我们主要通过 ```cvtColor```这个方法进行将图片在各种不同色彩空间进行转换
```
 cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
```
- img 是我们

### 文字识别
安装 tesseract-ocr
```
brew install tesseract-ocr
```
安装完之后我们就可以用命令行来使用工具来识别图片文字提取出来
```
tesseract lenet.png output
```

## 图像几何变换
图片变换的本质就是对图片的矩阵进行操作
### 图片平移
矩阵仿射变换，是一系列变换的基础

### 图片缩放
图片缩放就是将图片宽高信息进行改变
- 加载图片
- 修改大小
- 放大 缩小 等比例缩放


```python
def resize(img,param):
    height,width,_ = get_img_info(img)
    dstHeight = int(height * param)
    dstWidth = int(height * param)
    dst = cv2.resize(img,(dstWidth,dstHeight))
    return dst

def get_img_info(img):
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    mode = imgInfo[2]
    return height, width, mode
```


```python
img = cv2.imread('empire.jpeg',1)

dst = utils.resize(img,0.5)
cv2.imshow('resize img',dst)
cv2.waitKey(0)
```

#### 在 opencv 中提供的有关缩放的方法
- 最近临域插值
对于缩放后像素位置 x y 如果是小数，可以按临近进行取整
- 双线性插值
当我们进行缩放后得到点的 x y 都是小数，这次在双线插值法不是取最近点，而是按比例进行从周围 4 个点进行取值。根据该点距离其周围 4 点的比例来将周围 4 点像素值作为目标点。
- 像素关系重采样
- 立方插值


```python
def resize_nearby(img,scale):
    height,width,_ = get_img_info(img)
    dstHeight = int(height/scale)
    dstWidth = int(height/scale)
    # create empty
    dstImage = np.zeros((dstHeight,dstWidth,3),np.uint8)
    for i in range(0,dstHeight):# row
        for j in range(0,dstWidth):#column
            iNew = int(i*(height*1.0/dstHeight))
            jNew = int(j*(width*1.0)/dstWidth)
            dstImage[i,j] = img[iNew,jNew]
    return dstImage
```

通过变换矩阵实现图片缩放


```python
def resize_with_mat(img,scale):
    height,width,_ = get_img_info(img)
    # tranform array
    matTransform = np.float32([[scale,0,0],[0,scale,0]]) # 2 * 3
    dst = cv2.warpAffine(img,matTransform,(int(width*scale),int(height*scale)) )
    return dst
```

### 图片剪切
其本质是对数据进行操作，x 表示列信息 y 表示行信息。


```python
def img_crop(img,startX,startY,h,w):
    height,width,_ = get_img_info(img)
    print(startX,(startX+h),startY,startY + w)
    dst = img[startX:(startX+w),startY:(startY + h)]
    # dst = img[100:200,100:300]
    return dst
```

### 图片镜像


```python
def img_mirror_v(img):
    height,width,deep = get_img_info(img)
    dstInfo = (height*2,width,deep)
    dst = np.zeros(dstInfo,np.uint8)
    for i in range(0,height):
        for j in range(0,width):
            dst[i,j] = img[i,j]
            # x y 
            dst[height*2-i-1,j] = img[i,j]
    for i in range(0,width):
        dst[height,i] = (0,0,255) #BGR
    return dst
```

### 图片位移
了解 opencv 提供有关位移的 API,
这里 matTransform 就是偏移矩阵，由该矩阵来控制图片的位置
$$\begin{bmatrix}
   1 & 0 & 100 \
   0 & 1 & 200 
  \end{bmatrix}$$<br>
[1,0,100],[0,1,200] 对该矩阵进行拆分为[[1,0],[0,1]](A)和[[100],[200]](B)
xy 为 C 矩阵
A*C + B = [[1*x + 0*y],[0*x + 1*y]] + [[100],[200]] = [[x+100],[y+200]]



```python
def img_transform(img):
    height,width,_ = get_img_info(img)
    # tranform array
    matTransform = np.float32([[1,0,100],[0,1,200]]) # 2 * 3
    dst = cv2.warpAffine(img,matTransform,(width,height))
    return dst
```

### 仿射变换
所谓仿射变换就是线性变换将平移，通过矩阵来控制 2D 图像，
#### 直线变换
- 变换前是直线的，变换后依然是直线
- 直线比例保持不变
- 变换前是原点的，变换后依然是原点

这里通过 左上角、左下角和右上角


```python
def img_transform(img):
    height,width,deep = get_img_info(img)
    matSrc = np.float32([[0,0],[0,height-1],[width-1,0]])
    matDst = np.float32([[50,50],[300,height-200],[width-300,100]])
    # combination
    matAffine = cv2.getAffineTransform(matSrc,matDst) # mat
    dst = cv2.warpAffine(img,matAffine,(width,height))
    return dst
```

### 图片旋转
有关图像旋转这里我们使用 opencv 提供 getRotationMatrix2D 来获得一个矩阵来控制旋转，getRotationMatrix2D 接受三个参数分别为，
1. 旋转的中心点
2. 旋转的角度
3. 缩放的比例


```python
def img_rotate(img):
    height,width,deep = get_img_info(img)

    # mat rotate 1 center 2 angle 3 src
    matRotate = cv2.getRotationMatrix2D((height*0.5,width*0.5),45,0.5)
    dst = cv2.warpAffine(img,matRotate,(width,height))
    return dst
```

## 图像特效 
本次内容重点是如何使用滤镜为图片添加特效，以及这些滤镜背后算法。

### 灰度处理
灰度处理算法非常简单但是这不影响他重要地位。是很多图像处理基础，是我们对物体识别，边缘检测这些高级的计算机视觉处理都是基于灰度处理，这里介绍两个公式用于处理图片为灰度图 
- 第一个是取 RGB 的并均值来计算灰度值 Gray = (R+G+B)/3 
- 第二个是根据心里感受给出经验公式 Gray = R*0.299 + G*0.587 + B*0.114


```python
# method 1
img = cv2.imread('empire.jpeg',1)
# method 2
# 颜色空间的转换
dst = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
```


```python
def gray_process(img):
    height,width,deep = get_img_info(img)
    dst = np.zeros((height,width,3),np.uint8)
    for i in range(0,height):
        for j in range(0,width):
            (b,g,r) = img[i,j]
            gray = (int(b) + int(g) + int(r))/3
            dst[i,j] = np.uint8(gray)
    return dst

def gray_process_2(img):
    height,width,deep = get_img_info(img)
    dst = np.zeros((height,width,3),np.uint8)
    for i in range(0,height):
        for j in range(0,width):
            (b,g,r) = img[i,j]
            gray = int(b)*0.299 + int(g)*0.587 + int(r)*0.114
            dst[i,j] = np.uint8(gray)
    return dst
```

#### 算法优化 
加法运算优于乘法，移位运算要优于加法运算,定点运算优于浮点运算


```python
def gray_process_3(img):
    height,width,deep = get_img_info(img)
    dst = np.zeros((height,width,3),np.uint8)
    for i in range(0,height):
        for j in range(0,width):
            (b,g,r) = img[i,j]
            b = int(b)
            g = int(g)
            r = int(r)
            gray = (r + (g<<1) + b)>>2
            # gray = (r*1 + g*2 + b*1)/4

            dst[i,j] = np.uint8(gray)
    return dst
```

### 底板效果
这里底板效果，如果大家小时候去照相馆照相都会得到底板，灰度底板和彩色底板


```python
def color_invert(img):
    # img = gray_process_2(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    width = gray.shape[1]
    dst = np.zeros((height,width,1),np.uint8)
    for i in range(0,height):
        for j in range(0,width):
            grayPixel = gray[i,j]
            dst[i,j] = 255 - grayPixel
    return dst
```


```python
def color_invert(img):
    height,width,deep = get_img_info(img)
    dst = np.zeros((height,width,deep),np.uint8)
    for i in range(0,height):
        for j in range(0,width):
            (b,g,r) = img[i,j]
            dst[i,j] = (255-b,255-g,255-r)
    return dst
```

### 马赛克
```python
def mosaic_process(img):
    height,width,deep = get_img_info(img)
    print(height,width)
    # dst = np.zeros((height,width,deep),np.uint8)
    for m in range(100,500):
        for n in range(100,500):
            # pixel -> 10 * 10
            if m%20 == 0 and n%20 == 0:
                for i in range(0,20):
                    for j in range(0,20):
                        (b,g,r) = img[m,n]
                        img[i+m,j+n] = (b,g,r) 

    return img
```

### 毛玻璃效果
所有像素每一个像素都一个随机像素


```python
def rough_glass_effects_process(img):
    height,width,deep = get_img_info(img)
    dst = np.zeros((height,width,deep),np.uint8)
    mm = 8
    for m in range(0,height - mm):
        for n in range(0,width - mm):
            index = int(random.random()*8)
            (b,g,r) = img[m+index,n+index]
            dst[m,n] = (b,g,r)
    return dst

```

### 图像融合
通过两张图片融合的权重来控制每一个张图片在融合所占的比例。这里需要注意一点就是，我们这里的融合区域（也就是所谓 ROI 关注点区域）一定要小于两张融合图片中任何一张图片的尺寸。


```python
def merge_process(img1,img2):
    h1,w1,d1 = get_img_info(img1)
    h2,w2,d2 = get_img_info(img2)
    # ROI
    roiH = h2
    print(roiH)
    roiW = w2
    print(roiW)
    img1ROI = img1[0:roiH,0:roiW]
    img2ROI = img2[0:roiH,0:roiW]

    dst = np.zeros((roiH,roiW,3),np.uint8)
    dst = cv2.addWeighted(img1ROI,0.5,img2ROI,0.5,0)
    return dst
```

### 边缘检测
边缘检测实质就是卷积运算，先用 opencv 提供 API 实现边缘检测，然后gen'j
- 灰度处理
- 高斯滤波
- canny 边缘检测


```python
def edge_detect(img):
    # height,width,deep = get_img_info(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # height = gray.shape[0]
    # width = gray.shape[1]
    dst = cv2.GaussianBlur(gray,(3,3),0)
    dst = cv2.Canny(dst,50,50)
    return dst
```

保存图像细节信息，卷积运算，
算子模板、图片卷积、阈值筛选
sobel 算子
x [[1,2,1],[0,0,0],[-1,-2,0]](水平）
y [[1,0,-1],[2,0,-2],[1,0,-1]]（竖直）

矩阵是行列相乘，算子算法，对应元素相乘后相加运算 dst 运算结构就是梯度
sqrt(a*a+b*b) = f > thred 如果大于 thre 就是边缘否则不是边缘

### 浮雕效果
在边缘检测基础加上
newp = gray0 - gray1 + 150


```python
def emboss_effects_process(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    width = gray.shape[1]
    dst = np.zeros((height,width,1),np.uint8)
    for i in range(0,height):
        for j in range(0,width - 1):
            p0 = int(gray[i,j])
            p1 = int(gray[i,j+1])
            newP = p0 - p1 + 150
            if newP > 255:
                newP = 255
            if newP < 0:
                newP = 0

            dst[i,j] = newP
    return dst
```

### 颜色风格
### 亮度增量


```python
def increntment_brigtness(img):
    height,width,deep = get_img_info(img)
    dst = np.zeros((height,width,3),np.uint8)
    for i in range(0,height):
        for j in range(0,width):
            (b,g,r) = img[i,j]
            bb = int(b) + 40
            gg = int(g) + 40
            rr = int(r) + 40

            if bb > 255:
                bb = 255
            if gg > 255:
                gg = 255
            if rr > 255:
                rr = 255

            dst[i,j] = (bb,gg,rr)
    return dst
```

1. 彩色到灰度转化，
2. 
3. 
4. 灰度等级最多的像素
5. 用统计出来平均值来代替值


```python

```

## 形状绘制

### 图片旋转


## 图像美化
### 直方图
出现的概率，什么均衡化，统计每一个像素灰度出现的概率 0-255
### 图片修补

### 亮度增强
### 图片滤波
### 一维高斯滤波
### 二维高斯滤波
### 直方图均衡化
### 磨皮美白
### 

# 机器学习

- 如何准备样本
- 提取特征
- 分类
- 预测以及检验
## 视频与图片的分解合成（获取样本）
## Hog特征原理

什么是 hog 特征，某个像素进行某种运算得到结果，hog 在模板运算基础上进行。
1. 模块划分
2. 根据 hog 模板进行计算梯度和方向
3. 根据 hog 梯度和方向 bin 方向进行投影
4. 每个模块hog
- Window 窗口（size)
- block
- cell
- image
image > window > block > cell

## SVM 支持向量机

通过一条线来完成分类，可以使用任意多条线来进行分类。如果将直线投射到空间上，直线就变成了超平面。本质是寻求一个最优的超平面。SVM 直线。解决是分类问题，解决方式是通过一个超平面来进行分类，SVM 的核，SVM 支持很多核。正样本和负样本的数量不一定相等，每一个数据都有标签，所以 SVM 应该属于监督学习。
- svm 核：line
- 身高体重 训练过程 预测


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# data
rand1 = np.array([[152,46],[159,50],[165,55],[168,55],[172,60]])
rand2 = np.array([[152,54],[156,55],[160,56],[172,65],[178,68]])
# label
label = np.array([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1]])

data = np.vstack((rand1,rand2))
data = np.array(data,dtype='float32')
# create svm
svm = cv2.ml.SVM_create()
# set type
svm.setType(cv2.ml.SVM_C_SVC)
# line
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(0.01)
# train
result = svm.train(data,cv2.ml.ROW_SAMPLE,label)
# test
# [0,1]
pt_data = np.vstack([[167,55],[162,57]])
pt_data = np.array(pt_data,dtype='float32')
print(pt_data)
(par1,par2) = svm.predict(pt_data)
print(par1,par2)
```


```python
[[167.  55.]
 [162.  57.]]
(0.0, array([[0.],
       [1.]], dtype=float32))
```

### 图像二值化
图像只有两种颜色黑和白，分别用 0 和 1 表示，用二值化可以对图片进行压缩，也可以用于前景和背景进行分离。
#### 图像二值方法
- 三角阈值二值化
- 全局阈值
- 局部阈值
#### openCV 相关 API 使用
- OTSU
- Triangle
- 自动与手动
#### 自适应阈值

## 特征
### 测量特征
这里提取特征，不是识别出是什么物体，这里需要说一下，这里我们所说的特征并不是机器学习用于分类的识别特征。
例如，我们需要通过识别图像的特征来识别一些标志性旅游点和建筑物。更进一步说，不但要识别还要定位（导航）出现在位置，这也就是 SLAM 了。还有今天会介绍全景图，现在手机都提供了全景图效果。我们可以通过检测两张图片的共同特征点然后根据这些相同特征点进行拼接取图片
### 局部特征
在以后 3D 重建和照相机推导中都会用到局部特征点检测，随后会介绍 Harris 和 SIFT 特征点检测。
### 
## Haars特征原理
### 角点的用途
我们需要用一些特征来表述图片，角点作为图片的一个特征，可以用于描述图片中一些位置信息，尤其对于图像中的物体定位特别有用。
### 如何识别角点
那么首先我们需要区分边和角，并行于边方向变化像素灰度变化不大，垂直方向像素变化大。那么相对于边角点在各个方向上都变化就认为是边。

### 公式推导
在推导过程中我们会用到高等数学的知识，
$$ E(u,v) = \sum_{x,y}w(x,y)[I(x+u,y+v) - I(x,y)]^2$$
- w 我们暂时忽略考虑 w，在窗口 w 内的值全部都为 1 而窗口外面的值都 0
- UV u 表示向竖直方向移动一个,v 表示水平单位
- 我们将窗口移动一定距离后的窗口内 x y（x，y 表示窗口内相对于窗口的位置，然后用窗口移动后位置(x+u,y+v) 移动前位置信息。从而得到窗口的变化。然后我们在对这些差的平方进行求和从而得到像素在一定范围内变化率。
所以我们知道角点检测是角点位置明暗（灰度变化）无论是在水平还是竖直方向的变化都要大，这些变化率越大越好。
### 泰勒展开
$$ f(x+u,y+v) = f(x,y) + uf_{x}(x,y) + vf_y(x,y) + $$
$$ \frac{1}{2!}[u_{2}f_{xx}(x,y) + uvf_{xy}x,y+ v^2f_{yy}(x,y)] $$
$$ \frac{1}{3!}[u^3f_{xxx}(x,y) + u_{2}f_{xx}(x,y) + uvf_{xy}x,y+ v^2f_{yy}(x,y) + v^3f_{yyy}(x,y)] $$

$$f(x+u,y+v) \approx f(x,y) + uf_{x}(x,y) + vf_y(x,y)$$


```python

```

像素经过运算得到某一个结果，可以是具体值也可以是向量。第一种情况可能是具体值或者是向量或者是多维矩阵。特征的本质是像素运算结果。<br>
如果利用特征来区分目标，根据阈值进行判定是否为目标。通过简单的方法来使用特征
一些模板 Basic core ALL
- 特征 = 白色 - 黑色
- 特征 = 整个区域*权重 + 黑色*权重
- (p1-p2-p3)

Haar 特征遍历过程，可以根据face 大小
积分图
利用积分图进行快速计算
## SVM分类原理
## Adaboost分类原理
强分类级联、弱分类器

## Hog+SVM检测
SVM 支持向量机
Hog 特征计算
一系列的窗体，同时计算在每一个 cell 计算梯度和赋值方向。



```python
fileName = 'pos\\' + str(i+1) + '.jpg'
# 使用反斜杠获取图片

svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(0.01)
# train
svm.train(featureArray,cv2.ml.ROW_SAMPLE,labelArray)
# mHog -》mDetect -》resultArray rho （通过 svm 训练得到）

alpha = np.zeros((1),np.float32)
# hog 描述信息 rho 需要 svm
rho = svm.getDecisionFunction(0,alpha)
print(alpha)
print(rho)
alphaArray = np.zeros((1,1),np.float32)
# 作为参数使用
supportVArray = np.zeros((1,featureNum),np.float32)
resultArray = np.zeros((1,featureNum),np.float32)
alphaArray[0,0] = alpha
resultArray = -1 * alphaArray * supportVArray

# 创建 detect
mDetect = np.zeros((3781),np.float32)
for i in range(0,3780):
    mDetect[i] = resultArray[0,i]
mDetect[3780] = rho[0]
# create hog
mHog = cv2.HOGDescriptor()
mHog.setSVMDetector(mDetect)

imageSrc = cv2.imread('Test2.jpg',1)
# 8x8 窗口滑动步长，32 32 为窗体大小
objs = mHog.detectMultiScale(imageSrc,0,(8,8),(32,32),1.05,2)
# x y w h 是三维信息，参数放到三维最后一维
x = (int)(objs[0][0][0])
y = (int)(objs[0][0][1])
w = (int)(objs[0][0][2])
h = (int)(objs[0][0][3])

# 绘制和展示
cv2.rectangle(imageSrc,(x,y),(x + w,y + h),(255,0,0),2)
while(1):
    # channels = cv2.split(img1) # RGB R G B
    # for i in range(0,3):
    #     utils.image_hist(channels[i],31+i)
    # cv2.imshow('girl src',girl)
    cv2.imshow('girl dist',imageSrc)
    k = cv2.waitKey(33)
    if k == 27:
        break
    elif k==-1:
        continue
    else:
        print(k) 
    
```

## knn 邻域
根本原理 knn test 样本 k 个存在，K neights
K 最近的邻居，如果我们又两个特征

```python
('p10[]', array([1, 5, 0, 2, 6]))
('expect', array([2, 5, 0, 2, 6]))
```

### 图像导数
在图中有关图像强弱的变换情况是非常重要的信息。强度变化可以用灰度图像I的x和y方向导数$I_x$和$I_y$进行描述

## 图像去燥

## ffmpeg

```python
ffmpeg -i war.mp4 war.avi
```

# 图像到图像的映射

$$\begin{bmatrix}
   h_{00} & h_{01} & h_{02} \
   h_{10} & h_{11} & h_{12} \
   h_{20} & h_{21} & h_{22}
  \end{bmatrix}$$<br>


## 局部图像描述子

### Harris 角点检测
- 角点是在面与面交界处和线与线交接位置就是**角**
#### 角点具有哪些特点
- 轮廓之间的交点
- 对于同一场景，在视角发生变化时候，具有稳定性的点
- 在角点附近的点无论在梯度方向和梯度幅度上有明显的变化的点
#### 角点用途
基于上面所述角点特点角点可以用于定位
#### 如何识别角点
对我们认为不是问题，对于计算机并不一定不是问题。我们需要分析我们平时一些不以为然行为背后的原理，然后让计算机来实现原理达到和人类同样能力。
我们来思考一下边判断(也就是计算机如何识别一条直线),可以表示为直线位置在沿着直线方向灰度变化不明显，而在垂直于直线方向上灰度变化是明显的。
那么对于角，在竖直和水平方向灰度变化都很大。

在计算机视觉有关角点检测被广泛应用，例如图像匹配、相机定位等。表示图像的特点，角点可以用于定位，这是因为角点包含大量位置信息。要识别角点，我们首先需要了解角点有哪些特征。对于边在沿着边是灰度变化不大。有关 Harris 角点公式是推导，计算时候我们需要计算一点邻域

#### 判断是角点的基本原理
算法基本思想是使用一个固定大小的窗口(矩阵)在图像（矩阵）上进行任意方向上的滑动，比较滑动前与滑动后两种情况，窗口中的像素灰度变化程度，如果存在任意方向上的滑动，都有着较大灰度变化，那么我们可以认为该窗口中存在角点.。

![Harris_corners](https://upload-images.jianshu.io/upload_images/8207483-b26fccf3ec6257a9.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

通过上面图我们不难看出如何通过某个点的邻域来判断该点是面上的点，还是边上的点或是角点。
- 左图中因为点是在面上，所以邻域中所有方向没有明显梯度变化
- 在中间中，图因为点是在线上的点所以在某个方向上有明显梯度变化
- 在右图中，因为点是在角点上，所以邻域中所有方向都有明显梯度变化

$$ E(u,v) = \sum_{x,y} W(x,y) {[I(x+u,y+v) - I(x,y)]^2} $$
- $E(u,v) $u,v 表示窗口位置，
- x,y 
- w(x,y)是窗口函数，输入 x 矩阵，输出也是矩阵。最简单情形就是窗口内的所有像素所对应的 w 权重系数均为 1，窗口外面全部取 0。
- 这个公式反应当窗口移动时，窗口内亮度的变化。   

- 这里 $\sum $ 大家就可以知道这个求和所以返回的是一个值。
我们需要计算变化，$[I(x+u,y+v) - I(x,y)]^2$ ，x,y 表示像素的位置 I(x,y) 表示在 x 和 y 坐标像素的值
可以将 x 们会将 w(x,y) 函数设定为以窗口中心为原点的二元正态分布。如果窗口中心点是角点时，移动前与移动后，该点的灰度变化应该最为剧烈，所以该点权重系数可以设定大些，表示窗口移动时，该点在灰度变化贡献较大；而离窗口中心(角点)较远的点，这些点的灰度变化几近平缓，这些点的权重系数，可以设定小点，以示该点对灰度变化贡献较小，那么我们自然想到使用二元高斯函数来表示窗口函数，这里仅是个人理解，大家可以参考下。
#### 泰勒展开
我们公式推导会用到泰勒公式，这里我们先简单了解，不做过多详细的介绍了。
$$ f(x+u,y+v) = f(x,y) + uf_x(x,y) + vf_y(x,y) +  $$
$$ \frac{1}{2!}[ u^2 f_{xx}(x,y) + uvf_{xy}(x,y) + v^2f_{yy}(x,y)]$$
$$ \frac{1}{3!} [u^3f_{xxx}(x,y) + u^2vf_{xxy}(x,y) + uv^2f_{xyy}(x,y) + v^2f_{yy}(x,y) ]$$
我们这里只用一阶泰勒公式  
$$ f(x+u,y+v)  \approx f(x,y) + uf_x(x,y) + vf_y(x,y) $$

$$ \sum [I(x+u,y+v) - I(x,y)]^2 $$
$$ \sum [I(x,y) + uI_x + vI_y - I(x,y)]^2 $$
$$ \sum u^2I_x^2 + 2uvI_xI_y + v^2I_y^2$$
$$ \sum \begin{matrix}
    [ u & v ]
\end{matrix} 
    \left[ 
    \begin{matrix}
        I_x^2 & I_xI_y \\ 
    I_xI_y  & I_y^2 
\end{matrix} \right]  \left[ 
    \begin{matrix}
        u \\ 
        v
\end{matrix} \right] $$

$$  \begin{matrix}
    [ u & v ]
\end{matrix} \sum \left(
    \left[ 
    \begin{matrix}
        I_x^2 & I_xI_y \\ 
    I_xI_y  & I_y^2 
\end{matrix} \right]  \right) \left[ 
    \begin{matrix}
        u \\ 
        v
\end{matrix} \right] $$

# 照相机模型与增强现实
## 照相机的结构
物体每一个点发出光线都会投射到平面（胶片）上任意位置，这样最终胶片是无法成像的。
针孔照相机，这里针孔不是偷拍的针孔照相机，而是一个经典的照相机模型。通过物体和胶片间的障碍物上小孔，可以保证物体上每一个点都对应胶片上的一点。这个小孔就是我们熟悉**光圈**。其实小孔成像本质是将 3维压缩为 2维胶片上。从而也就丢失距离信息。角度信息的就是丢失，看到相互，并行关系被破坏，长度也发生变化了。所以我们需要通过学习将这些变形和错觉进行还原。当然我们用照相机都不是针孔相机，这是因为如果针孔过大，图像就会blur。如果孔太小那么问题需要很长时间曝光才能成像。可以用一个凸透镜来将到达镜面都聚到一点。凸透镜也会将点散开到小圆。 $$\frac{1}{d_0} + \frac{1}{d_1} = \frac{1}{f}$$
$d_0$ 为物体的到凸透镜距离，$d_1$ 表示成像平面（胶片）到凸透镜的距离。f 表示焦距。存在这么一个公式就可以满足在$d_0$ 位置的物体在 $d_1$ 平面上成像最佳（清晰）。
### 光圈
控制进光的量，光圈就是开孔大小来控制进入光，可以控制景深。一旦光圈开大的就会缩小清晰区域。光圈也就是（depth field)就会变大，一些物体的散射不会那么严重，有时候我们需要大光圈得到背景发虚。小光圈带来问题曝光时间。
f 对于定焦的focal length 是不变，我们只会调整 aperture。
小光圈也会影响快门时间。小光圈的曝光时间可能会引起运动模糊。
### 焦距
(Field of View)
同一位置相机用不同焦距，28mm Field of View 就变小，85mm 时候的 Field of view 也就是只有 28度视野，每一个物体在通常尺寸的胶片上像素也就是越多，
$$\phi = tan^{-1}( \frac{d}{2f})$$
尝试对对相机进行建模
## 针孔照相机
针孔摄像机是计算机视觉中广泛使用的照相机模型，这是因为模型简单且能够提供足够的精度。在针孔照相机模型中，在光线投射到图像平面之前，仅从一个点（孔）经过，也就是照相机模型中。
（图，针孔照相机模型）


### 照相机矩阵
P = K[R|t]
### 三维点的投影
## 颜色
颜色说简单也简单，说复杂也复杂，我们在高中物理已经知道可见光是电磁波，不同颜色对应不同波长的电磁波。我们人又是怎么分清颜色呢？自我们视网膜上有一个锥形细胞，这些细胞对光是敏感的，而且有红绿蓝三种锥形细胞对应三种不同波长的颜色。在人眼中除了锥形细胞以外还有柱状细胞，柱状希望功能是在环境比较昏暗情况下对外界光进行感知，不过这种细胞对颜色并不敏感。随意我们在黑暗中看到都是黑白的。其实颜色使我们人感知到东西。
### 反射
其实我们看到物体颜色，都是由于物体对不同波段光的反射不同而造成。如果物体对短波反射率比较低而长波反射率比较高那么这个物体在白光的照射下就呈现的蓝色。
### 成像流程
光源发射的光具有一定能量，然后光源的能量到达环境后，环境对光进行一定反射率。也就是通过一定运算得到一定能量波，这个波也就决定我们看到是什么样的颜色
### 数值颜色
我们需要对颜色的特征进行描述，用于语言无法精准地定位颜色的特征，所以我们需要用数字来描述颜色。**颜色匹配**以后我们用RGB（红绿蓝）的配比来准确地描述某一种颜色。因为人眼对光的感知是线性的。

## 图像分割
### 相关概念

### 概念
**图像分割**是图像预处理的重要步骤之一，主要目标是将图像划分为不同的区域。
### 图像分割的方法
- 基于阈值的分割
    基本原理：通过设定不同的**特征阈值**，把图像象素点分为若干类。
- 基于边缘的分割
- 基于区域的分割
**分水岭算法**是一种图像区域分割法，在分割的过程中，会把跟临近像素间的相似性作为重要的参考依据，从而将在空间位置上相近并且灰度值相近的像素点互相连接起来构成一个封闭的轮廓，**封闭性**是分水岭算法的一个重要特征。
### GrabCut 算法函数

## 像素的由来
我们一直然后图片变化，以及在视觉神经网络都是对图像一个一个像素进行操作和计算。例如一个像素值为 255 那么这个值是怎么来的。
在开始之前我们需要弄清一些概念
- 辐照度（ irradiance ）是单位时间单位面积的功率。 
- 辐射出射度（ radiant e ）是从单位时间内单位面积上辐射出的功率。

###相机成像的流程
物体反射光发射光辐射由 cmos 感光器后，还要考虑快门时间，快门时间长进入光子数量多（辐射出射度是单位时间），我们都知道光子是携带能量，结束能量后胶片感光颗粒感光率是和光子能量成为非线性的关系。

###数码相机成像流程
在数码相机中，是将光子能量转换为电流，手机代替数码相机我们可以随时随地进行拍照，相机能够照出好图片也要归功于 ISP(Image Singal Processor),那么 ISP 都干就是将图像转换为像素。当然不是简单转换，在这个过程中进行一系列优化。

#### 白平衡
我们知道看见的物体颜色是取决于光强度和物体反射率，在拍照时候需要环境进行估计，从而尽量第消除光对环境的影响。这种我们都会有体会在物体在日光灯下会偏蓝色些，而在白炽灯灯光下会显示暖色，偏红色。我们在看东西时候，人眼会自动调节来消除光源对物体的影响。所谓白平衡就是消除光源对物体反色的影响。
#### vignetting 周边偏暗
这个 vignetting 现象，也是需要校正的，一般通过事先设置好的查找表来实现。光从小孔投射到感光底板上，边缘接收光是斜射所以面积变大，光束是发散的，面积变大所以
####

这一步，主要是将 sensor 收集到的光能量进行转换成数字信号，出来的就是我们常见的 RAW 图，这个时候的 RAW 图，其像素值和环境光线的强弱是线性关系。


### OpenCV
#### findContours
轮廓检测也是图像处理中经常用到的。OpenCV-Python接口中使用cv2.findContours()函数来查找检测物体的轮廓。需要注意的是 cv2.findContours( 函数接受的参数为二值图，即黑白的(注意这里不是灰度图），所以读取的图像要先转成灰度的，再转成二值图。