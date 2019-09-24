

# 概述

图像的基本操作，最近对计算机矩阵进行操作。
- 图像基本操作
- 图像特效处理
- 图像特征点检测
- 图像到图像的映射
- 针孔照相机模型

- 增强现实
- 重建三维场景
- 

# 图像数据结构

### 图像数据格式
### 图像储存

# 图像的基本操作

## 图像几何变换
图片变换的本质就是对图片的矩阵进行操作
### 图片位移
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


# 局部图像描述子

## Harris 角点检测
在计算机视觉有关角点检测被广泛应用，例如图像匹配、相机定位等。表示图像的特点，角点可以用于定位，这是因为角点包含大量位置信息。要识别角点，我们首先需要了解角点有哪些特征。对于边在沿着边是灰度变化不大。有关 Harris 角点公式是推导，计算时候我们需要计算一点邻域

### 角点具有哪些特点
- 轮廓之间的交点
- 对于同一场景，在视角发生变化时候，具有稳定性的点
- 在角点附近的点无论在梯度方向和梯度幅度上有明显的变化的点

### 判断是角点的基本原理
算法基本思想是使用一个固定大小的窗口(矩阵)在图像（矩阵）上进行任意方向上的滑动，比较滑动前与滑动后两种情况，窗口中的像素灰度变化程度，如果存在任意方向上的滑动，都有着较大灰度变化，那么我们可以认为该窗口中存在角点。<br>
E(u,v) = $\sum_{x,y}$W(x,y)${[I(x+u,y+v) - I(x,y)]^2}$<br>
- u,v 表示窗口位置
- x,y 
- w(x,y)是窗口函数，输入 x 矩阵，输出也是矩阵。最简单情形就是窗口内的所有像素所对应的w权重系数均为1。可以将 x 们会将w(x,y)函数设定为以窗口中心为原点的二元正态分布。如果窗口中心点是角点时，移动前与移动后，该点的灰度变化应该最为剧烈，所以该点权重系数可以设定大些，表示窗口移动时，该点在灰度变化贡献较大；而离窗口中心(角点)较远的点，这些点的灰度变化几近平缓，这些点的权重系数，可以设定小点，以示该点对灰度变化贡献较小，那么我们自然想到使用二元高斯函数来表示窗口函数，这里仅是个人理解，大家可以参考下。

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



# 图像分割

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