## 准备 CV 环境
如何安装 opencv 
推荐安装 conda 后在 conda 环境下安装 opencv
1. 安装 conda(window 为例安装 conda) 这个很方便在官方安装即可
2. 创建python 环境
```
conda create -n env_cv python=3.6
```
这里我选择使用 python 3.6 版本来进行开发，可以选择自己熟悉的 python 版本

3. 安装 python 版本的 opencv
这里得提及一下在 opencv 3.4.2 版本以后，因为 SIFT 专利的问题。我们就无法在免费版本中使用到 SIFT 相关的方法来进行提取特征，所以这里推荐安装 3.4.2.16 版本

```
pip install opencv-python==3.4.2.16
```
4. 安装扩展，在安装完 opencv-python 我们还需安装一下扩展，这样就可以使用到一些比较高级的功能
```
pip install opencv-contrib-python==3.4.2.16
```
5. 引入依赖