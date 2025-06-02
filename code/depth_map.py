# coding=utf-8
import numpy as np
import cv2

class App():

    def run(slef):
        imgLeft = cv2.imread('images/tsukuba_l.png', 0)
        imgRight = cv2.imread('images/tsukuba_r.png', 0)

        # 构建深度图的函数
        # 即最大视差值与最小视差值之差, 窗口大小必须是16的整数倍，int 型
        # 匹配的块大小。它必须是> = 1的奇数。通常情况下，应该在 3--11的范围内。这里设置为大于11也可以，但必须为奇数。
        stereo = cv2.StereoBM_create(numDisparities=32, blockSize=13)

        # 计算图片差异
        disparity = stereo.compute(imgLeft, imgRight)

        min = disparity.min()
        max = disparity.max()
        disparity = np.uint8(255 * (disparity - min) / (max - min))

        # 显示结果
        cv2.imshow('disparity', np.hstack((imgLeft, imgRight, disparity)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    App().run()