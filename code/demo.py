import cv2
import utils

img = cv2.imread('empire.jpeg',1)
img1 = cv2.imread('tesla-model-s3.jpg',1)
img2 = cv2.imread('damaged.jpg')
girl = cv2.imread('girl.jpg',1)
matthew = cv2.imread('sample/image1.jpg')


# dst = utils.img_mirror_v(img)
dst = utils.resize(matthew,0.2)
# img = utils.resize(img,0.2)
# dst = utils.oil_painting_process(img1)
# utils.train_fix_img(img1)
# dst = utils.hist_yuv_process(img1)
# dst = cv2.imread('damaged.jpg')
# dst = utils.mean_blur(matthew)
# utils.get_sample('matthew.mov')
dst = utils.beauty_filter(dst)
# dst = utils.increntment_brigtness(dst)
while(1):
    # channels = cv2.split(img1) # RGB R G B
    # for i in range(0,3):
    #     utils.image_hist(channels[i],31+i)
    # cv2.imshow('girl src',girl)
    cv2.imshow('girl dist',dst)
    k = cv2.waitKey(33)
    if k == 27:
        break
    elif k==-1:
        continue
    else:
        print(k)        
