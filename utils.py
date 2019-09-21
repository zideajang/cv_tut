import cv2
import numpy as np
import random
import math
def resize(img,param):
    height,width,_ = get_img_info(img)
    dstHeight = int(height * param)
    dstWidth = int(width * param)
    dst = cv2.resize(img,(dstWidth,dstHeight),cv2.INTER_AREA)
    return dst

def get_img_info(img):
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    mode = imgInfo[2]
    return height, width, mode

# load
# template
# x y 
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

def resize_with_mat(img,scale):
    height,width,_ = get_img_info(img)
    # tranform array
    matTransform = np.float32([[scale,0,0],[0,scale,0]]) # 2 * 3
    dst = cv2.warpAffine(img,matTransform,(int(width*scale),int(height*scale)) )
    return dst

def img_crop(img,startX,startY,h,w):
    height,width,_ = get_img_info(img)
    print(startX,(startX+h),startY,startY + w)
    dst = img[startX:(startX+w),startY:(startY + h)]
    # dst = img[100:200,100:300]
    return dst

def img_translate(img):
    height,width,_ = get_img_info(img)
    # tranform array
    matTransform = np.float32([[1,0,100],[0,1,200]]) # 2 * 3
    dst = cv2.warpAffine(img,matTransform,(width,height))
    return dst
def img_custom_transform(img,_x,_y):
    height,width,_ = get_img_info(img)
    dst = np.zeros((height + _y, width + _x),np.uint8)
    print(dst)
    for i in range(0,height):
        for j in range(0,width):
            dst[(i+ _y),(j+_x)] = img[i,j]
    return dst

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

def img_transform(img):
    height,width,deep = get_img_info(img)
    matSrc = np.float32([[0,0],[0,height-1],[width-1,0]])
    matDst = np.float32([[50,50],[300,height-200],[width-300,100]])
    # combination
    matAffine = cv2.getAffineTransform(matSrc,matDst) # mat
    dst = cv2.warpAffine(img,matAffine,(width,height))
    return dst

def img_rotate(img):
    height,width,deep = get_img_info(img)

    # mat rotate 1 center 2 angle 3 src
    matRotate = cv2.getRotationMatrix2D((height*0.5,width*0.5),45,0.5)
    dst = cv2.warpAffine(img,matRotate,(width,height))
    return dst

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

def color_gray_invert(img):
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

def color_invert(img):
    height,width,deep = get_img_info(img)
    dst = np.zeros((height,width,deep),np.uint8)
    for i in range(0,height):
        for j in range(0,width):
            (b,g,r) = img[i,j]
            dst[i,j] = (255-b,255-g,255-r)
    return dst

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

def edge_detect(img):
    # height,width,deep = get_img_info(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # height = gray.shape[0]
    # width = gray.shape[1]
    dst = cv2.GaussianBlur(gray,(3,3),0)
    dst = cv2.Canny(dst,50,50)
    return dst
def edge_custom_detect(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    width = gray.shape[1]
    y = np.array([[1,2,1],[0,0,0],[-1,-2,0]])
    x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    dst = np.zeros((height,width,1),np.uint8)
    for i in range(0,height - 2):
        for j in range(0,width - 2):
            # gy = gray[i,j] * 1 + gray[i,j+1] * 2 + gray[i,j+2]*1 - gray[i+2,j]*1 -gray[i+2,j+1]*2
            # gx = gray[i,j] * 1 - gray[i,j+2] * 2 +  gray[i+1,j] * 2 - gray[i+1,j+2] * 2 + gray[i+2,j] * 1 - gray[i+2,j+2] * 2
            # imgy = np.array([[ gray[i,j], ] ])
            gy = np.dot(y)
            grad = math.sqrt(gx*gx+gy*gy)
            print(grad)
            if grad > 50:
                dst[i,j] = 255
            else:
                dst[i,j] = 0

    return dst

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

def movie_effects_process(img):
    height,width,deep = get_img_info(img)
    
    dst = np.zeros((height,width,3),np.uint8)
    for i in range(0,height):
        for j in range(0,width):
            

            (b,g,r) = img[i,j]
            b = b * 1.5
            g = g * 1.3

            if b > 255:
                b = 255
            if g > 255:
                g = 255

            dst[i,j] = (b,g,r)
    return dst

def oil_painting_process(img):
    height,width,deep = get_img_info(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dst = np.zeros((height,width,3),np.uint8)
    for i in range(4,height - 4):
        for j in range(4,width - 4):
            arr1 = np.zeros(8,np.uint8)
            for m in range(-4,4):
                for n in range(-4,4):
                    p1 = int(gray[i+m,j+n]/32)
                    arr1[p1] = arr1[p1] + 1
            currentMax = arr1[0]
            l = 0
            for k in range(0,8):
                if currentMax< arr1[k]:
                    currentMax = arr1[k]
                    l = k
            for m in range(-4,4):
                for n in range(-4,4):
                    if gray[i+m,j+n]>=(l*32) and gray[i+m,j+n] <= ((l+1)*32):
                        (b,g,r) = img(i+m,j+n)
            dst[i,j] = (b,g,r)

    return dst

def image_hist(img,type):
    color = (255,255,255)
    windowName = 'Gray'
    if type == 31:
        color = (255,0,0)
        windowName = 'B Hist'
    elif type == 32:
        color = (0,255,0)
        windowName = 'G Hist'
    elif type == 33:
        color = (0,0,255)
        windowName = 'R Hist'
    # 1 image 2 [0] 3 mask None 4 256 5 0-255
    hist = cv2.calcHist([img],[0],None,[256],[0.0,255.0])
    minV,maxV,minL,maxL = cv2.minMaxLoc(hist)
    histImg = np.zeros([256,256,3],np.uint8)
    for h in range(256):
        intenNormal = int(hist[h]*256/maxV)
        cv2.line(histImg,(h,256),(h,256 -intenNormal),color)
    cv2.imshow(windowName,histImg)
    return histImg
def hist_gray_process(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    return dst

def hist_color_process(img):
    (b,g,r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # channels merge
    result = cv2.merge((bH,gH,rH))
    return result

def hist_yuv_process(img):
    imgYUV = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    channelYUV = cv2.split(imgYUV)
    channelYUV[0] = cv2.equalizeHist(channelYUV[0])
    channels = cv2.merge(channelYUV)
    dst = cv2.cvtColor(channels,cv2.COLOR_YCrCb2BGR)
    return dst

def train_fix_img(img):
    for i in range(100,200):
        img[i,100] = (255,255,255)
        img[i,100 + 1] = (255,255,255)
        img[i,100 - 1] = (255,255,255)
    for i in range(150,250):
        img[150,i] =  (255,255,255)
        img[150 + 1,i] =  (255,255,255)
        img[150 + 1,i] =  (255,255,255)
    
    cv2.imwrite('damaged.jpg',img)

def fix_img(img):
    height,width,deep = get_img_info(img)
    paint = np.zeros((height,width,1),np.uint8)
    for i in range(100,200):
        paint[i,100] = 255
        paint[i,100 + 1] = 255
        paint[i,100 - 1] = 255
    for i in range(150,250):
        paint[150,i] =  255
        paint[150 + 1,i] =  255
        paint[150 + 1,i] =  255

    dst = cv2.inpaint(img,paint,3,cv2.INPAINT_TELEA)
    return dst    

def custom_hist_process(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    count = np.zeros(256,np.float)
    for i in range(0,height):
        for j in range(0,width):
            pixel = gray(i,j)
            index = int(pixel)
            count[index] = count[index] + 1

def beauty_filter(img):
    dst = cv2.bilateralFilter(img,15,35,35)
    return dst

def guassian_filter(img):
    dst = cv2.GaussianBlur(img,(5,5),1.5)
    return dst

def mean_blur(img):
    height,width,deep = get_img_info(img)
    dst = np.zeros((height,width,3),np.uint8)
    for i in range(0,height - 3):
        for j in range(0,width - 3):
            sum_b = int(0)
            sum_g = int(0)
            sum_r = int(0)
            for m in range(-3,3): #-3,-2,-1,0,1,2
                for n in range(-3,3):
                    (b,g,r) = img[i+m,j+n]
                    sum_b = sum_b + int(b)
                    sum_g = sum_g + int(g)
                    sum_r = sum_r + int(r)
                b = np.uint8(sum_b/36)
                g = np.uint8(sum_g/36)
                r = np.uint8(sum_r/36)
                dst[i,j] = (b,g,r)
    return dst
def increntment_brigtness(img):
    height,width,deep = get_img_info(img)
    dst = np.zeros((height,width,3),np.uint8)
    for i in range(0,height):
        for j in range(0,width):
            (b,g,r) = img[i,j]
            bb = int(b * 1.3) + 10
            gg = int(g * 1.2) + 15
            # rr = int(r) + 40

            if bb > 255:
                bb = 255
            if gg > 255:
                gg = 255
            # if rr > 255:
            #     rr = 255

            dst[i,j] = (bb,gg,r)
    return dst

# 1 load 2 info 3 parse 4 imshow imwrite

def get_sample(video):
    cap = cv2.VideoCapture(video)
    isOpened = cap.isOpened
    print(isOpened)
    # get info about video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(fps,width,height)
    i = 0
    while(isOpened):
        if i == 10:
            break;
        else:
            i = i + 1
        (flag, frame) = cap.read()
        fileName = 'sample/image'+str(i)+'.jpg'
        if flag == True:
            cv2.imwrite(fileName,frame,[cv2.IMWRITE_JPEG_QUALITY,100])
    print('end')

def get_video(img):
    imgInfo = img.shape
    size = (imgInfo[0],imgInfo[1])
    # write 1. file name 2. encoder, fps, size
    videoWrite = cv2.VideoWriter('matthew.mp4',-1,5,size)
    for i in range(1,11):
        fileName = 'sample/image'+str(i) + '.jpg'
        img = cv2.imread(fileName)
        videoWrite.write(img)

    print('end')