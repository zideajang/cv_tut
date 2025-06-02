import cv2
import numpy as np
import matplotlib.pyplot as plt

PosNum = 820
NegNum = 1931

# params setting
winSize = (64,128)
# 105
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nBin = 9 # 3780
# hog
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nBin)
# svm
svm = cv2.ml.SVM_create()
# hog cal 3780
featureNum = int(( (128 - 16)/8 + 1) * ((64 - 16)/8 + 1)*4*9 ) 
print(featureNum)
# label
featureArray = np.zeros(((PosNum+NegNum),featureNum),np.float32)
labelArray = np.zeros(( (PosNum + NegNum),1),np.int32)
# svm image hog as sample

for i in range(0,PosNum):
    fileName = 'pos/' + str(i+1) + '.jpg'
    img = cv2.imread(fileName)

    hist = hog.compute(img,(8,8)) #3780
    for j in range(0,featureNum):
        featureArray[i,j] = hist[j]
        # featureArray hog [1,:] hog1 [2,:] hog2
        labelArray[i,0] = 1
        # post label
for i in range(0,NegNum):
    fileName = 'neg/' + str(i+1) + '.jpg'
    img = cv2.imread(fileName)
    hist = hog.compute(img,(8,8)) #3780
    for j in range(0,featureNum):
        featureArray[i + PosNum,j] = hist[j]
        # featureArray hog [1,:] hog1 [2,:] hog2
        labelArray[i + PosNum,0] = -1 
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(0.01)
# train
svm.train(featureArray,cv2.ml.ROW_SAMPLE,labelArray)
# test
alpha = np.zeros((1),np.float32)
rho = svm.getDecisionFunction(0,alpha)
print(alpha)
print(rho)
alphaArray = np.zeros((1,1),np.float32)
supportVArray = np.zeros((1,featureNum),np.float32)
resultArray = np.zeros((1,featureNum),np.float32)
alphaArray[0,0] = alpha
resultArray = -1 * alphaArray * supportVArray
# detect
mDetect = np.zeros((3781),np.float32)
for i in range(0,3780):
    mDetect[i] = resultArray[0,i]
mDetect[3780] = rho[0]
# create hog
mHog = cv2.HOGDescriptor()
mHog.setSVMDetector(mDetect)
imageSrc = cv2.imread('Test2.jpg',1)
objs = mHog.detectMultiScale(imageSrc,0,(8,8),(32,32),1.05,2)
x = (int)(objs[0][0][0])
y = (int)(objs[0][0][1])
w = (int)(objs[0][0][2])
h = (int)(objs[0][0][3])

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