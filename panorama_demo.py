import cv2
import numpy as np
import utils

img_left = cv2.imread('left_side.jpg')
# print(img_left.shape[:2])
# print(img_left.shape[1])
img_left = utils.resize(img_left,0.25)
# with = int(img_left.shape[1] )
img_left_gray = cv2.cvtColor(img_left,cv2.COLOR_BGR2GRAY)
# cv2.cvtColor
img_right = cv2.imread('center_side.jpg')
img_right = utils.resize(img_right,0.25)

img_right_gray = cv2.cvtColor(img_right,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

kp1, dest1 = sift.detectAndCompute(img_left_gray,None)
kp2, dest2 = sift.detectAndCompute(img_right_gray,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params = dict(checks=50)

# cv2.imshow('left image',cv2.drawKeypoints(img_left_gray,kp1,img_left))
# cv2.imshow('right image',cv2.drawKeypoints(img_right_gray,kp2,img_right))
match = cv2.FlannBasedMatcher(index_params,search_params)
matches = match.knnMatch(dest1,dest2,k=2)
# print(matches)
good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append(m)
draw_parameters = dict(matchColor=(0,255,0),singlePointColor=None,flags=2)
# print(len(good))
img3 = cv2.drawMatches(img_left,kp1,img_right,kp2,good,None,**draw_parameters)

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    # pass
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask =  cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)

    h,w = img_left_gray.shape
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    # cv2.imshow("original_image",drawMatches)
    img_right_gray = cv2.ploylines(img_right_gray,[np.init32(dst)],True,255,3,cv2.LINE_AA)
    cv2.imshow('original_image_drawMatches.jpg',img_right_gray)
else:
    print("Not enought matches are found - %d/%d",(len(good)/MIN_MATCH_COUNT))
while(1):
    # cv2.imshow('girl dist',img_left)
    
    k = cv2.waitKey(33)
    if k == 27:
        break
    elif k==-1:
        continue
    else:
        print(k)   
