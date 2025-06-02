#coding=utf-8
import cv2
import numpy as np
import argparse

class App():

    def watershed_img_segmetation(self,img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1

        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0
        markers = cv2.watershed(img,markers)
        img[markers == -1] = [255,0,0]
        return img

    def find_counter(self,img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY) 
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        cv2.drawContours(img,contours,-1,(0,0,255),3) 

        return img
    def threld_img_segmentation(self,img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # 高斯模糊
        # blur =cv2.GaussianBlur(img,(5,5),0)
        # canny 边缘检测
        # dst = cv2.Canny(blur, 100, 200)
        ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY) 
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        cv2.drawContours(img,contours,-1,(0,0,255),3) 
        return img


    def run(self,img_name):
        dir = 'images/'
        img = cv2.imread( dir +  img_name + '.jpg')
        height,width,_ = img.shape
        scale_ratio = 2
        # (h,)
        # print(width,height)
        resize_img = cv2.resize(img,((width//scale_ratio),(height//scale_ratio)))
        
        # 获取
        dst = self.watershed_img_segmetation(resize_img)
        
        while(1):
            cv2.imshow("dst",dst)
            # cv2.imshow("img",resize_img)
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