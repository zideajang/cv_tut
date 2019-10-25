#coding=utf-8
import cv2
import numpy as np
import argparse

class App():

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
        dst = self.find_counter(resize_img)
        
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