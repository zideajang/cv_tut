#coding=utf-8
import cv2
import numpy as np
import argparse

class App():
    def run(self,img_name):
        dir = 'images/'
        img = cv2.imread( dir +  img_name + '.jpg')
        height,width,_ = img.shape
        scale_ratio = 4
        # (h,)
        # print(width,height)
        resize_img = cv2.resize(img,((width//scale_ratio),(height//scale_ratio)))
        gray = cv2.cvtColor(resize_img,cv2.COLOR_BGR2GRAY)
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