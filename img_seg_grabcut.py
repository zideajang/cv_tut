#coding=utf-8
import cv2
import numpy as np
import argparse

class App():
    def run(self,img_name):
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("imagename", help="display image name", type=str)
    args = parser.parse_args()
    
    # 
    App().run(args.imagename)
    cv2.destroyAllWindows()