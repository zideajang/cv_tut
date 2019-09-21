import cv2
import numpy as np
import matplotlib.pyplot as plt

# data
rand1 = np.array([[152,46],[159,50],[165,55],[168,55],[172,60]])
rand2 = np.array([[152,54],[156,55],[160,56],[172,65],[178,68]])
# label
label = np.array([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1]])

data = np.vstack((rand1,rand2))
data = np.array(data,dtype='float32')
# create svm
svm = cv2.ml.SVM_create()
# set type
svm.setType(cv2.ml.SVM_C_SVC)
# line
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(0.01)
# train
result = svm.train(data,cv2.ml.ROW_SAMPLE,label)
# test
# [0,1]
pt_data = np.vstack([[167,55],[162,57]])
pt_data = np.array(pt_data,dtype='float32')
print(pt_data)
(par1,par2) = svm.predict(pt_data)
print(par1,par2)