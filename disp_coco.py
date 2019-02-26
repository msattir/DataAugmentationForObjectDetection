import sys
import os
import cv2
import numpy as np

#det=np.genfromtxt('000000255186.txt', delimiter=',')
#det=det[:,:-1].astype(int)

#img=cv2.imread('000000255186.jpg')
#cv2.imshow('fig1', img)
#cv2.waitKey(0)

path =""
#prefix_path = "../yolo_dataset2/yolo_thesis/datasets/bdd-tiny/train/images/"
prefix_path="./"
if os.path.exists(prefix_path+str(sys.argv[1])):
    path = os.path.abspath(prefix_path+str(sys.argv[1]))
else:
    print("Cannot find " + sys.argv[1])
    exit(0)


#out_path="/home/msattir/Thesis/yolo_dataset2/yolo_thesis/datasets/bdd-tiny-augm/train/"
out_path = "./gen/"
tmp = path.rsplit('/', 2)
image = tmp[2]
label_path = tmp[0]+"/labels/"+tmp[2].replace('jpg', 'txt')
img = cv2.imread(path)#[:,:,::-1]
det = np.genfromtxt(label_path, delimiter=',').astype(int)



font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.4
fontColor              = (255,255,255)
lineType               = 1

for i in range(0,det.shape[0]):
    cv2.rectangle(img, (det[i,0],det[i,1]), (det[i,2],det[i,3]), (0, 255, 0), 2)
    det2=det[i,4:]
    for j in range(0,51,3):
        cv2.circle(img, (det2[j],det2[j+1]), 2, (0,255,0))
        cv2.putText(img, "{}".format(int(j/3)), (det2[j],det2[j+1]), font, fontScale, fontColor, lineType)


cv2.imshow('fig_{}'.format(path.rsplit('.',1)[0].rsplit('_',1)[1]), img)
cv2.waitKey(0)
