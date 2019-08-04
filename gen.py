import os
import sys
import numpy as np
import cv2
from data_aug.data_aug import *
from data_aug.bbox_util import *
import matplotlib.pyplot as plt 
from random import randint

if len(sys.argv) != 2:
   print ("Usage gen.py <input_img>")
   exit(0)

path =""
#prefix_path = "../yolo_dataset2/yolo_thesis/datasets/bdd-tiny/train/images/"
prefix_path="./"
prefix_path="../yolo_masks/yolo_thesis/datasets/coco-masks-tiny/train/images/"
prefix_path="./try_10_2/images/"
#prefix_path="/home/eceftl9/Thesis/bdd-dataset/bdd-small/train/images/"
if os.path.exists(prefix_path+str(sys.argv[1])):
    path = os.path.abspath(prefix_path+str(sys.argv[1]))
else:
    print("Cannot find " + sys.argv[1])
    exit(0)


#out_path="/home/msattir/Thesis/yolo_dataset2/yolo_thesis/datasets/bdd-tiny-augm/train/"
out_path = "./coco/"
out_path = "../yolo_masks/yolo_thesis/datasets/coco-masks-agum/train/"
out_path = "./try_10_out/"
#out_path = "/home/eceftl9/Thesis/bdd-dataset/bdd-small-augm/train/"

tmp = path.rsplit('/', 2)
image = tmp[2]
label_path = tmp[0]+"/labels/"+tmp[2].replace('jpg', 'txt')
img = cv2.imread(path)#[:,:,::-1]
bboxes = np.genfromtxt(label_path, delimiter=',')

if bboxes.ndim == 1:
    bboxes = bboxes.reshape(1,-1)

bboxes[:,2] += bboxes[:,0]
bboxes[:,3] += bboxes[:,1]

i = 1

while (i<=10):
    rand = i#randint(1,8)
    if rand == 1:
       img_, bboxes_ = RandomHorizontalFlip(1)(img.copy(), bboxes.copy())
       bboxes_[:,2] -= bboxes_[:,0]
       bboxes_[:,3] -= bboxes_[:,1]
       cv2.imwrite((out_path+'images/'+image.split('.')[0]+'_{}').format(i)+'.jpg', img_)
       np.savetxt((out_path+'labels/'+image.split('.')[0]+'_{}').format(i)+'.txt', bboxes_, delimiter=',')
       i += 1
    elif rand == 2:
       img_, bboxes_ = RandomScale(0.3, diff = True)(img.copy(), bboxes.copy())
       bboxes_[:,2] -= bboxes_[:,0]
       bboxes_[:,3] -= bboxes_[:,1]
       cv2.imwrite((out_path+'images/'+image.split('.')[0]+'_{}').format(i)+'.jpg', img_)
       np.savetxt((out_path+'labels/'+image.split('.')[0]+'_{}').format(i)+'.txt', bboxes_, delimiter=',')
       i += 1
    elif rand == 3:
       img_, bboxes_ = RandomTranslate(0.3, diff = True)(img.copy(), bboxes.copy())
       bboxes_[:,2] -= bboxes_[:,0]
       bboxes_[:,3] -= bboxes_[:,1]
       cv2.imwrite((out_path+'images/'+image.split('.')[0]+'_{}').format(i)+'.jpg', img_)
       np.savetxt((out_path+'labels/'+image.split('.')[0]+'_{}').format(i)+'.txt', bboxes_, delimiter=',')
       i += 1
    elif rand == 4:
       img_, bboxes_ = RandomRotate(20)(img.copy(), bboxes.copy())
       bboxes_[:,2] -= bboxes_[:,0]
       bboxes_[:,3] -= bboxes_[:,1]
       cv2.imwrite((out_path+'images/'+image.split('.')[0]+'_{}').format(i)+'.jpg', img_)
       np.savetxt((out_path+'labels/'+image.split('.')[0]+'_{}').format(i)+'.txt', bboxes_, delimiter=',')
       i += 1
    elif rand == 5:
       img_, bboxes_ = RandomShear(0.2)(img.copy(), bboxes.copy())
       bboxes_[:,2] -= bboxes_[:,0]
       bboxes_[:,3] -= bboxes_[:,1]
       cv2.imwrite((out_path+'images/'+image.split('.')[0]+'_{}').format(i)+'.jpg', img_)
       np.savetxt((out_path+'labels/'+image.split('.')[0]+'_{}').format(i)+'.txt', bboxes_, delimiter=',')
       i += 1
    elif rand == 12:
       img_, bboxes_ = Resize(608)(img.copy(), bboxes.copy())
       bboxes_[:,2] -= bboxes_[:,0]
       bboxes_[:,3] -= bboxes_[:,1]
       cv2.imwrite((out_path+'images/'+image.split('.')[0]+'_{}').format(i)+'.jpg', img_)
       np.savetxt((out_path+'labels/'+image.split('.')[0]+'_{}').format(i)+'.txt', bboxes_, delimiter=',')
       i += 1
    elif rand == 7:
       img_, bboxes_ = RandomHSV(100, 100, 100)(img.copy(), bboxes.copy())
       bboxes_[:,2] -= bboxes_[:,0]
       bboxes_[:,3] -= bboxes_[:,1]
       cv2.imwrite((out_path+'images/'+image.split('.')[0]+'_{}').format(i)+'.jpg', img_)
       np.savetxt((out_path+'labels/'+image.split('.')[0]+'_{}').format(i)+'.txt', bboxes_, delimiter=',')
       i += 1
    else:
       seq=Sequence([RandomHSV(40, 40, 30),RandomHorizontalFlip(), RandomScale(), RandomTranslate(), RandomRotate(10), RandomShear()])
       img_, bboxes_ = seq(img.copy(), bboxes.copy())
       bboxes_[:,2] -= bboxes_[:,0]
       bboxes_[:,3] -= bboxes_[:,1]
       cv2.imwrite((out_path+'images/'+image.split('.')[0]+'_{}').format(i)+'.jpg', img_)
       np.savetxt((out_path+'labels/'+image.split('.')[0]+'_{}').format(i)+'.txt', bboxes_, delimiter=',')
       i += 1
    #bboxes_[:,2] += bboxes_[:,0]
    #bboxes_[:,3] += bboxes_[:,1]
    #plotted_img = draw_rect(img_, bboxes_)
    #print (bboxes_)
    #plt.imshow(plotted_img)
    #plt.show()
    #plt.pause(3)
    #plt.close()
