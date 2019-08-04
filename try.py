import json
import numpy as np
import cv2

with open('person_keypoints_train2017.json') as f:
    data=json.load(f)

img=cv2.imread('000000537548.jpg')
bbox = np.array(data['annotations'][0]['bbox'])

cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])), (0,0,255), 2)
cv2.imshow('img', img)
print (bbox)
cv2.waitKey(0)

