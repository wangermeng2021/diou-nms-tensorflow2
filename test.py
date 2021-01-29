
import numpy as np
import cv2
from nms import  diou_nms_np
from nms import  diou_nms_tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
ori_img = cv2.imread('test_pictures/1.jpg')
dst_img = ori_img.copy()
#
boxes = np.array([[[130,50,300.,310],[110,20,270.,280],[160,80,330.,330]]])
scores = np.array([[[0.1,0.8,0.2],[0.3,0.7,0.01],[0.1,0.6,0.21]]])

for box in boxes[0]:
    cv2.rectangle(ori_img,tuple(box[0:2].astype(np.int32)),tuple(box[2:4].astype(np.int32)),(255,255,0),1)

np_result = diou_nms_np(boxes, scores, iou_threshold=0.5, score_threshold=0.2)
tf_result = diou_nms_tf(boxes, scores, iou_threshold=0.5, score_threshold=0.2)

boxes = tf_result[0].numpy()[0]
scores = tf_result[1].numpy()[0]
classes = tf_result[2].numpy()[0]
num_valid = tf_result[3].numpy()[0]

for i in range(num_valid):
    box = boxes[i]
    cv2.rectangle(dst_img, tuple(box[0:2].astype(np.int32)), tuple(box[2:4].astype(np.int32)), (0, 0, 255), 1)

cv2.imshow("test",np.hstack([ori_img,dst_img]))
cv2.waitKey()
