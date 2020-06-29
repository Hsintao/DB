import cv2
import numpy as np

with open('/Users/xintao/Desktop/res_img_5112.txt', 'r') as f:
    preds = [i.strip().split(',')[:-1] for i in f.readlines()]
img = cv2.imread('/Volumes/WDSSD/文本检测/自然场景文字检测挑战赛初赛数据/total_text/test_images/img_5112.jpg')
pts = []
for i in preds:
    pts.append(np.array(i, np.int32).reshape(-1, 1, 2))
cv2.polylines(img, pts, True, (0, 255, 255), 2)
cv2.imwrite('res.jpg', img)
