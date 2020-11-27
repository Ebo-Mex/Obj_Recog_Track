import cv2
import os
import numpy as np

frame = cv2.imread(os.getcwd() + '/TEST/zips/R-5JSKO4sVc_0/0.jpg')
BB_file = os.getcwd() + '/TEST/anno/R-5JSKO4sVc_0.txt'

ArrayBB = np.loadtxt(BB_file, delimiter=",")

x_min = int(ArrayBB[0])
x_max = int(ArrayBB[2] + ArrayBB[0])
y_min = int(ArrayBB[1])
y_max = int(ArrayBB[3] + ArrayBB[1])

cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,0,255), 2)

cv2.imshow('Ground Truth bbox', frame)

cv2.imwrite('/home/ebo/Desktop/imgs/atom/0.png', frame)

cv2.waitKey(0)
