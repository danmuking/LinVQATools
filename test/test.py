import time

import cv2
import numpy as np

from PIL import Image

'''
读取视频并保存截帧

cv2.imwrite(filename, frame)
'''
videoName = r'G:/Dataset/VQA_ODV/Group1/Reference/G1AbandonedKingdom_7680x3840_fps30.mp4'
cap = cv2.VideoCapture(videoName)

# 视频属性
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取原视频的宽
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取原视频的搞
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # 视频的编码

n, i = 0, 0  # 总的帧数，保存的第i张图片

while cap.isOpened():
    ret, frame = cap.read()
    print(frame)
    filename = '{:0>4}.png'.format(str(i))
    print(filename)
    cv2.imwrite(filename, frame)  # 存入快照
    break
start = time.time()
image = cv2.imread(r"./0000.png")
# image = Image.open(r"./0000.png")
# image = np.array(image)
print(time.time()-start)
