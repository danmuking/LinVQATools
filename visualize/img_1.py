import decord

video_520 = "/data/ly/KoNViD_1k/KoNViD_1k_videos/2999049224.mp4"
video_4k = "/data/ly/VQA_ODV/Group1/G1BikingToWork_ERP_3840x2160_fps23.976_qp27_12306k.mp4"

# 使用cv2读取video_520的第一帧
import cv2
cap = cv2.VideoCapture(video_520)
frame_520 = None
while(cap.isOpened()):
    print(cap.isOpened())
    ret, frame_520 = cap.read()
    break
cap.release()
# 获取图片的尺寸
height, width = frame_520.shape[:2]
# 计算方框的左上角和右下角坐标
box_size = 224
top_left = (int((width - box_size) / 2), int((height - box_size) / 2))
bottom_right = (top_left[0] + box_size, top_left[1] + box_size)
cropped_image = frame_520[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
cv2.imwrite('frame_520_crop.png', cropped_image)
# 在图片中绘制方框
color = (0, 0, 255)  # 绿色
thickness = 2
cv2.rectangle(frame_520, top_left, bottom_right, color, thickness)
cv2.imwrite('frame_520.png', frame_520)

cap = cv2.VideoCapture(video_520)
frame_520 = None
while(cap.isOpened()):
    print(cap.isOpened())
    ret, frame_520 = cap.read()
    break
cap.release()
height, width = frame_520.shape[:2]
radio = 224/height
# resize
frame_520_resize = cv2.resize(frame_520, (int(width*radio),224))
# 计算方框的左上角和右下角坐标
height, width = frame_520_resize.shape[:2]
box_size = int(224*radio)
top_left = (int((width - box_size) / 2), int((height - box_size) / 2))
bottom_right = (top_left[0] + box_size, top_left[1] + box_size)
cropped_image = frame_520_resize[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
cv2.imwrite('frame_520_crop_resize.png', cropped_image)
# 在图片中绘制方框
color = (0, 0, 255)  # 绿色
thickness = 2
cv2.rectangle(frame_520_resize, top_left, bottom_right, color, thickness)
cv2.imwrite('frame_520_resize.png', frame_520_resize)






cap = cv2.VideoCapture(video_4k)
frame_4k = None
while(cap.isOpened()):
    print(cap.isOpened())
    ret, frame_4k = cap.read()
    break
cap.release()
# 获取图片的尺寸
height, width = frame_4k.shape[:2]
# 计算方框的左上角和右下角坐标
box_size = 224
top_left = (int((width - box_size) / 2), int((height - box_size) / 2))
bottom_right = (top_left[0] + box_size, top_left[1] + box_size)
cropped_image = frame_4k[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
cv2.imwrite('frame_4k_crop.png', cropped_image)
# 在图片中绘制方框
color = (0, 0, 255)  # 绿色
thickness = 2
cv2.rectangle(frame_4k, top_left, bottom_right, color, thickness)
cv2.imwrite('frame_4k.png', frame_4k)

cap = cv2.VideoCapture(video_4k)
frame_4k = None
while(cap.isOpened()):
    print(cap.isOpened())
    ret, frame_4k = cap.read()
    break
cap.release()
height, width = frame_4k.shape[:2]
radio = 224/height
# resize
frame_4k_resize = cv2.resize(frame_4k, (int(width*radio),224))
# 计算方框的左上角和右下角坐标
height, width = frame_4k_resize.shape[:2]
box_size = int(224*radio)
top_left = (int((width - box_size) / 2), int((height - box_size) / 2))
bottom_right = (top_left[0] + box_size, top_left[1] + box_size)
cropped_image = frame_4k_resize[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
cv2.imwrite('frame_4k_crop_resize.png', cropped_image)
# 在图片中绘制方框
color = (0, 0, 255)  # 绿色
thickness = 2
cv2.rectangle(frame_4k_resize, top_left, bottom_right, color, thickness)
cv2.imwrite('frame_4k_resize.png', frame_4k_resize)