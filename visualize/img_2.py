import random

import cv2

video = [
    "/data/ly/VQA_ODV/Group1/G1BikingToWork_ERP_3840x2160_fps23.976_qp27_12306k.mp4",
    "/data/ly/VQA_ODV/Group1/G1BajaCalifonia_TSP_3840x2160_fps23.976_qp27_29617k_ERP.mp4",
    "/data/ly/VQA_ODV/Group1/G1LateShow_ERP_3840x2160_fps23.976_qp37_2051k.mp4",
    "/data/ly/VQA_ODV/Group2/G2AstonVillaGoal_TSP_3840x1920_fps24_qp27_9436k.mp4"
]

# 抽取视频中的相邻4帧
def extract_frame(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    while(cap.isOpened()):
        print(cap.isOpened())
        ret, frame_4k = cap.read()
        frame_list.append(frame_4k)
        if len(frame_list) == 4:
            break
    cap.release()
    return frame_list


for i in range(4):
    video_path = video[i]
    video_frame = extract_frame(video_path)
    # 获取图片的尺寸
    height, width = video_frame[0].shape[:2]
    # 计算方框的左上角和右下角坐标
    box_size = 224
    randnum = random.randint(0,300)
    top_left = (int((width - box_size) / 2)+randnum, int((height - box_size) / 2)+randnum)
    bottom_right = (top_left[0] + box_size, top_left[1] + box_size)
    for j,frame in enumerate(video_frame):
        cropped_image = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        cv2.imwrite('img_{}_{}_crop.png'.format(j,i), cropped_image)
        # 在图片中绘制方框
        color = (0, 0, 255)  # 绿色
        thickness = 2
        cv2.rectangle(frame, top_left, bottom_right, color, thickness)
        cv2.imwrite('img_{}_{}.png'.format(j, i), frame)
