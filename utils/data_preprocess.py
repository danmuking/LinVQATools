import cv2
import lmdb
import numpy as np

video = dict()
with open("/home/ly/code/LinVQATools/data/odv_vqa/VQA_ODV.txt",'r') as f:
    lines = f.readlines()
for line in lines:
    line:str = line.strip()
    _,_,ref,vid,_,_,_,frame = line.split(' ')
    video[ref] = int(frame)
    video[vid] = int(frame)
for k,v in video.items():
    print(k,v)
# env = lmdb.open('/data/ly/lmdb', map_size=1099511627776)
# cache = {}  # 存储键值对
# image_path = './test.jpg'
# with open(image_path, 'rb') as f:
#     # 读取图像文件的二进制格式数据
#     image_bin = f.read()
#
# # 用两个键值对表示一个数据样本
# cache['image_000'] = image_bin
#
# with env.begin(write=True) as txn:
#     for k, v in cache.items():
#         if isinstance(v, bytes):
#             # 图片类型为bytes
#             txn.put(k.encode(), v)
#         else:
#             # 标签类型为str, 转为bytes
#             txn.put(k.encode(), v.encode())  # 编码
# env.close()


# with env.begin(write=False) as txn:
#     # 获取图像数据
#     image_bin = txn.get('image_000'.encode())
#     label = txn.get('label_000'.encode()).decode()  # 解码
#
#     # 将二进制文件转为十进制文件（一维数组）
#     image_buf = np.frombuffer(image_bin, dtype=np.uint8)
#     # 将数据转换(解码)成图像格式
#     # cv2.IMREAD_GRAYSCALE为灰度图，cv2.IMREAD_COLOR为彩色图
#     img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
#     cv2.imshow('image', img)
#     cv2.waitKey(0)