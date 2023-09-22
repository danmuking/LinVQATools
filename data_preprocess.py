"""
    实现faster vqa fragment数据预处理
"""
import os
from multiprocessing.pool import Pool

import cv2
import decord
from decord import VideoReader
from einops import rearrange
from torchvision.transforms import transforms

from data.meta_reader import ODVVQAReader

decord.bridge.set_bridge("torch")

def makedir(path: str):
    dir_path = path
    if os.path.exists(dir_path):
        pass
    else:
        os.makedirs(dir_path)

def get_save_path(video_path, frame_num, epoch):
    video_path = video_path.split('/')
    video_path.insert(3, 'frame')
    video_path.insert(4, str(epoch))
    video_path[0] = "/data"
    video_path[1] = ""
    video_path = os.path.join(*video_path)[:-4]
    makedir(video_path)
    img_path = os.path.join(video_path, '{}.png'.format(frame_num))
    return img_path


# TODO: 在时间上位置没有变化
def sampler(video_path: str, epoch: int):
    vreader = VideoReader(video_path)
    for i in range(len(vreader)):
        img = vreader[i]
        img = rearrange(img, 'h w c -> c h w')
        save_path = get_save_path(video_path, i, 0)
        target_img = transforms.Resize(512)(img)
        target_img = rearrange(target_img, 'c h w -> h w c ')
        target_img = target_img.numpy()
        target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, (target_img).astype('uint8'))
        print('{}已保存'.format(save_path))


if __name__ == '__main__':

    # os.chdir('/')
    file = os.path.dirname(os.path.abspath(__file__))
    anno_path = os.path.join(file, './data/odv_vqa')
    data_anno = ODVVQAReader(anno_path).read()
    pool = Pool(4)
    for video_info in data_anno:
        video_path = video_info['video_path']
        print(video_path)
        pool.apply_async(func=sampler, kwds={'video_path': video_path,'epoch': 0})
    pool.close()
    pool.join()
    # for video_info in data_anno:
    #     video_path = video_info['video_path']
    #     sampler(video_path, 0)
