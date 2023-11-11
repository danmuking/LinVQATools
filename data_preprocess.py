"""
    实现faster vqa fragment数据预处理
"""
import json
import os
import random
from multiprocessing.pool import Pool

import cv2
import lmdb
import numpy as np
import torch
import decord
import torchvision
from decord import VideoReader
from einops import rearrange
from tqdm import tqdm

from data.meta_reader import ODVVQAReader
from data.split.dataset_split import DatasetSplit

# from SoftPool import soft_pool2d, SoftPool2d

decord.bridge.set_bridge("torch")

env = lmdb.open('/data/ly/resize/',map_size=1099511627776)
train_info = dict()
test_info = dict()
class FragmentSampleFrames:
    """
    时间上的fragment采样
    """

    def __init__(self, fsize_t, fragments_t, frame_interval=1, num_clips=1, drop_rate=0., **opt):
        # 每个fragment采样几帧
        self.fragments_t = fragments_t
        # 采样几个fragment
        self.fsize_t = fsize_t
        # 总采样帧数
        self.size_t = fragments_t * fsize_t
        # 帧间隔
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.drop_rate = drop_rate

    def get_frame_indices(self, num_frames, train=False):
        """
        获取帧索引
        :param num_frames: 总帧数
        :param train: 模式
        :return:
        """
        rand = np.random.RandomState()
        tgrids = np.array(
            [num_frames // self.fragments_t * i for i in range(self.fragments_t)],
            dtype=np.int32,
        )
        # fragment总数
        tlength = num_frames // self.fragments_t

        if tlength > self.fsize_t * self.frame_interval:
            rnd_t = rand.randint(
                0, tlength - self.fsize_t * self.frame_interval, size=len(tgrids)
            )
        else:
            rnd_t = np.zeros(len(tgrids), dtype=np.int32)

        ranges_t = (
                np.arange(self.fsize_t)[None, :] * self.frame_interval
                + rnd_t[:, None]
                + tgrids[:, None]
        )

        drop = random.sample(list(range(self.fragments_t)), int(self.fragments_t * self.drop_rate))
        dropped_ranges_t = []
        for i, rt in enumerate(ranges_t):
            if i not in drop:
                dropped_ranges_t.append(rt)
        return np.concatenate(dropped_ranges_t)

    def __call__(self, total_frames, train=False, start_index=0):
        frame_inds = []

        for i in range(self.num_clips):
            frame_inds += [self.get_frame_indices(total_frames)]

        frame_inds = np.concatenate(frame_inds)
        frame_inds = np.mod(frame_inds + start_index, total_frames)
        print(frame_inds)
        return frame_inds.astype(np.int32)


def makedir(path: str):
    dir_path = path
    if os.path.exists(dir_path):
        pass
    else:
        os.makedirs(dir_path)


def get_save_path(video_path, frame_num, epoch):
    video_path = video_path.split('/')
    video_path.insert(3, '4frame')
    video_path.insert(4, str(epoch))
    video_path[0] = "/data"
    video_path[1] = ""
    video_path = os.path.join(*video_path)[:-4]
    makedir(video_path)
    img_path = os.path.join(video_path, '{}.png'.format(frame_num))
    return img_path


# TODO: 在时间上位置没有变化
def sampler(video_path: str, is_train):
    vreader = VideoReader(video_path)
    # frame_index = [x for x in range(len(vreader))]
    frame_sampler = FragmentSampleFrames(fsize_t=16, fragments_t=1, frame_interval=4, num_clips=1, )
    sample_list = []
    if is_train:
        for i in range(160):
            sample_list.append(frame_sampler(len(vreader)))
        sample_list = np.array(sample_list)
        sample_list = sample_list.reshape(-1)
        sample_list = sample_list.reshape(160,16)
        train_info[video_path] = sample_list.tolist()
    else:
        sample_list.append(frame_sampler(len(vreader)))
        sample_list = np.array(sample_list)
        test_info[video_path] = sample_list.tolist()

    frame_list = np.unique(sample_list.reshape(-1))

    frame_dict = dict()
    for index in frame_list:
        frame_dict[index] = vreader[index]

    h,w,c = vreader[0].shape
    min_scale = 224**2/(h*w)*3
    max_scale = 224**2/(h*w)*5 if min_scale*4<1 else 1

    for index in range(sample_list.shape[0]):
        frames = []
        for frame_index in sample_list[index]:
            frames.append(frame_dict[frame_index])
        frames = torch.stack(frames,dim=0)
        frames = rearrange(frames, 't h w c -> t c h w')
        crop = torchvision.transforms.RandomResizedCrop(size=224, scale=(min_scale, max_scale), ratio=(1, 1))(frames)
        crop = rearrange(crop, 't c h w -> t h w c').numpy()
        for i,frame_index in enumerate(sample_list[index]):
            crop[i]=cv2.cvtColor(crop[i], cv2.COLOR_RGB2BGR)
            # cv2.imwrite('te')
            with env.begin(write=True) as txn:
                img = np.array(cv2.imencode('.png', crop[i])[1]).tobytes()
                path = video_path+'/{}/{}'.format(index, frame_index)
                txn.put(path.encode(), img)

if __name__ == '__main__':

    # os.chdir('/')
    file = os.path.dirname(os.path.abspath(__file__))
    anno_path = os.path.join(file, './data/odv_vqa')
    data_anno = ODVVQAReader(anno_path).read()
    video_info  = DatasetSplit.split(data_anno, './data/odv_vqa/tr_te_VQA_ODV.txt')
    train = video_info['train']
    test = video_info['test']
    # pool = Pool(6)
    # for i in tqdm(range(0, 40)):
    #     for video_info in data_anno:
    #         video_path = video_info['video_path']
    #         print(video_path)
    #         pool.apply_async(func=sampler, kwds={'video_path': video_path, 'epoch': i})
    # pool.close()
    # pool.join()

    for video_info in tqdm(train):
        video_path = video_info['video_path']
        sampler(video_path, True)

    with open('/data/ly/test/train.json', 'w') as f:
        json.dump(train_info, f)

    for video_info in tqdm(test):
        video_path = video_info['video_path']
        sampler(video_path, False)

    with open('/data/ly/test/test.json', 'w') as f:
        json.dump(test_info, f)
    # with env.begin(write=False) as txn:
    #     img = txn.get('/data/ly/VQA_ODV/Group1/G1AbandonedKingdom_ERP_7680x3840_fps30_qp27_45406k.mp4/0/211'.encode())  # 解码
    #     image_buf = np.frombuffer(img, dtype=np.uint8)
    #     img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
    #     cv2.imwrite('test.png', img)