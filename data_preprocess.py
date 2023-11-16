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
    video_path.insert(3, '4frame_4interval')
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
    # frame_index = [x for x in range(len(vreader))]
    frame_sampler = FragmentSampleFrames(fsize_t=4, fragments_t=1, frame_interval=4, num_clips=1, )
    frame_index = frame_sampler(len(vreader))

    fragments_h = 7
    fragments_w = 7
    fsize_h = 32
    fsize_w = 32
    # 采样图片的高
    size_h = fragments_h * fsize_h
    # 采样图片的长
    size_w = fragments_w * fsize_w
    img = vreader[0]
    img = rearrange(img, 'h w c -> c h w ')
    res_h, res_w = img.shape[-2:]
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture
    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )
    hlength, wlength = res_h // fragments_h, res_w // fragments_w
    if hlength > fsize_h:
        rnd_h = torch.randint(
            hlength - fsize_h, (len(hgrids), len(wgrids), 8)
        )
    else:
        rnd_h = torch.zeros((len(hgrids), len(wgrids)).int())
    if wlength > fsize_w:
        rnd_w = torch.randint(
            wlength - fsize_w, (len(hgrids), len(wgrids), 8)
        )
    else:
        rnd_w = torch.zeros((len(hgrids), len(wgrids)).int())

    # softpool = SoftPool2d()

    for index, frame_num in enumerate(frame_index):
        save_path = get_save_path(video_path, frame_num, epoch)
        img = vreader[frame_num]
        img = rearrange(img, 'h w c -> c h w ')
        # img = torchvision.transforms.CenterCrop((h, w))(img)
        target_img = torch.zeros((3, 224, 224))

        for i, hs in enumerate(hgrids):
            for j, ws in enumerate(wgrids):
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                h_so, h_eo = hs + rnd_h[i][j][int(index/4)], hs + rnd_h[i][j][int(index/4)] + fsize_h
                w_so, w_eo = ws + rnd_w[i][j][int(index/4)], ws + rnd_w[i][j][int(index/4)] + fsize_w
                # print(i,j,int(index/8))
                # print(rnd_h[i][j][int(index/8)],rnd_w[i][j][int(index/8)])
                # print(h_so, w_so)
                target_img[:, h_s:h_e, w_s:w_e] = img[:, h_so:h_eo, w_so:w_eo]

        # target_img = rearrange(target_img, '(b c) h w -> b c h w', b=1)
        # target_img = target_img / 255
        # target_img = softpool(target_img)
        # target_img = rearrange(target_img, 'b c h w -> (b c) h w', b=1)

        target_img = rearrange(target_img, 'c h w -> h w c ')
        target_img = target_img.numpy()
        target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, target_img.astype('uint8'))
        print('{}已保存'.format(save_path))


if __name__ == '__main__':

    # os.chdir('/')
    file = os.path.dirname(os.path.abspath(__file__))
    anno_path = os.path.join(file, './data/odv_vqa')
    data_anno = ODVVQAReader(anno_path).read()
    num = 160
    # pool = Pool(8)
    # for i in tqdm(range(0, num)):
    #     for video_info in data_anno:
    #         video_path = video_info['video_path']
    #         print(video_path)
    #         pool.apply_async(func=sampler, kwds={'video_path': video_path, 'epoch': i})
    # pool.close()
    # pool.join()
    #
    #
    #     # print(dir)
    #
    #
    # info = dict()
    # for i in range(num):
    #     path = "/data/ly/4frame_4interval/{}/VQA_ODV".format(i)
    #     for group_name in os.listdir(path):
    #         video_group_path = os.path.join(path, group_name)
    #         if info.get(group_name, None) is None:
    #             info[group_name] = dict()
    #         for video_name in os.listdir(video_group_path):
    #             video_path = os.path.join(video_group_path, video_name)
    #             if info[group_name].get(video_name, None) is None:
    #                 info[group_name][video_name] = []
    #             frame_list = []
    #             for video_frame_name in os.listdir(video_path):
    #                 video_frame_path = os.path.join(video_path, video_frame_name)
    #                 # print(video_frame_path)
    #                 frame_list.append(int(video_frame_name[:-4]))
    #             frame_list.sort()
    #             info[group_name][video_name].append(frame_list)
    # with open('/data/ly/4frame_4interval/info.json', 'w') as f:
    #     json.dump(info, f)

    import lmdb

    env = lmdb.open('/data/ly/resize', map_size=1099511627776)
    for i in tqdm(range(num)):
        path = "/data/ly/resize/{}/VQA_ODV".format(i)
        for group_name in os.listdir(path):
            video_group_path = os.path.join(path, group_name)
            for video_name in os.listdir(video_group_path):
                video_path = os.path.join(video_group_path, video_name)
                frame_list = []
                for video_frame_name in os.listdir(video_path):
                    video_frame_path = os.path.join(video_path, video_frame_name)
                    with open(video_frame_path, 'rb') as f:
                        # 读取图像文件的二进制格式数据
                        image_bin = f.read()
                    with env.begin(write=True) as txn:
                        txn.put(video_frame_path.encode(), image_bin)
    env.close()