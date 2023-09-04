"""
    实现faster vqa fragment数据预处理
"""
import os
import random
import time
from multiprocessing import Pool

import cv2
import numpy as np
import torch
import decord
from decord import VideoReader
from einops import rearrange
from tqdm import tqdm

from data.meta_reader import ODVVQAReader
random.seed(time.perf_counter())

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
            # rnd_t = np.random.randint(
            #     0, tlength - self.fsize_t * self.frame_interval, size=len(tgrids)
            # )
            rnd_t = rand.randint(0, tlength - self.fsize_t * self.frame_interval, size=len(tgrids))
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
        # print(frame_inds)
        return frame_inds.astype(np.int32)


def makedir(path: str):
    dir_path = path
    if os.path.exists(dir_path):
        pass
    else:
        os.makedirs(dir_path)


def get_save_path(path, frame_num, epoch):
    path = path.split('/')
    path.insert(3, 'imp_ref')
    path.insert(4, str(epoch))
    path[0] = "/data"
    path[1] = ""
    path = os.path.join(*path)[:-4]
    makedir(path)
    img_path = os.path.join(path, '{}.png'.format(frame_num))
    return img_path


frame_sampler = FragmentSampleFrames(fsize_t=8, fragments_t=2, frame_interval=2, num_clips=1, )


def sampler(video_path: str, ref_path: str, epoch: int):
    vreader = VideoReader(video_path)
    # frame_index = [x for x in range(len(vreader))]

    frame_index = frame_sampler(len(vreader))

    ref_vreader = VideoReader(ref_path)

    for frame_num in frame_index:
        save_path = get_save_path(video_path, frame_num, epoch)
        img = vreader[frame_num]
        img = rearrange(img, 'h w c -> c h w')

        ref_img = ref_vreader[frame_num]
        ref_img = rearrange(ref_img, 'h w c -> c h w')

        fragments_h = 7
        fragments_w = 7
        fsize_h = 32
        fsize_w = 32

        # 采样图片的高
        size_h = fragments_h * fsize_h
        # 采样图片的长
        size_w = fragments_w * fsize_w

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
                hlength - fsize_h, (len(hgrids), len(wgrids))
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids)).int())
        if wlength > fsize_w:
            rnd_w = torch.randint(
                wlength - fsize_w, (len(hgrids), len(wgrids))
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids)).int())

        ###########################################################
        # 受损视频
        ###########################################################
        target_img = torch.zeros((3, 224, 224))
        for i, hs in enumerate(hgrids):
            for j, ws in enumerate(wgrids):
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                h_so, h_eo = hs + rnd_h[i][j], hs + rnd_h[i][j] + fsize_h
                w_so, w_eo = ws + rnd_w[i][j], ws + rnd_w[i][j] + fsize_w
                # print(h_so, w_so)
                target_img[:, h_s:h_e, w_s:w_e] = img[:, h_so:h_eo, w_so:w_eo]
        target_img = rearrange(target_img, 'c h w -> h w c ')
        target_img = target_img.numpy()
        target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, target_img)
        print('{}已保存'.format(save_path))

        ###########################################################
        # 参考视频
        ###########################################################
        target_img = torch.zeros((3, 224, 224))
        for i, hs in enumerate(hgrids):
            for j, ws in enumerate(wgrids):
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                h_so, h_eo = hs + rnd_h[i][j], hs + rnd_h[i][j] + fsize_h
                w_so, w_eo = ws + rnd_w[i][j], ws + rnd_w[i][j] + fsize_w
                # print(h_so, w_so)
                target_img[:, h_s:h_e, w_s:w_e] = ref_img[:, h_so:h_eo, w_so:w_eo]
        target_img = rearrange(target_img, 'c h w -> h w c ')
        target_img = target_img.numpy()
        target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
        save_path = save_path.split('/')
        save_path.insert(-1, 'ref/')
        # print(os.path.join('', *save_path[:-1]))
        makedir(os.path.join('/', *save_path[:-1]))
        save_path = os.path.join('/', *save_path)
        cv2.imwrite(save_path, target_img)
        print('{}已保存'.format(save_path))


if __name__ == '__main__':
    # os.chdir('/')
    file = os.path.dirname(os.path.abspath(__file__))
    anno_path = os.path.join(file, './data/odv_vqa')
    data_anno = ODVVQAReader(anno_path).read()
    pool = Pool(4)
    for i in tqdm(range(0, 40)):
        for video_info in data_anno:
            v_path = video_info['video_path']
            ref_video_path = video_info['ref_video_path']
            pool.apply_async(func=sampler, kwds={'video_path': v_path, 'ref_path': ref_video_path, 'epoch': i})
    pool.close()
    pool.join()
    # sampler(data_anno[0]['video_path'], data_anno[0]['ref_video_path'], 0)
