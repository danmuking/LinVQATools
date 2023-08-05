"""
    实现fragment数据预处理
"""
import os
from multiprocessing.pool import Pool

import cv2
import torch
import decord
from decord import VideoReader
from einops import rearrange
from tqdm import tqdm

from data.meta_reader import ODVVQAReader

decord.bridge.set_bridge("torch")


def makedir(path: str):
    dir_path = path
    if os.path.exists(dir_path):
        pass
    else:
        os.makedirs(dir_path)


def get_save_path(video_path,frame_num):
    video_path = video_path.split('/')
    video_path.insert(3, 'fragment')
    video_path[0] = "/data"
    video_path[1] = ""
    video_path = os.path.join(*video_path)[:-4]
    makedir(video_path)
    img_path = os.path.join(video_path,'{}.png'.format(frame_num))
    return img_path


def sampler(video_path: str):
    vreader = VideoReader(video_path)
    for frame_num in range(len(vreader)):
        save_path = get_save_path(video_path, frame_num)
        img = vreader[frame_num]
        img = rearrange(img, 'h w c -> c h w')
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


if __name__ == '__main__':

    # os.chdir('/')
    file = os.path.dirname(os.path.abspath(__file__))
    anno_path = os.path.join(file, './data/odv_vqa')
    data_anno = ODVVQAReader(anno_path).read()
    pool = Pool(16)
    for video_info in tqdm(data_anno):
        video_path = video_info['video_path']
        print(video_path)
        pool.apply_async(func=sampler, kwds={'video_path':video_path})
    pool.close()
    pool.join()
