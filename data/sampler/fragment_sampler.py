import numpy as np
import random

import torch

from utils.sphere import viewport_alignment


class TimeFragmentSampler:
    """
    视频时间采样器
    """
    def __init__(self, fsize_t, fragments_t, frame_interval=1, num_clips=1, drop_rate=0., ):
        """
        :param fsize_t: 每一个fragment的长度
        :param fragments_t: 可用于采样的时间长度
        :param frame_interval: 帧间隔
        :param num_clips: clip数量
        :param drop_rate:
        """
        self.fragments_t = fragments_t
        self.fsize_t = fsize_t
        self.size_t = fragments_t * fsize_t
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.drop_rate = drop_rate

    def get_frame_indices(self, num_frames, train=False):
        """
        获取时间采样的帧序列
        :param num_frames: 帧总数
        :param train:
        :return: 包含clip采样帧索引的list
        """
        # 时间采样网格
        tgrids = np.array(
            [num_frames // self.fragments_t * i for i in range(self.fragments_t)],
            dtype=np.int32,
        )
        tlength = num_frames // self.fragments_t

        # 如果可采样时间大于每个fragment的时间
        if tlength > self.fsize_t * self.frame_interval:
            rnd_t = np.random.randint(
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
        """
        :param total_frames: 总帧数
        :param train: 是否为训练模式
        :param start_index: 开始帧
        :return: 包含视频采样索引的list
        """
        frame_inds = []

        for i in range(self.num_clips):
            frame_inds += [self.get_frame_indices(total_frames)]

        frame_inds = np.concatenate(frame_inds)
        frame_inds = np.mod(frame_inds + start_index, total_frames)
        return frame_inds.astype(np.int32)


class SpatialFragmentSampler:
    """
    视频空间采样
    """
    def __init__(
            self,
            fragments_h=7,
            fragments_w=7,
            fsize_h=32,
            fsize_w=32,
            aligned=32,
            nfrags=1,
            random=False,
            random_upsample=False,
            fallback_type="upsample",
            **kwargs
    ):
        """
        :param fragments_h: 每一列的fragment数量
        :param fragments_w: 每一行的fragment数量
        :param fsize_h: 每个fragment的列pix
        :param fsize_w: 每个fragment的行pix
        :param aligned:
        :param nfrags:
        :param random:
        :param random_upsample:
        :param fallback_type:
        """
        self.fragments_h = fragments_h
        self.fragments_w = fragments_w
        self.fsize_h = fsize_h
        self.fsize_w = fsize_w
        self.aligned = aligned
        self.nfrags = nfrags
        self.random = random
        self.random_upsample = random_upsample
        self.fallback_type = fallback_type

    def __call__(self, video, **kwargs):
        size_h =  self.fragments_h *  self.fsize_h
        size_w =  self.fragments_w *  self.fsize_w
        ## video: [C,T,H,W]
        ## situation for images
        if video.shape[1] == 1:
            self.aligned = 1

        # 帧数，高，宽
        dur_t, res_h, res_w = video.shape[-3:]
        ratio = min(res_h / size_h, res_w / size_w)
        # 上采样？
        if  self.fallback_type == "upsample" and ratio < 1:
            ovideo = video
            video = torch.nn.functional.interpolate(
                video / 255.0, scale_factor=1 / ratio, mode="bilinear"
            )
            video = (video * 255.0).type_as(ovideo)
        # 随机上采样
        if  self.random_upsample:
            randratio = random.random() * 0.5 + 1
            video = torch.nn.functional.interpolate(
                video / 255.0, scale_factor=randratio, mode="bilinear"
            )
            video = (video * 255.0).type_as(ovideo)

        assert dur_t % self.aligned == 0, "Please provide match vclip and align index"
        size = size_h, size_w

        ## make sure that sampling will not run out of the picture
        # 计算采样网格
        hgrids = torch.LongTensor(
            [min(res_h //  self.fragments_h * i, res_h -  self.fsize_h) for i in range( self.fragments_h)]
        )
        wgrids = torch.LongTensor(
            [min(res_w //  self.fragments_w * i, res_w -  self.fsize_w) for i in range( self.fragments_w)]
        )
        # 每个网格的宽高
        hlength, wlength = res_h //  self.fragments_h, res_w //  self.fragments_w

        # 生成采样位置
        if hlength >  self.fsize_h:
            rnd_h = torch.randint(
                hlength -  self.fsize_h, (len(hgrids), len(wgrids), dur_t // self.aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // self.aligned)).int()
        if wlength >  self.fsize_w:
            rnd_w = torch.randint(
                wlength -  self.fsize_w, (len(hgrids), len(wgrids), dur_t // self.aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // self.aligned)).int()

        target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
        # target_videos = []

        # 视频矩阵采样
        # for i, hs in enumerate(hgrids):
        #     for j, ws in enumerate(wgrids):
        #         for t in range(dur_t // aligned):
        #             t_s, t_e = t * aligned, (t + 1) * aligned
        #             h_s, h_e = i *  self.fsize_h, (i + 1) *  self.fsize_h
        #             w_s, w_e = j *  self.fsize_w, (j + 1) *  self.fsize_w
        #             temp = viewport_alignment(video[:,t,:,:],0,0,(32,32))
        #             # if random:
        #             #     h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] +  self.fsize_h
        #             #     w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] +  self.fsize_w
        #             # else:
        #             #     h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] +  self.fsize_h
        #             #     w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] +  self.fsize_w
        #             target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
        #                                                          :, t_s:t_e, h_so:h_eo, w_so:w_eo
        #                                                          ]
        video = video.float()
        for t in range(dur_t):
            frame = video[:,t,:,:]
            frame = torch.squeeze(frame)
            fragments = viewport_alignment(frame,torch.zeros(49),torch.zeros(49),(32,32))
            for i, hs in enumerate(hgrids):
                for j, ws in enumerate(wgrids):
                    h_s, h_e = i * self.fsize_h, (i + 1) * self.fsize_h
                    w_s, w_e = j * self.fsize_w, (j + 1) * self.fsize_w
                    target_video[:, t, h_s:h_e, w_s:w_e] = torch.squeeze(fragments[7*i+j])

        # target_videos.append(video[:,t_s:t_e,h_so:h_eo,w_so:w_eo])
        # target_video = torch.stack(target_videos, 0).reshape((dur_t // aligned, fragments, fragments,) + target_videos[0].shape).permute(3,0,4,1,5,2,6)
        # target_video = target_video.reshape((-1, dur_t,) + size) ## Splicing Fragments
        return target_video