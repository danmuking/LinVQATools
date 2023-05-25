import random
import time

import numpy as np
import torch

from utils.Equirec2Perspec import Equirectangular


class PlaneSpatialFragmentSampler:
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
        self.fragments_h = fragments_h
        self.fragments_w = fragments_w
        self.fsize_h = fsize_h
        self.fsize_w = fsize_w
        self.aligned = aligned
        self.nfrags = nfrags
        self.random = random
        self.random_upsample = random_upsample
        self.fallback_type = fallback_type

    def __call__(self, video, *args, **kwargs):
        # 采样图片的高
        size_h = self.fragments_h * self.fsize_h
        # 采样图片的长
        size_w = self.fragments_w * self.fsize_w
        ## video: [C,T,H,W]
        ## situation for images
        if video.shape[1] == 1:
            self.aligned = 1

        dur_t, res_h, res_w = video.shape[-3:]
        ratio = min(res_h / size_h, res_w / size_w)
        # 上采样
        # if self.fallback_type == "upsample" and ratio < 1:
        #     ovideo = video
        #     video = torch.nn.functional.interpolate(
        #         video / 255.0, scale_factor=1 / ratio, mode="bilinear"
        #     )
        #     video = (video * 255.0).type_as(ovideo)
        #
        # if self.random_upsample:
        #     randratio = random.random() * 0.5 + 1
        #     video = torch.nn.functional.interpolate(
        #         video / 255.0, scale_factor=randratio, mode="bilinear"
        #     )
        #     video = (video * 255.0).type_as(ovideo)

        assert dur_t % self.aligned == 0, "Please provide match vclip and align index"
        size = size_h, size_w

        ## make sure that sampling will not run out of the picture
        hgrids = torch.LongTensor(
            [min(res_h // self.fragments_h * i, res_h - self.fsize_h) for i in range(self.fragments_h)]
        )
        wgrids = torch.LongTensor(
            [min(res_w // self.fragments_w * i, res_w - self.fsize_w) for i in range(self.fragments_w)]
        )
        hlength, wlength = res_h // self.fragments_h, res_w // self.fragments_w
        if self.random:
            print("This part is deprecated. Please remind that.")
            if res_h > self.fsize_h:
                rnd_h = torch.randint(
                    res_h - self.fsize_h, (len(hgrids), len(wgrids), dur_t // self.aligned)
                )
            else:
                rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // self.aligned)).int()
            if res_w > self.fsize_w:
                rnd_w = torch.randint(
                    res_w - self.fsize_w, (len(hgrids), len(wgrids), dur_t // self.aligned)
                )
            else:
                rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // self.aligned)).int()
        else:
            if hlength > self.fsize_h:
                rnd_h = torch.randint(
                    hlength - self.fsize_h, (len(hgrids), len(wgrids), dur_t // self.aligned)
                )
            else:
                rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // self.aligned)).int()
            if wlength > self.fsize_w:
                rnd_w = torch.randint(
                    wlength - self.fsize_w, (len(hgrids), len(wgrids), dur_t // self.aligned)
                )
            else:
                rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // self.aligned)).int()

        target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
        # target_videos = []

        for i, hs in enumerate(hgrids):
            for j, ws in enumerate(wgrids):
                for t in range(dur_t // self.aligned):
                    t_s, t_e = t * self.aligned, (t + 1) * self.aligned
                    h_s, h_e = i * self.fsize_h, (i + 1) * self.fsize_h
                    w_s, w_e = j * self.fsize_w, (j + 1) * self.fsize_w
                    if self.random:
                        h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + self.fsize_h
                        w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + self.fsize_w
                    else:
                        h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + self.fsize_h
                        w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + self.fsize_w
                    target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
                                                                 :, t_s:t_e, h_so:h_eo, w_so:w_eo
                                                                 ]
        # target_videos.append(video[:,t_s:t_e,h_so:h_eo,w_so:w_eo])
        # target_video = torch.stack(target_videos, 0).reshape((dur_t // aligned, fragments, fragments,) + target_videos[0].shape).permute(3,0,4,1,5,2,6)
        # target_video = target_video.reshape((-1, dur_t,) + size) ## Splicing Fragments
        return target_video


class SphereSpatialFragmentSampler:
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
        self.fragments_h = fragments_h
        self.fragments_w = fragments_w
        self.fsize_h = fsize_h
        self.fsize_w = fsize_w
        self.aligned = aligned
        self.nfrags = nfrags
        self.random = random
        self.random_upsample = random_upsample
        self.fallback_type = fallback_type

    def __call__(self, video, *args, **kwargs):
        # 采样图片的高
        size_h = self.fragments_h * self.fsize_h
        # 采样图片的长
        size_w = self.fragments_w * self.fsize_w
        ## video: [C,T,H,W]
        ## situation for images
        if video.shape[1] == 1:
            self.aligned = 1

        dur_t, res_h, res_w = video.shape[-3:]
        ratio = min(res_h / size_h, res_w / size_w)
        # 上采样
        # if self.fallback_type == "upsample" and ratio < 1:
        #     ovideo = video
        #     video = torch.nn.functional.interpolate(
        #         video / 255.0, scale_factor=1 / ratio, mode="bilinear"
        #     )
        #     video = (video * 255.0).type_as(ovideo)
        #
        # if self.random_upsample:
        #     randratio = random.random() * 0.5 + 1
        #     video = torch.nn.functional.interpolate(
        #         video / 255.0, scale_factor=randratio, mode="bilinear"
        #     )
        #     video = (video * 255.0).type_as(ovideo)

        assert dur_t % self.aligned == 0, "Please provide match vclip and align index"
        size = size_h, size_w

        ## make sure that sampling will not run out of the picture
        hgrids = torch.LongTensor(
            [min(res_h // self.fragments_h * i, res_h - self.fsize_h) for i in range(self.fragments_h)]
        )
        wgrids = torch.LongTensor(
            [min(res_w // self.fragments_w * i, res_w - self.fsize_w) for i in range(self.fragments_w)]
        )
        hlength, wlength = res_h // self.fragments_h, res_w // self.fragments_w
        if self.random:
            print("This part is deprecated. Please remind that.")
            if res_h > self.fsize_h:
                rnd_h = torch.randint(
                    res_h - self.fsize_h, (len(hgrids), len(wgrids), dur_t // self.aligned)
                )
            else:
                rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // self.aligned)).int()
            if res_w > self.fsize_w:
                rnd_w = torch.randint(
                    res_w - self.fsize_w, (len(hgrids), len(wgrids), dur_t // self.aligned)
                )
            else:
                rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // self.aligned)).int()
        else:
            if hlength > self.fsize_h:
                rnd_h = torch.randint(
                    hlength - self.fsize_h, (len(hgrids), len(wgrids), dur_t // self.aligned)
                )
            else:
                rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // self.aligned)).int()
            if wlength > self.fsize_w:
                rnd_w = torch.randint(
                    wlength - self.fsize_w, (len(hgrids), len(wgrids), dur_t // self.aligned)
                )
            else:
                rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // self.aligned)).int()

        target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
        # target_videos = []

        for i, hs in enumerate(hgrids):
            for j, ws in enumerate(wgrids):
                for t in range(dur_t):
                    t_s, t_e = t, (t + 1)
                    h_s, h_e = i * self.fsize_h, (i + 1) * self.fsize_h
                    w_s, w_e = j * self.fsize_w, (j + 1) * self.fsize_w
                    frames = video[:, t_s:t_e, :, :]
                    frames = torch.squeeze(frames)
                    frames = frames.permute(1, 2, 0)
                    frames = frames.numpy()
                    equ = Equirectangular(frames)
                    frames = equ.GetPerspective(3, -180 + (360 / 6) * j, 60 - (120 / 6) * i, 32, 32)
                    frames = torch.from_numpy(frames)
                    frames = frames.permute(2, 0, 1)
                    frames = torch.unsqueeze(frames, dim=1)
                    target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = frames
        # target_videos.append(video[:,t_s:t_e,h_so:h_eo,w_so:w_eo])
        # target_video = torch.stack(target_videos, 0).reshape((dur_t // aligned, fragments, fragments,) + target_videos[0].shape).permute(3,0,4,1,5,2,6)
        # target_video = target_video.reshape((-1, dur_t,) + size) ## Splicing Fragments
        return target_video


class FastPlaneSpatialFragmentSampler:
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
        self.fragments_h = fragments_h
        self.fragments_w = fragments_w
        self.fsize_h = fsize_h
        self.fsize_w = fsize_w
        self.aligned = aligned
        self.nfrags = nfrags
        self.random = random
        self.random_upsample = random_upsample
        self.fallback_type = fallback_type

    def __call__(self, video, *args, **kwargs):
        device = video.device
        size_h = self.fragments_h * self.fsize_h
        size_w = self.fragments_w * self.fsize_w

        if video.shape[1] == 1:
            self.aligned = 1

        dur_t, res_h, res_w = video.shape[-3:]
        ratio = min(res_h / size_h, res_w / size_w)

        hgrids = torch.arange(0, res_h, res_h // self.fragments_h, device=device)[:self.fragments_h]
        wgrids = torch.arange(0, res_w, res_w // self.fragments_w, device=device)[:self.fragments_w]

        if self.random:
            print("This part is deprecated. Please remind that.")
            rnd_h = torch.randint(0, res_h - self.fsize_h, (self.fragments_h, self.fragments_w, dur_t // self.aligned),
                                  device=device)
            rnd_w = torch.randint(0, res_w - self.fsize_w, (self.fragments_h, self.fragments_w, dur_t // self.aligned),
                                  device=device)
        else:
            hlength = res_h // self.fragments_h
            wlength = res_w // self.fragments_w
            rnd_h = torch.zeros((self.fragments_h, self.fragments_w, dur_t // self.aligned), dtype=torch.int,
                                device=device)
            rnd_w = torch.zeros((self.fragments_h, self.fragments_w, dur_t // self.aligned), dtype=torch.int,
                                device=device)
            if hlength > self.fsize_h:
                rnd_h = torch.randint(hlength - self.fsize_h,
                                      (self.fragments_h, self.fragments_w, dur_t // self.aligned), device=device)
            if wlength > self.fsize_w:
                rnd_w = torch.randint(wlength - self.fsize_w,
                                      (self.fragments_h, self.fragments_w, dur_t // self.aligned), device=device)

        target_fragments = []
        for i, hs in enumerate(hgrids):
            for j, ws in enumerate(wgrids):
                for t in range(dur_t // self.aligned):
                    t_s, t_e = t * self.aligned, (t + 1) * self.aligned
                    h_s, h_e = i * self.fsize_h, (i + 1) * self.fsize_h
                    w_s, w_e = j * self.fsize_w, (j + 1) * self.fsize_w
                    if self.random:
                        h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + self.fsize_h
                        w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + self.fsize_w
                    else:
                        h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + self.fsize_h
                        w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + self.fsize_w
                    target_fragments.append(video[
                                                                 :, t_s:t_e, h_so:h_eo, w_so:w_eo
                                                                 ])

        target_video = torch.cat(target_fragments, dim=2).reshape(video.shape[0], dur_t, size_h, size_w)
        return target_video