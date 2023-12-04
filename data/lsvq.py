import os.path
from typing import List, Dict

import decord
from decord import VideoReader
from decord import cpu, gpu
import glob
import os.path as osp
import numpy as np
import torch, torchvision
from mmengine import DATASETS
from tqdm import tqdm
import cv2
import data.shuffler as shuffler
from functools import lru_cache

import random
import copy

import skvideo.io

random.seed(42)

decord.bridge.set_bridge("torch")


def get_spatial_fragments(
        video,
        fragments_h=7,
        fragments_w=7,
        fsize_h=32,
        fsize_w=32,
        aligned=16,
        nfrags=1,
        random=False,
        random_upsample=False,
        fallback_type="upsample",
        **kwargs,
):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    ## video: [C,T,H,W]
    ## situation for images
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if fallback_type == "upsample" and ratio < 1:
        ovideo = video
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=1 / ratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)

    if random_upsample:
        randratio = random.random() * 0.5 + 1
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=randratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)

    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture
    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )
    hlength, wlength = res_h // fragments_h, res_w // fragments_w

    if random:
        print("This part is deprecated. Please remind that.")
        if res_h > fsize_h:
            rnd_h = torch.randint(
                res_h - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if res_w > fsize_w:
            rnd_w = torch.randint(
                res_w - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    else:
        if hlength > fsize_h:
            rnd_h = torch.randint(
                hlength - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if wlength > fsize_w:
            rnd_w = torch.randint(
                wlength - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    # target_videos = []

    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                if random:
                    h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + fsize_w
                else:
                    h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
                                                             :, t_s:t_e, h_so:h_eo, w_so:w_eo
                                                             ]
    # target_videos.append(video[:,t_s:t_e,h_so:h_eo,w_so:w_eo])
    # target_video = torch.stack(target_videos, 0).reshape((dur_t // aligned, fragments, fragments,) + target_videos[0].shape).permute(3,0,4,1,5,2,6)
    # target_video = target_video.reshape((-1, dur_t,) + size) ## Splicing Fragments
    return target_video


@lru_cache
def get_resize_function(size_h, size_w, target_ratio=1, random_crop=False):
    if random_crop:
        return torchvision.transforms.RandomResizedCrop((size_h, size_w), scale=(0.40, 1.0))
    if target_ratio > 1:
        size_h = int(target_ratio * size_w)
        assert size_h > size_w
    elif target_ratio < 1:
        size_w = int(size_h / target_ratio)
        assert size_w > size_h
    return torchvision.transforms.Resize((size_h, size_w))


def get_resized_video(
        video,
        size_h=224,
        size_w=224,
        random_crop=False,
        arp=False,
        **kwargs,
):
    video = video.permute(1, 0, 2, 3)
    resize_opt = get_resize_function(size_h, size_w,
                                     video.shape[-2] / video.shape[-1] if arp else 1,
                                     random_crop)
    video = resize_opt(video).permute(1, 0, 2, 3)
    return video


def get_arp_resized_video(
        video,
        short_edge=224,
        train=False,
        **kwargs,
):
    if train:  ## if during training, will random crop into square and then resize
        res_h, res_w = video.shape[-2:]
        ori_short_edge = min(video.shape[-2:])
        if res_h > ori_short_edge:
            rnd_h = random.randrange(res_h - ori_short_edge)
            video = video[..., rnd_h:rnd_h + ori_short_edge, :]
        elif res_w > ori_short_edge:
            rnd_w = random.randrange(res_w - ori_short_edge)
            video = video[..., :, rnd_h:rnd_h + ori_short_edge]
    ori_short_edge = min(video.shape[-2:])
    scale_factor = short_edge / ori_short_edge
    ovideo = video
    video = torch.nn.functional.interpolate(
        video / 255.0, scale_factors=scale_factor, mode="bilinear"
    )
    video = (video * 255.0).type_as(ovideo)
    return video


def get_arp_fragment_video(
        video,
        short_fragments=7,
        fsize=32,
        train=False,
        **kwargs,
):
    if train:  ## if during training, will random crop into square and then get fragments
        res_h, res_w = video.shape[-2:]
        ori_short_edge = min(video.shape[-2:])
        if res_h > ori_short_edge:
            rnd_h = random.randrange(res_h - ori_short_edge)
            video = video[..., rnd_h:rnd_h + ori_short_edge, :]
        elif res_w > ori_short_edge:
            rnd_w = random.randrange(res_w - ori_short_edge)
            video = video[..., :, rnd_h:rnd_h + ori_short_edge]
    kwargs["fsize_h"], kwargs["fsize_w"] = fsize, fsize
    res_h, res_w = video.shape[-2:]
    if res_h > res_w:
        kwargs["fragments_w"] = short_fragments
        kwargs["fragments_h"] = int(short_fragments * res_h / res_w)
    else:
        kwargs["fragments_h"] = short_fragments
        kwargs["fragments_w"] = int(short_fragments * res_w / res_h)
    return get_spatial_fragments(video, **kwargs)


def get_cropped_video(
        video,
        size_h=224,
        size_w=224,
        **kwargs,
):
    kwargs["fragments_h"], kwargs["fragments_w"] = 1, 1
    kwargs["fsize_h"], kwargs["fsize_w"] = size_h, size_w
    return get_spatial_fragments(video, **kwargs)


def get_single_sample(
        video,
        sample_type="resize",
        **kwargs,
):
    if sample_type.startswith("resize"):
        video = get_resized_video(video, **kwargs)
    elif sample_type.startswith("arp_resize"):
        video = get_arp_resized_video(video, **kwargs)
    elif sample_type.startswith("fragments"):
        video = get_spatial_fragments(video, **kwargs)
    elif sample_type.startswith("arp_fragments"):
        video = get_arp_fragment_video(video, **kwargs)
    elif sample_type.startswith("crop"):
        video = get_cropped_video(video, **kwargs)
    elif sample_type == "original":
        return video

    return video


def get_spatial_samples(
        video,
        random_crop=0,  ## 1: ARP-kept Crop; 2: Square-like Crop
        sample_types={"resize": {}, "fragments": {}},  ## resize | arp_resize | crop | fragments
):
    if random_crop == 1:
        print("Alert!")
        ## Random Crop but keep the ARP
        res_h, res_w = video.shape[-2:]
        rnd_ratio = random.random() * 0.2 + 0.8
        new_h, new_w = int(rnd_ratio * res_h), int(rnd_ratio * res_w)
        rnd_h = random.randrange(res_h - new_h)
        rnd_w = random.randrange(res_w - new_w)
        video = video[..., rnd_h:rnd_hn + new_h, rnd_w:rnd_w + new_w]
        ovideo = video
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=random.random() * 0.3 + 1.0, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)

    if random_crop == 2:
        ## Random Crop into a Size similar to Square
        rnd_ratio = random.random() * 0.2 + 0.8
        res_h, res_w = video.shape[-2:]
        new_h = new_w = int(rnd_ratio * min(res_h, res_w))
        rnd_h = random.randrange(res_h - new_h)
        rnd_w = random.randrange(res_w - new_w)
        video = video[..., rnd_h:rnd_h + new_h, rnd_w:rnd_w + new_w]
    sampled_video = {}
    for sample_type, arg in sample_types.items():
        sampled_video[sample_type] = get_single_sample(video, sample_type,
                                                       **arg)
    return sampled_video


def get_spatial_and_temporal_samples(
        video_path,
        samplers,
):
    vreader = VideoReader(video_path)
    ### Avoid duplicated video decoding!!! Important!!!!
    all_frame_inds = []
    frame_inds = samplers(len(vreader))
    all_frame_inds.append(frame_inds)

    ### Each frame is only decoded one time!!!
    all_frame_inds = np.concatenate(all_frame_inds, 0)
    frame_dict = {idx: vreader[idx] for idx in np.unique(all_frame_inds)}

    imgs = [frame_dict[idx] for idx in frame_inds]
    video = torch.stack(imgs, 0).permute(3, 0, 1, 2)

    sampled_video = get_single_sample(video, 'fragments')
    return sampled_video, frame_inds


class SampleFrames:
    def __init__(self, clip_len, frame_interval=1, num_clips=1):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips
            )
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(num_frames - ori_clip_len + 1, size=self.num_clips)
            )
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        return clip_offsets

    def _get_test_clips(self, num_frames, start_index=0):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int32)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)
        return clip_offsets

    def __call__(self, total_frames, train=False, start_index=0):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if train:
            clip_offsets = self._get_train_clips(total_frames)
        else:
            clip_offsets = self._get_test_clips(total_frames)
        frame_inds = (
                clip_offsets[:, None]
                + np.arange(self.clip_len)[None, :] * self.frame_interval
        )
        frame_inds = np.concatenate(frame_inds)

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        frame_inds = np.mod(frame_inds, total_frames)
        frame_inds = np.concatenate(frame_inds) + start_index
        return frame_inds.astype(np.int32)


import numpy as np
import random


class FragmentSampleFrames:
    def __init__(self, fsize_t, fragments_t, frame_interval=1, num_clips=1, drop_rate=0., ):

        self.fragments_t = fragments_t
        self.fsize_t = fsize_t
        self.size_t = fragments_t * fsize_t
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.drop_rate = drop_rate

    def get_frame_indices(self, num_frames):

        tgrids = np.array(
            [num_frames // self.fragments_t * i for i in range(self.fragments_t)],
            dtype=np.int32,
        )
        tlength = num_frames // self.fragments_t

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

    def __call__(self, total_frames, start_index=0):
        frame_inds = []

        for i in range(self.num_clips):
            frame_inds += [self.get_frame_indices(total_frames)]

        frame_inds = np.concatenate(frame_inds)
        frame_inds = np.mod(frame_inds + start_index, total_frames)
        return frame_inds.astype(np.int32)


@DATASETS.register_module()
class LSVQDataset(torch.utils.data.Dataset):
    def __init__(self, root='/data/ly/LSVQ', phase='train', norm=True,argument: List[Dict] = [],clip=1,):
        super().__init__()
        self.phase = phase
        self.norm = norm
        anno_file_path = "train.txt" if phase == 'train' else 'test.txt'
        self.ann_file = os.path.join(root, anno_file_path)
        self.data_prefix = root
        # 数据增强
        self.argument = [getattr(shuffler, item['name'])(**item) for item in argument]
        self.clip = clip

        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
        self.std = torch.FloatTensor([0.229, 0.224, 0.225])

        self.samplers = FragmentSampleFrames(4, 4, 2)
        self.video_infos = []
        with open(self.ann_file, "r") as fin:
            for line in fin:
                line_split = line.strip().split(",")
                filename, _, _, label = line_split
                label = float(label)
                filename = osp.join(self.data_prefix, filename)
                self.video_infos.append(dict(filename=filename, label=label))

    def __getitem__(self, index):
        video_info = self.video_infos[index]
        video_path = video_info["filename"]
        score = video_info["label"]/100

        ## Read Original Frames
        ## Process Frames
        videos = []
        for i in range(self.clip):
            video, frame_inds = get_spatial_and_temporal_samples(video_path, self.samplers)
            # 视频后处理
            for item in self.argument:
                video = item(video)
            videos.append(video)
        video = torch.stack(videos, dim=0)
        if self.norm:
            video = video / 255.0
            video = ((video.permute(0, 2, 3, 4, 1) - self.mean) / self.std).permute(0, 4, 1, 2, 3)

        data = {
            "inputs": video, "num_clips": {},
            # "frame_inds": frame_idxs,
            "gt_label": score,
            "name": osp.basename(video_path)
        }
        return data

    def __len__(self):
        return len(self.video_infos)
