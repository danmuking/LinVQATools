import yaml
import torch, torchvision
import decord
from decord import VideoReader
from decord import cpu, gpu
import glob
import os.path as osp
import numpy as np
from tqdm import tqdm
import cv2
from functools import lru_cache

import random
import copy

import skvideo.io

from data.vqa_odv import FusionDataset

random.seed(42)

decord.bridge.set_bridge("torch")

with open("./config/f3dvqa-b.yml", "r") as f:
    opt = yaml.safe_load(f)
print(opt['data']['train']['args'])
opt = opt['data']['train']['args']
for stype, sopt in opt["sample_types"].items():
    print(stype)
    print(sopt)

dataset = FusionDataset(opt)
dataset.__getitem__(0)

class FragmentSampleFrames:
    def __init__(self, fsize_t, fragments_t, frame_interval=1, num_clips=1, drop_rate=0., ):

        self.fragments_t = fragments_t
        self.fsize_t = fsize_t
        self.size_t = fragments_t * fsize_t
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.drop_rate = drop_rate

    def get_frame_indices(self, num_frames, train=False):

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

    def __call__(self, total_frames, train=False, start_index=0):
        frame_inds = []

        for i in range(self.num_clips):
            frame_inds += [self.get_frame_indices(total_frames)]

        frame_inds = np.concatenate(frame_inds)
        frame_inds = np.mod(frame_inds + start_index, total_frames)
        return frame_inds.astype(np.int32)

fragment = FragmentSampleFrames(32 // 8, 8, 2, 1)
print(fragment(240, True))
