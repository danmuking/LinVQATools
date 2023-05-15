from typing import Dict, List

import cv2
import torch
import decord
import glob
import os.path as osp
import numpy as np
import random
import copy

from decord import VideoReader

from data.sampler.fragment_sampler import TimeFragmentSampler

random.seed(42)

decord.bridge.set_bridge("torch")


def get_spatial_fragments(
        video,
        fragments_h=7,
        fragments_w=7,
        fsize_h=32,
        fsize_w=32,
        aligned=32,
        nfrags=1,
        random=False,
        random_upsample=False,
        fallback_type="upsample",
        **kwargs,
):
    """
    进行空间采样
    :param video: 视频矩阵
    :param fragments_h: 每一列的fragment数量
    :param fragments_w: 每一行的fragment数量
    :param fsize_h: 每个fragment的列pix
    :param fsize_w: 每个fragment的行pix
    :param aligned:
    :param nfrags:
    :param random:
    :param random_upsample:
    :param fallback_type:
    :param kwargs:
    :return:
    """
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    ## video: [C,T,H,W]
    ## situation for images
    if video.shape[1] == 1:
        aligned = 1

    # 帧数，高，宽
    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    # 上采样？
    if fallback_type == "upsample" and ratio < 1:
        ovideo = video
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=1 / ratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)
    # 随机上采样
    if random_upsample:
        randratio = random.random() * 0.5 + 1
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=randratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)

    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture
    # 计算采样网格
    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )
    # 每个网格的宽高
    hlength, wlength = res_h // fragments_h, res_w // fragments_w

    # 生成采样位置
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

    # 视频矩阵采样
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

def get_single_sample(
        video,
        sample_type="resize",
        **kwargs,
):
    if sample_type.startswith("fragments"):
        video = get_spatial_fragments(video, **kwargs)
    elif sample_type == "original":
        return video

    return video


def get_spatial_and_temporal_samples(
        video_path,
        sample_types,
        samplers,
        is_train=False,
        augment=False,
):
    """
    数据预处理，对视频帧进行时间和空间采样
    :param video_path: 视频路径
    :param sample_types: 采样类型
    :param samplers: 时间采样器
    :param is_train: 是否是训练
    :param augment: 数据增强
    :return:
    """
    video = {}
    vreader = VideoReader(video_path)

    # 时间采样
    ### Avoid duplicated video decoding!!! Important!!!!
    all_frame_inds = []
    frame_inds = {}
    for stype in samplers:
        # 获取用于训练的帧
        frame_inds[stype] = samplers[stype](len(vreader), is_train)
        all_frame_inds.append(frame_inds[stype])

    ### Each frame is only decoded one time!!!
    all_frame_inds = np.concatenate(all_frame_inds, 0)
    frame_dict = {idx: vreader[idx] for idx in np.unique(all_frame_inds)}

    for stype in samplers:
        imgs = [frame_dict[idx] for idx in frame_inds[stype]]
        video[stype] = torch.stack(imgs, 0).permute(3, 0, 1, 2)

    # 空间采样
    sampled_video = {}
    for stype, sopt in sample_types.items():
        sampled_video[stype] = get_single_sample(video[stype], stype,
                                                 **sopt)
    return sampled_video, frame_inds


class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        ## opt is a dictionary that includes options for video sampling

        super().__init__()

        self.video_infos = []
        self.tr_ts_file = opt["tr_ts_file"]
        # 元数据文件路径
        self.ann_file = opt["anno_file"]
        # 数据前缀
        self.data_prefix = opt["data_prefix"]
        self.opt = opt
        # 采样方式
        self.sample_types = opt["sample_types"]
        # 数据后端类型
        self.data_backend = opt.get("data_backend", "disk")
        # 数据增强
        self.augment = opt.get("augment", False)

        self.phase = opt["phase"]
        self.crop = opt.get("random_crop", False)

        # TODO: 计算数据集均值和标准差
        # self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        # self.std = torch.FloatTensor([58.395, 57.12, 57.375])

        # 初始化时间采样器
        self.samplers = {}
        for stype, sopt in opt["sample_types"].items():
            if "t_frag" not in sopt:
                # revised legacy temporal sampling
                self.samplers[stype] = TimeFragmentSampler(sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"])
            else:
                self.samplers[stype] = TimeFragmentSampler(sopt["clip_len"] // sopt["t_frag"], sopt["t_frag"],
                                                            sopt["frame_interval"], sopt["num_clips"])
            # 这应该只是一个示例，不是真实的采样帧
            print(stype + " branch sampled frames:", self.samplers[stype](240, self.phase == "train"))

        if isinstance(self.ann_file, list):
            self.video_infos = self.ann_file
        else:
            self.video_infos = self.get_video_infos()

    def refresh_hypers(self):
        if not hasattr(self, "initial_sample_types"):
            self.initial_sample_types = copy.deepcopy(self.sample_types)

        types = self.sample_types

        if "fragments_up" in types:
            ubh, ubw = self.initial_sample_types["fragments_up"]["fragments_h"] + 1, \
                       self.initial_sample_types["fragments_up"]["fragments_w"] + 1
            lbh, lbw = self.initial_sample_types["fragments"]["fragments_h"] + 1, \
                       self.initial_sample_types["fragments"]["fragments_w"] + 1
            dh, dw = types["fragments_up"]["fragments_h"], types["fragments_up"]["fragments_w"]

            types["fragments_up"]["fragments_h"] = random.randrange(max(lbh, dh - 1), min(ubh, dh + 2))
            types["fragments_up"]["fragments_w"] = random.randrange(max(lbw, dw - 1), min(ubw, dw + 2))

        if "resize_up" in types:
            types["resize_up"]["size_h"] = types["fragments_up"]["fragments_h"] * types["fragments_up"]["fsize_h"]
            types["resize_up"]["size_w"] = types["fragments_up"]["fragments_w"] * types["fragments_up"]["fsize_w"]

        self.sample_types.update(types)

        # print("Refreshed sample hyper-paremeters:", self.sample_types)

    def __getitem__(self, index):
        video_info = self.video_infos[index]
        # 视频文件路径
        filename = video_info["filename"]
        # dmos
        label = video_info["label"]

        ## Read Original Frames
        ## Process Frames
        data, frame_inds = get_spatial_and_temporal_samples(filename, self.sample_types, self.samplers,
                                                            self.phase == "train",
                                                            self.augment and (self.phase == "train"),
                                                            )

        for k, v in data.items():
            data[k] = ((v.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)

        data["num_clips"] = {}
        for stype, sopt in self.sample_types.items():
            data["num_clips"][stype] = sopt["num_clips"]
        data["frame_inds"] = frame_inds
        data["gt_label"] = label
        data["name"] = osp.basename(video_info["filename"])

        return data

    def __len__(self):
        return len(self.video_infos)

    def _get_scenses(self) -> List:
        """
        获取训练集或测试集的场景index
        :return: scenses
        """
        # 读取文件
        # 第一行为训练集index
        # 第二行为测试集index
        with open(self.tr_ts_file, 'r', encoding="utf-8") as f:
            if self.phase == 'train':
                scenses = f.readlines()[0].split()
            elif self.phase == "test":
                scenses = f.readlines()[1].split()
            else:
                assert RuntimeError("dataset's mode should be train or test, but get {}".format(self.mode))
        scenses = [int(i) for i in scenses]
        return scenses

    def get_video_infos(self) -> List:
        """
        获取数据集元数据
        :return:
        """
        # 获取采用的场景
        scenses = self._get_scenses()
        video_infos = []
        with open(self.ann_file, "r") as fin:
            for line in fin:
                line_split = line.strip().split()
                scense_id,_,reference_path,impaired_path,score,_,_,_ = line_split
                score = float(score)
                scense_id = int(scense_id)
                if scense_id not in scenses:
                    continue
                video_infos.append(dict(filename=impaired_path, label=score))
        return video_infos

