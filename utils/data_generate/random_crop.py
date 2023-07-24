"""
根据时间块随机从视频上裁剪224*224大小的块
"""
import os
from typing import Dict, List
from torch.utils.data import Dataset
from data import logger, loader
from data.meta_reader import AbstractReader
from data.split.dataset_split import DatasetSplit
import data.meta_reader as meta_reader


class RandomCropDataset(Dataset):
    """
    双分支网络数据集加载器
    """

    def __init__(self, loader_opt: Dict, **opt: Dict):
        """
        初始化
        Args:
            loader_opt: 加载器参数
        """
        # 数据集声明文件根路径
        if 'anno_root' not in opt:
            anno_root = '/home/ly/code/LinVQATools/data/odv_vqa'
            logger.warning("anno_root参数未找到，默认为/home/ly/code/LinVQATools/data/odv_vqa")
        else:
            anno_root = opt['anno_root']

        # 训练集测试集划分文件路径
        split_file = opt.get("split_file", None)
        # 读取数据集声明文件
        self.anno_reader: AbstractReader = getattr(meta_reader, opt['anno_reader'])(anno_root)
        self.phase = opt.get("phase", 'train')
        # 数据集信息
        self.video_info = self.anno_reader.read()
        # 划分数据集
        self.video_info: Dict = DatasetSplit.split(self.video_info, split_file)
        # 用于获取的训练集/测试集信息
        self.data: List = self.video_info[self.phase]

        self.loader = getattr(loader, loader_opt['name'])(phase=self.phase,**loader_opt)

    def __getitem__(self, index):
        video_info = self.data[index]
        video_path = video_info["video_path"]
        score = video_info["score"]
        video = [self.loader.read(video_path), self.loader2.read(video_path)]

        data = {
            "inputs": video, "num_clips": {},
            # "frame_inds": frame_idxs,
            "gt_label": score,
            "name": os.path.basename(video_path)
        }
        return data

    def __len__(self):
        return len(self.data)
