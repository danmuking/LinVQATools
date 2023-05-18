from typing import Dict, List

from mmengine import MMLogger
from torch.utils.data import Dataset
from mmengine import DATASETS
import data.meta_reader as meta_reader
from data.meta_reader import AbstractReader
from data.split.dataset_split import DatasetSplit


@DATASETS.register_module()
class DefaultDataset(Dataset):
    def __init__(self, **opt):
        logger = MMLogger.get_instance('mmengine', log_level='INFO')

        # 数据集声明文件根路径
        if 'anno_root' not in opt:
            anno_root = './data/odv_vqa'
            logger.warning("anno_root参数未找到，默认为./data/odv_vqa")
        else:
            anno_root = opt['anno_root']

        # 训练集测试集划分文件路径
        split_file = opt.get("split_file", None)
        self.phase = opt.get("phase", 'train')

        # 读取数据集声明文件
        self.anno_reader: AbstractReader = getattr(meta_reader, opt['anno_reader'])(anno_root)
        # 数据集信息
        self.video_info = self.anno_reader.read()
        # 划分数据集
        self.video_info: Dict = DatasetSplit.split(self.video_info, split_file)

        # 用于获取的训练集/测试集信息
        self.data: List = self.video_info[self.phase]

    # TODO:
    def __getitem__(self, index):
        video_info = self.data[index]
        video_path = video_info["video_path"]
        score = video_info["score"]

        ## Read Original Frames
        ## Process Frames
        data, frame_inds = get_spatial_and_temporal_samples(video_path, self.sample_types, self.samplers,
                                                            self.phase == "train",
                                                            self.augment and (self.phase == "train"),
                                                            )

        for k, v in data.items():
            data[k] = ((v.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)

        data["num_clips"] = {}
        for stype, sopt in self.sample_types.items():
            data["num_clips"][stype] = sopt["num_clips"]
        data["frame_inds"] = frame_inds
        data["gt_label"] = score
        data["name"] = osp.basename(video_info["filename"])
        # print(data['fragments'].shape)
        return data

    def __len__(self):
        return len(self.data)
