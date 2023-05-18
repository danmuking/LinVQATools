from typing import Dict

from mmengine import MMLogger
from torch.utils.data import Dataset
from mmengine import DATASETS
import data.meta_reader as meta_reader
from data.meta_reader import AbstractReader


@DATASETS.register_module()
class DefaultDataset(Dataset):
    def __init__(self, **opt):
        logger = MMLogger.get_instance('mmengine', log_level='INFO')

        if 'anno_file' not in opt:
            anno_file = './data/odv_vqa'
            logger.warning("anno_file参数未找到，默认为./data/odv_vqa")
        else:
            anno_file = opt['anno_file']
        self.anno_reader: AbstractReader = getattr(meta_reader, opt['anno_reader'])(anno_file)
        self.video_info: Dict = self.anno_reader.read()
        print(self.video_info)
