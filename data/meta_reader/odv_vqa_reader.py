import os
from typing import List

from .abstract_reader import AbstractReader


class ODVVQAReader(AbstractReader):
    """
    完成odv_vqa数据集声明数据的加载

    """

    def __init__(self, anno_root):
        """
        初始化ODVVQAReader，根路径下必须包含VQA_ODV.txt
        :param anno_root: 声明数据的根路径
        """
        self.anno_root = anno_root
        self.anno_file = os.path.join(anno_root, 'VQA_ODV.txt')

    def read(self):
        """
        加载声明文件
        :return:
        """
        return self.get_video_infos()

    def get_video_infos(self) -> List:
        """
        获取数据集元数据
        :return:
        """
        video_infos = []
        with open(self.anno_file, "r") as fin:
            for line in fin:
                line_split = line.strip().split()
                scene_id, _, reference_path, impaired_path, score, _, _, frame_num = line_split
                score = 1 - float(score)
                scene_id = int(scene_id)
                video_infos.append(dict(
                    scene_id=scene_id,
                    video_path=impaired_path,
                    ref_video_path=reference_path,
                    score=score,
                    frame_num=int(frame_num)))
        return video_infos
