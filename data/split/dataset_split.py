import random
from typing import List


class DatasetSplit:
    """
    数据集划分器，将数据集划分为训练集和测试集
    """
    # 训练集和测试集的比例
    scale = 0.8

    @staticmethod
    def split(data: List, split_file=None):
        """
        将根据场景将数据划分为训练集和测试集
        """
        # 如果split_file非空，则读取文件，第一行是训练集索引，第二行是测试集索引
        if split_file is not None:
            with open(split_file, 'r') as f:
                idxs = f.readlines()
            for i, line in enumerate(idxs):
                line = line.strip().split()
                idxs[i] = [int(item) for item in line]
        else:
            max_idx = max([item['scene_id'] for item in data]) + 1
            idxs = [i for i in range(max_idx)]
            random.shuffle(idxs)
            idxs = [
                idxs[:int(max_idx * 0.8)], idxs[int(max_idx * 0.8):]
            ]

        train_data = []
        test_data = []
        for item in data:
            if item['scene_id'] in idxs[0]:
                train_data.append(item)
            else:
                test_data.append(item)
        return dict(train=train_data, test=test_data)
