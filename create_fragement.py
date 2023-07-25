"""
    实现fragment数据预处理
"""
import os

import cv2
import numpy as np
from torch.utils.data import DataLoader
# import cv2
# import numpy as np
from tqdm import tqdm

from data.default_dataset import DefaultDataset


def makedir(path: str):
    # dir_path = os.path.dirname(path)
    dir_path = path
    if (os.path.exists(dir_path)):
        pass
    else:
        os.makedirs(dir_path)


os.chdir('/')
frame_sampler = dict(
    name='FragmentSampleFrames',
    fsize_t=32 // 8,
    fragments_t=8,
    clip_len=32,
    frame_interval=4,
    t_frag=8,
    num_clips=1,
)
spatial_sampler = dict(
    name='PlaneSpatialFragmentSampler',
    fragments_h=1,
    fragments_w=1,
    fsize_h=32 * 7,
    fsize_w=32 * 7,
    aligned=8,
)
shuffler = dict(
    name='BaseShuffler',
)
post_sampler = dict(
    name='PostProcessSampler',
    num=4
)
dataset = DefaultDataset(anno_reader='ODVVQAReader',
                         split_file='/home/ly/code/LinVQATools/data/odv_vqa/tr_te_VQA_ODV.txt',
                         frame_sampler=frame_sampler, spatial_sampler=spatial_sampler, prefix='',
                         shuffler=shuffler, post_sampler=post_sampler, norm=False,phase='test')
dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
for j in range(40):
    index = 0
    for item in tqdm(dataloader):
        data = item
        video_info = dataset.data[index]
        video_path = video_info["video_path"]
        video_path = video_path.split('/')
        # print(video_path)
        # video_path[0] = '/data/ly'
        video_path.insert(3, 'crop')
        video_path.insert(4, '{}'.format(j))
        # video_path[0] = "G:\\"
        # video_path[1] = ""
        video_path = os.path.join("/", *video_path)[:-4]
        # video_path = os.path.join('D:/code/LinVQATools/data/odv_vqa/',video_path)
        print(video_path)
        makedir(video_path)
        video = data['inputs']
        video = video[0]
        video = video.permute(1, 2, 3, 0).numpy().astype(np.uint8)
        print(data['name'])
        print(video.shape)
        for i in range(video.shape[0]):
            img = video[i]
            img_path = os.path.join(video_path, "{}.png".format(i))
            print(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, img)  # 存入快照
        index = index + 1
