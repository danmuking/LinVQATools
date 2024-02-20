# model = dict(
#     type='VideoMAEVQAWrapper',
#     model_type='s',
#     mask_ratio=0.0,
#     head_dropout=0.1,
#     drop_path_rate=0.1
# )
import os
import time

import torch
import torch.nn.functional as F
from einops import rearrange
from matplotlib import pyplot as plt
from tqdm import tqdm

from data.default_dataset import SingleBranchDataset
from models.video_mae_vqa import VideoMAEVQAWrapper
os.chdir('../')
model = VideoMAEVQAWrapper(
    model_type='s',
    mask_ratio=0.0,
    head_dropout=0.1,
    drop_path_rate=0.1
)
path = "/data/ly/code/LinVQATools/work_dir/video_mae_vqa/02021555 vit mask_0 mae 4clip/best_SROCC_epoch_135.pth"
weight = torch.load(path, map_location="cpu")
info = model.load_state_dict(weight['state_dict'])
video_loader = dict(
            name='FragmentLoader',
            prefix='4frame',
            argument=[
                # dict(
                #     name='FragmentShuffler',
                #     fragment_size=32,
                #     frame_cube=4
                # ),
                # # dict(
                #     name='SpatialShuffler',
                #     fragment_size=32,
                # ),
                dict(
                    name='PostProcessSampler',
                    num=4,
                    frame_cube=4
                )
            ]
        )
dataset = SingleBranchDataset(video_loader=video_loader, norm=False)
data = dataset[0]
# video = torch.from_numpy(np.load("temp.npy"))
inputs = data['inputs'][0].unsqueeze(0).unsqueeze(0)
output = model(inputs=inputs, gt_label=torch.rand((2)),mode='tensor')
feat = output['feats']
feat = feat[-1][0]
feat = rearrange(feat, '(c1 t c2 w c3 h) c -> (c1 c2 c3) (t w h) c', c1=4, c2=7, c3=7, t=2, w=2, h=2)

for i in tqdm(range(feat.shape[0])):
    similar = []
    for j in tqdm(range(feat.shape[1])):
        for k in range(feat.shape[1]):
            similar.append(F.cosine_similarity(feat[i][j].unsqueeze(0), feat[i][k].unsqueeze(0)))
    similar = torch.tensor(similar).reshape(8, 8)
    plt.axis('off')  # 取消坐标轴
    plt.imshow(similar)
    plt.savefig("{}.png".format(i),bbox_inches='tight', pad_inches = -0.1)
    # plt.show()

