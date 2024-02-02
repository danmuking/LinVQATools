import os

import numpy as np
from sklearn.metrics import mean_squared_error

os.chdir('/data/ly/code/LinVQATools')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.default_dataset import SingleBranchDataset
from models.video_mae_vqa import VideoMAEVQAWrapper

model = VideoMAEVQAWrapper(
    model_type='s',
    mask_ratio=0.75,
    head_dropout=0.1,
    drop_path_rate=0.1
)
weight_path = '/data/ly/code/LinVQATools/work_dir/video_mae_vqa/01171449 vit random_cell_mask_75 mae last 4clip/best_SROCC_epoch_555.pth'
weight = torch.load(weight_path,map_location="cpu")
info = model.load_state_dict(weight['state_dict'])
print(info)

num_workers = 6
prefix = '4frame'
argument = [
        dict(
            name='FragmentShuffler',
            fragment_size=32,
            frame_cube=4
        ),
        dict(
            name='PostProcessSampler',
            frame_cube=4,
            num=4
        )
]
val_video_loader = dict(
    name='FragmentLoader',
    prefix=prefix,
    frame_sampler=None,
    spatial_sampler=None,
    argument=argument,
    phase='test',
    use_preprocess=True,
)
dataset=SingleBranchDataset(
    video_loader=val_video_loader,
    anno_root='./data/odv_vqa',
    anno_reader='ODVVQAReader',
    split_file='./data/odv_vqa/tr_te_VQA_ODV.txt',
    phase='test',
    norm=True,
    clip=4
)
val_dataloader = DataLoader(batch_size=1,
                            shuffle=False,
                            dataset=dataset)
model = model.cuda().eval()

max_srocc = 0
max_plcc = 0
max_krocc=0
max_rmse=1000
for i in range(10000):
    gt = []
    pr = []
    with torch.no_grad():
        for item in tqdm(val_dataloader):
            gt.append(item['gt_label'])
            y = model(inputs=item["inputs"].cuda(), gt_label=item['gt_label'].cuda(),mode='predict')
            pr.append(y[0])
        # print(i)

    from scipy.stats import spearmanr, pearsonr, kendalltau
    import copy

    all_gt = copy.deepcopy(gt)
    all_pr = copy.deepcopy(pr)
    all_gt = torch.tensor(all_gt)
    all_pr = torch.tensor(all_pr)


    srocc = spearmanr(all_gt, all_pr)[0]
    plcc, _ = pearsonr(all_gt, all_pr)
    krocc, _ = kendalltau(all_gt, all_pr)
    MSE = mean_squared_error(all_gt, all_pr)
    RMSE = np.sqrt(MSE) * 100

    max_srocc = max(srocc, max_srocc)
    max_krocc = max(krocc, max_krocc)
    max_plcc = max(plcc, max_plcc)
    max_rmse = min(RMSE, max_rmse)
    print('PLCC: {}, SROCC: {}, KROCC: {}, RMSE: {}'.format(max_plcc, max_srocc, max_krocc, max_rmse))
    # srocc_list = []
    # for i in range(12):
    #     part_pr = all_pr[i*9:(i+1)*9]
    #     part_gt = all_gt[i*9:(i+1)*9]
    #     srocc = spearmanr(part_gt, part_pr)[0]
    #     srocc_list.append(srocc)
    # print(srocc_list)