import os
from collections import OrderedDict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from data.default_dataset import SingleBranchDataset
from models.video_mae_vqa import VideoMAEVQAWrapper
from sklearn.svm import SVR

if __name__ == '__main__':
    os.chdir('../')
    # video_loader = dict(
    #     name='FragmentLoader',
    #     prefix='4frame',
    #     argument=[
    #         dict(
    #             name='FragmentShuffler',
    #             fragment_size=32,
    #             frame_cube=4
    #         ),
    #         dict(
    #             name='PostProcessSampler',
    #             num=4,
    #             frame_cube=4
    #         )
    #     ]
    # )
    # predict_list = []
    # gt_list = []
    # train_dataset = SingleBranchDataset(video_loader=video_loader, norm=True, clip=4)
    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    # test_dataset = SingleBranchDataset(video_loader=video_loader, norm=True, clip=4, phase='test')
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    #
    # weight = torch.load(
    #     "/data/ly/code/LinVQATools/work_dir/video_mae_vqa/11240027 vit random_cell_mask_75 mae last1/best_SROCC_epoch_358.pth",
    #     map_location='cpu')["state_dict"]
    # t_state_dict = OrderedDict()
    # for key in weight.keys():
    #     if key == "steps":
    #         continue
    #     weight_value = weight[key]
    #     # key = key[7:]
    #     t_state_dict[key] = weight_value
    #
    # model = VideoMAEVQAWrapper(model_type="s", mask_ratio=0.75)
    # info = model.load_state_dict(t_state_dict)
    # print(info)
    # for i in train_dataloader:
    #     y = model(i['inputs'][0].unsqueeze(0), i['gt_label'][0], mode='predict')
    #     predict_list.append(y[0].item())
    #     gt_list.append(y[1].item())
    # for i in train_dataloader:
    #     y = model(i['inputs'][0].unsqueeze(0), i['gt_label'][0], mode='predict')
    #     predict_list.append(y[0].item())
    #     gt_list.append(y[1].item())
    # filename = open("./visualize/2.txt",'w')
    # for i,_ in enumerate(predict_list):
    #     filename.write(str(predict_list[i])+" "+str(gt_list[i])+"\n")
    # filename.close()

    # predict_list = []
    # gt_list = []
    # with open("/data/ly/code/LinVQATools/visualize/1.txt",'r') as f:
    #     lines = f.readlines()
    # for line in lines:
    #     line = line.strip().split(" ")
    #     predict_list.append(float(line[0]))
    #     gt_list.append(float(line[1]))
    # predict_list = np.array(predict_list).reshape((-1,1))
    # gt_list = np.array(gt_list).reshape((-1,1))
    # plt.scatter(predict_list, gt_list)
    #
    #
    # linear_svr = SVR(kernel='poly',degree=2)
    # linear_svr.fit(predict_list, gt_list)
    # predict_list = np.sort(predict_list,axis=0)
    # y = linear_svr.predict(predict_list)
    # plt.plot(predict_list,y,color='red')
    # title_text_obj = plt.title('Ours', fontsize=23, va="bottom")
    # xaxis_label_text_obj = plt.xlabel('Predict Score', fontsize=16,
    #                                   alpha=1.0)
    # yaxis_label_text_obj = plt.ylabel("Subject Score", fontsize=16,
    #                                   alpha=1.0)
    # plt.show()
