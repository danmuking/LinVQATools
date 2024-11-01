import os

import torch
from mmengine.visualization import Visualizer
from mpmath.tests.test_pickle import pickler

from data.default_dataset import SingleBranchDataset
from models.model import ModelWrapper
# from utils.data_preprocess import video

if __name__ == '__main__':
    os.chdir('../')

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
    dataset = SingleBranchDataset(video_loader=video_loader)

    # print(img.shape)

    weight = torch.load(
        "/data/ly/code/LinVQATools/work_dir/model/09091545 model clip resize img baseline 4clip/best_SROCC_epoch_333.pth",
        map_location='cpu')["state_dict"]
    model = ModelWrapper(mask_ratio=0.5).cuda()
    info = model.load_state_dict(weight, strict=False)




    for i in range(30):
        data = dataset[10*i]
        data['inputs'] = {k: v.unsqueeze(0).cuda() for k, v in data['inputs'].items()}
        feat = model(data["inputs"],mode='predict',gt_label=torch.rand((1)).cuda(),gt_class=torch.empty(1, dtype=torch.long).random_(5).cuda())
        feat = feat[2][-1]
        feat = feat.reshape(1,4,14,14,384)
        feat = feat[0][0].permute(2,0,1)

        img = data['raw_video'][0,:,0,...].permute(1,2,0).cpu().detach().numpy()
        # img = ((img*OPENAI_DATASET_STD)+OPENAI_DATASET_MEAN)*255
        img = img.astype('uint8')
        print(feat.shape)
        print(img.shape)
        visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')],
                                save_dir='temp_dir')
        drawn_img = visualizer.draw_featmap(feat,img)

        visualizer.add_image('demo_0_{}'.format(i), drawn_img)
    for i in range(30):
        data = dataset[10*i]
        resnet = model.model.clip_model
        def _forward(x):
            x = resnet.visual.stem(x)
            x = resnet.visual.layer1(x)
            x = resnet.visual.layer2(x)
            x = resnet.visual.layer3(x)
            x = resnet.visual.layer4(x)
            # x = self.attnpool(x)

            return x

        # video = torch.from_numpy(np.load("temp.npy"))
        img = data['inputs']['img'][0]
        img = img.unsqueeze(0)
        img = img.cuda()
        OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
        OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
        vis_model = resnet.visual.forward = _forward
        feat = resnet.encode_image(img)
        feat = feat[0].cpu().detach()
        img = img[0].permute(1,2,0).cpu().detach().numpy()
        img = ((img*OPENAI_DATASET_STD)+OPENAI_DATASET_MEAN)*255
        img = img.astype('uint8')
        print(feat.shape)
        print(img.shape)
        visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')],
                                save_dir='temp_dir')
        drawn_img = visualizer.draw_featmap(feat,img)

        visualizer.add_image('demo_1_{}'.format(i), drawn_img)
    #