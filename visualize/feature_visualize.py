import os

import torch
from mmengine.visualization import Visualizer

from data.default_dataset import SingleBranchDataset
from models.model import ModelWrapper


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
    data = dataset[10]
    # video = torch.from_numpy(np.load("temp.npy"))
    img = data['inputs']['img'][0]
    img = img.unsqueeze(0)
    # print(img.shape)

    weight = torch.load(
        "/data/ly/code/LinVQATools/work_dir/model/09091545 model clip resize img baseline 4clip/best_SROCC_epoch_333.pth",
        map_location='cpu')["state_dict"]
    model = ModelWrapper(mask_ratio=0.5).cuda()
    info = model.load_state_dict(weight, strict=False)
    img = img.cuda()
    resnet = model.model.clip_model
    def _forward(x):
        x = resnet.visual.stem(x)
        x = resnet.visual.layer1(x)
        x = resnet.visual.layer2(x)
        x = resnet.visual.layer3(x)
        x = resnet.visual.layer4(x)
        # x = self.attnpool(x)

        return x
    vis_model = resnet.visual.forward = _forward
    feat = resnet.encode_image(img)
    feat = feat[0]
    visualizer = Visualizer()
    drawn_img = visualizer.draw_featmap(feat,img, channel_reduction='select_max')
    visualizer.show(drawn_img)