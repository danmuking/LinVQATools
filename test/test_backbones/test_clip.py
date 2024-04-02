from unittest import TestCase

import torch
import torchvision

import models.backbones.clip as clip
from models.model import text_encode, classes, imagenet_templates


class TestClip(TestCase):
    def test_clip(self):
        backbone = "RN50"
        # total_feat_path = os.path.join('cache', 'total_features', backbone)
        # label_path = os.path.join('cache', 'label', backbone)
        # os.makedirs(total_feat_path, exist_ok=True)
        # os.makedirs(label_path, exist_ok=True)

        clip.available_models()
        model, preprocess = clip.load(backbone,device='cpu')
        model.eval()

        # print(
        #     f"Loading {cfg['dataset']} and templates for CALIP: {len(CLASS_NAME[cfg['dataset']])} classes, {len(TEMPLATE[cfg['dataset']])} templates")
        # dataset = torchvision.datasets.ImageNet(cfg['data_root'] + cfg['dataset'], split='val', transform=preprocess)
        # loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=8, shuffle=False)

        print('Encoding text features...')
        feat_t = text_encode(classes, imagenet_templates, model)
        print('Finish encoding text features.')
        print('No cached features and labels, start encoding image features with clip...')
        total_features = []
        labels = []
        with torch.no_grad():
            images = torch.zeros(2,3,224,224)
            label = torch.zeros(2)
            images = images#.cuda()
            label = label#.cuda()
            features = model.encode_image(images)

            features = features.permute(1, 0, 2)
            features /= features.norm(dim=-1, keepdim=True)

            total_features.append(features)
            labels.append(label)
            print(features.shape)