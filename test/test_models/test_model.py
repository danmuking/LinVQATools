from unittest import TestCase

import torch
from torch import nn

from models.model import Model, ModelWrapper
from models.video_mae_vqa import CellRunningMaskAgent


class TestCellRunningMaskAgent(TestCase):
    def test_Model(self):
        agent = CellRunningMaskAgent(0.75)
        model = Model(mask_ratio=0.75)
        inputs = {"video":torch.rand((2, 3, 16, 224, 224)),"img":torch.rand((2,3,224,224))}
        agent.train()
        mask = agent(inputs, [8, 14, 14])['mask']
        mask = mask.reshape(mask.size(0), 8, -1)
        model(inputs,mask)

    def test_model_wrapper(self):
        model = ModelWrapper(mask_ratio=0.75)
        inputs = {"video":torch.rand((3,1, 3, 16, 224, 224)),"img":torch.rand((3,1,3,224,224))}
        y = model(inputs,gt_label=torch.rand((3)),gt_class=torch.empty(3, dtype=torch.long).random_(5),mode='loss')
        print(y)
    def test_clip(self):
        import torch
        from PIL import Image
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')

        image = preprocess(Image.open("/data/ly/resize/0/VQA_ODV/Group1/G1AbandonedKingdom_ERP_7680x3840_fps30_qp27_45406k/16.png")).unsqueeze(0)
        text = tokenizer(["a diagram", "a dog", "a cat"])

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        print("image features shape:", image_features.shape)
        print("text features shape:", text_features.shape)
        print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
    def test_celoss(self):
        loss = nn.CrossEntropyLoss()
        input = torch.randn(3, 5, requires_grad=True)
        target = torch.empty(3, dtype=torch.long).random_(5)
        print(input.shape)
        print(target.shape)
        print(target)
        # output = loss(input, target)
        # output.backward()