from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class MnistDatasetWrapper(Dataset):
    def __init__(self):
        self.dataset = MNIST(root='/data/ly/mnist', download=True)

    def __getitem__(self, item):
        img, target = self.dataset.__getitem__(item)
        print(img)
        print(target)
        views = {}
        score = target
        data = {
            "inputs": views, "num_clips": {},
            # "frame_inds": frame_idxs,
            "gt_label": score,
            "name": 'test'
        }
        return data

    def __len__(self):
        return len(self.dataset)
