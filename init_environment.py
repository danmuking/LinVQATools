import os
import sys

import torchvision
from mmengine import DATASETS

DATASETS.register_module(name='CIFAR10',module=torchvision.datasets.CIFAR10)
project_path = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(project_path,'models')
datasets_path = os.path.join(project_path,'data')
evaluators_path = os.path.join(project_path,'evaluators')
sys.path.append(models_path)
sys.path.append(evaluators_path)
sys.path.append(datasets_path)