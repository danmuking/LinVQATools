import os

os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import torch
import decord
import init_environment
from mmengine import Config

from mmengine.runner import Runner

config = Config.fromfile('./config/fast_vqa/faster_vqa.py')
runner = Runner.from_cfg(config)
runner.train()
