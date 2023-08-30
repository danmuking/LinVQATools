import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["WANDB_API_KEY"] = '2125be99aae223011562a86318a7e1edd8556487'
# os.environ["WANDB_MODE"] = "offline"
import torch
import decord
import init_environment
from mmengine import Config

from mmengine.runner import Runner

config = Config.fromfile('./config/fast_vqa/mvit.py')
runner = Runner.from_cfg(config)
runner.train()
while True:
    pass

