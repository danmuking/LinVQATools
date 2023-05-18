import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import init_environment
from mmengine import Config

from mmengine.runner import Runner

config = Config.fromfile('./config/exp_config.py')
runner = Runner.from_cfg(config)
runner.train()