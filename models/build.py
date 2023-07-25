from typing import Dict

from models.faster_vqa import FasterVQA


def build(config:Dict):
    if config['model_type']=='faster_vqa':
        model = FasterVQA(

        )