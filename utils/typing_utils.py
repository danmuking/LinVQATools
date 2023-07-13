# Copyright (c) OpenMMLab. All rights reserved.
"""Collecting some commonly used type hint in mmaction."""
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData, LabelData

# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
# Type hint of one or more config data
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

InstanceList = List[InstanceData]
OptInstanceList = Optional[InstanceList]

LabelList = List[LabelData]
OptLabelList = Optional[LabelList]



class SamplingResult:
    """Dummy :class:`SamplingResult` in mmdet."""

    def __init__(self, *args, **kwargs):
        pass