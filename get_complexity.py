import torch
from mmengine.analysis import get_model_complexity_info

from models.video_mae_vqa import VideoMAEVQAWrapper
from thop import profile, clever_format

# input_shape = (1, 3, 16, 224, 224)
# model = VideoMAEVQAWrapper(model_type="s",mask_ratio=0.75,).eval()
# analysis_results = get_model_complexity_info(model, input_shape)
# print("Model Flops:{}".format(analysis_results['flops_str']))
# # Model Flops:10.684G
# print("Model Parameters:{}".format(analysis_results['params_str']))
# # Model Parameters:25.659M

model = VideoMAEVQAWrapper(model_type="s",mask_ratio=0.75,).eval()
input = torch.randn(1,1, 3, 16, 224, 224)
macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print("Model Flops:{}".format(macs))
print("Model Parameters:{}".format(params))