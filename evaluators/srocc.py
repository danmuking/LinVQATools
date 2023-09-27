import numpy as np
import torch
from mmengine import METRICS
from mmengine.evaluator import BaseMetric
from scipy.stats import spearmanr


@METRICS.register_module()
class SROCC(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt, name = data_samples
        if torch.is_tensor(score) and torch.is_tensor(gt):
            score = score.detach().cpu().numpy()
            gt = gt.detach().cpu().numpy()
        # 将一个批次的中间结果保存至 `self.results`
        for i, _ in enumerate(name):
            self.results.append({
                'score': score[i],
                'gt': gt[i],
                'name': name[i],
            })

    def compute_metrics(self, results):
        result_dict = dict()
        for item in self.results:
            if item['name'] in result_dict.keys():
                result_dict[item['name']]['score'].append(item['score'])
                result_dict[item['name']]['gt'].append(item['gt'])
            else:
                result_dict[item['name']] = {
                    'score': [item['score']],
                    'gt': [item['gt']],
                }
        for key in result_dict.keys():
            result_dict[key]['score'] = np.array(result_dict[key]['score']).mean()
            result_dict[key]['gt'] = np.array(result_dict[key]['gt']).mean()

        gt_labels = [result_dict[key]['gt'] for key in result_dict.keys()]
        gt_labels = np.array(gt_labels).flatten()
        pr_labels = [result_dict[key]['score'] for key in result_dict.keys()]
        pr_labels = np.array(pr_labels).flatten()
        s = spearmanr(gt_labels, pr_labels)[0]
        # 返回保存有评测指标结果的字典，其中键为指标名称
        return dict(SROCC=s)
