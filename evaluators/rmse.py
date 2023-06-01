import numpy as np
from mmengine import METRICS
from mmengine.evaluator import BaseMetric


@METRICS.register_module()
class RMSE(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        score = score.cpu().numpy()
        gt = gt.cpu().numpy()
        # 将一个批次的中间结果保存至 `self.results`
        self.results.append({
            'score': score,
            'gt': gt,
        })

    def compute_metrics(self, results):
        gt_labels = [i['gt'] for i in self.results]
        temp = []
        for i in gt_labels:
            temp.extend(i)
        gt_labels = temp
        gt_labels = np.array(gt_labels).flatten()
        pr_labels = [i['score'] for i in self.results]
        temp = []
        for i in pr_labels:
            temp.extend(i)
        pr_labels = temp
        pr_labels = np.array(pr_labels).flatten()
        r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())
        # 返回保存有评测指标结果的字典，其中键为指标名称
        return dict(RMSE=r)
