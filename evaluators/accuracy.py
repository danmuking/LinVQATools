from mmengine import METRICS
from mmengine.evaluator import BaseMetric

@METRICS.register_module()
class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt_c,gt_r = data_samples
        y_c = score[0]
        y_r = score[1]
        # 将一个批次的中间结果保存至 `self.results`
        self.results.append({
            'batch_size': len(y_c),
            'correct_c': (y_c.argmax(dim=1) == gt_c).sum().cpu(),
            'correct_r': (y_r.argmax(dim=1) == gt_r).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct_c = sum(item['correct_c'] for item in results)
        total_correct_r = sum(item['correct_r'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        # 返回保存有评测指标结果的字典，其中键为指标名称
        return dict(accuracy_c=100 * total_correct_c / total_size,
                    accuracy_r=100 * total_correct_r / total_size)