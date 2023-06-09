from typing import Optional

import wandb
from mmengine import HOOKS
from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from mmengine.visualization import Visualizer

from global_class.train_recorder import TrainResultRecorder


@HOOKS.register_module()
class TrainEvaluatorHook(Hook):
    """
    训练指标钩子,用于增加记录训练指标的功能
    """

    def __init__(self):
        self.visualizer = None
        self.evaluator = None
        self.recoder = TrainResultRecorder.get_instance('mmengine')

    def before_run(self, runner):
        """
        初始化evaluator
        :param runner:
        :return:
        """
        if self.evaluator is None:
            evaluator = [
                dict(type='SROCC'),
                dict(type='KRCC'),
                dict(type='PLCC'),
                dict(type='RMSE'),
            ]
            self.evaluator = runner.build_evaluator(evaluator)
        self.visualizer = Visualizer.get_current_instance()
        # print(self.wandb)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        outputs = (self.recoder.iter_y_pre, self.recoder.iter_y)
        self.recoder.y_pre.append(self.recoder.iter_y_pre)
        self.recoder.y.append(self.recoder.iter_y)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)

    def after_train_epoch(self, runner) -> None:
        metrics = self.evaluator.evaluate(len(runner.train_dataloader.dataset))
        metrics = {'train_SROCC': metrics['SROCC'], 'train_KRCC': metrics['KRCC'],
                   'train_PLCC': metrics['PLCC'], 'train_RMSE': metrics['RMSE']}
        self.visualizer.add_scalars(metrics)

