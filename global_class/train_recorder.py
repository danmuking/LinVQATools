from mmengine import ManagerMixin


class TrainResultRecorder(ManagerMixin):
    """
    用于记录训练过程的结果
    """

    def __init__(self, name):
        super().__init__(name)
        # 每轮迭代的结果
        self.iter_y_pre = None
        self.iter_y = None
        # 每个epoch的结果
        self.y_pre = []
        self.y = []
