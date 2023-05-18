from abc import ABC, abstractmethod


class AbstractReader(ABC):
    @abstractmethod
    def read(self):
        """
        完成数据集声明的加载
        :return:
        """
        pass
