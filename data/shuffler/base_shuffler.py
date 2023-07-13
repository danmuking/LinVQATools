class BaseShuffler(object):
    """
    基类,不对视频做任何处理
    """
    def __init__(self,**kargs):
        pass
    def shuffle(self,video):
        return video