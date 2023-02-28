import numpy as np


def randNormalFloat(l: float, h: float, num: int):
    """生成num个给定范围内的浮点数，符合正态分布"""
    if l > h:
        return None
    else:
        return np.random.normal((h + l) / 2, (h - l) / 6, num)


def randUniFormFloat(l: float, h: float, num: int):
    """生成num个给定范围内的浮点数，符合均匀分布"""
    if l > h:
        return None
    else:
        return np.array([np.random.random() * (h - l) + l for i in range(num)])
