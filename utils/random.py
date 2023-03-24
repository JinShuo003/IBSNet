import numpy as np
import math


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


def randPointsUniform(num: int, radius: float):
    u = np.random.uniform(size=(num, 1))
    v = np.random.uniform(size=(num, 1))
    theta = u * 2.0 * math.pi
    phi = np.arccos(2.0 * v - 1.0)
    r = np.cbrt(np.random.uniform(size=(num, 1))) * radius
    sinTheta = np.sin(theta)
    cosTheta = np.cos(theta)
    sinPhi = np.sin(phi)
    cosPhi = np.cos(phi)
    x = r * sinPhi * cosTheta
    y = r * sinPhi * sinTheta
    z = r * cosPhi
    return np.concatenate([x, y, z], axis=1)
