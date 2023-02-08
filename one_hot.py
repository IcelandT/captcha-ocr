# -*- coding: utf-8 -*-
import string
import numpy as np
import torch


numbers = string.ascii_uppercase + string.ascii_lowercase + string.digits


def one_hot(captcha_numbers):
    """ 独热编码 """
    char_and_int = dict((val, index) for index, val in enumerate(numbers))
    one_hot_encode = list()
    for word in captcha_numbers:
        one = [0 for _ in range(len(numbers))]
        one[char_and_int[word]] = 1
        one_hot_encode.append(one)
    return torch.Tensor(one_hot_encode).view(1, -1)[0]


def encode(onehot):
    """ 解码 """
    # 获取最大值的下标
    index = torch.argmax(onehot, dim=1)
    label_text = ""
    for idx in index:
        label_text += numbers[idx]
    return label_text

