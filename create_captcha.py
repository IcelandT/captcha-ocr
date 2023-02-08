# -*- coding: utf-8 -*-
import string
import random
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt


WIDTH = 170
HEIGHT = 80
numbers = string.ascii_uppercase + string.ascii_lowercase + string.digits
# 次数
FREQUENCY = 2000


def captcha(file_name):
    """ 生成验证码 """
    frame = ImageCaptcha(width=WIDTH, height=HEIGHT)
    random_str = "".join(random.sample(numbers, 4))
    img = frame.generate_image(random_str)
    img.save(f"./captcha/{file_name}/{random_str}.jpg")
    img.close()
    print(f"./captcha/{file_name}/{random_str}.jpg")


if __name__ == '__main__':
    for i in range(2001):
        if i < 1801:
            captcha("train")
        else:
            captcha("test")