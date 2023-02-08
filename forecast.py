# -*- coding: utf-8 -*-
import torch
from one_hot import one_hot, encode
from torch.utils.data import Dataset
import torch.nn as nn
import cv2
import numpy as np
import os


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )       # 64, 76, 166

        self.pool1 = nn.MaxPool2d(kernel_size=2)    # 16, 38, 83

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )       # 128, 34, 79

        self.pool2 = nn.MaxPool2d(kernel_size=2)    # 128, 17, 39

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )       # 256, 13, 35

        self.pool3 = nn.MaxPool2d(2)

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )       # 512, 11, 33

        self.pool4 = nn.MaxPool2d(2)    # 512, 5, 16
        self.layer5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2 * 7 * 512, out_features=4096),
            nn.Dropout(0.2),  # drop 20% of the neuron
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4 * 62)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.layer2(out)
        out = self.pool2(out)
        out = self.layer3(out)
        out = self.pool3(out)
        out = self.layer4(out)
        out = self.pool4(out)
        out = self.layer5(out)
        return out


class CaptchaDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.image_path = file_path
        self.image_names = os.listdir(self.image_path)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = cv2.imread(os.path.join(self.image_path, image_name))
        # 灰度图转换
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 阈值处理
        t, image = cv2.threshold(image, 205, 255, cv2.THRESH_BINARY)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))
        image = image.reshape(1, image.shape[0], image.shape[1])
        image = torch.from_numpy(image).to(dtype=torch.float32)

        captcha = image_name.split(".")[0]
        label = one_hot(captcha)    # 4 * 62
        # 展平
        label = label.view(1, -1)[0].to(dtype=torch.int64)    # 4 * 62 -> 248
        return image, label


def run():
    test_dataset = CaptchaDataset("./captcha/test/")
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=1,
                                                  shuffle=False)

    count = 0
    captcha_model = torch.load("model.pth")
    captcha_model = captcha_model.to("cuda")
    captcha_model.eval()
    with torch.no_grad():
        for i, (image, label) in enumerate(test_dataloader):
            image = image.to("cuda")
            label = label.to("cuda")
            # 还原
            label = label.view(-1, 62)
            # 真实值
            label_text = encode(label)
            # 预测值
            output = captcha_model(image).view(-1, 62)
            output_text = encode(output)
            print(f"真实值: {output_text}, 预测值: {label_text}")
            if output_text == label_text:
                count += 1

    if count:
        print(f"模型正确率: {len(test_dataset) / count * 100}%")
    else:
        print("模型正确率: 0%")


if __name__ == '__main__':
    run()

