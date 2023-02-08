# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import forecast
from torch.utils.data import Dataset
import torch.nn as nn
import cv2
import os
from one_hot import one_hot


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


# 加载训练集
train_dataset = CaptchaDataset("./captcha/train/")

BATCH_SIZE = 64    # 每一批数据大小

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)


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


cnn = CNN()
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
cnn = cnn.to(DEVICE)

criterion = nn.MultiLabelSoftMarginLoss().to(DEVICE)

# 学习率
LEARNING_RATE = 0.01
# 优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

# 训练轮数
EPOCHS = 50

cnn.train()
losses = list()
for epoch in range(EPOCHS):
    for i, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        # 清零
        optimizer.zero_grad()
        outputs = cnn(image)

        # 计算损失函数
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        losses.append(loss.data.cpu().item())
        if (i + 1) % 10 == 0:
            print("轮数: {}/{}, 当前: {}/{}, 损失: {:.4f}".format(epoch + 1, EPOCHS, i+1, len(train_loader), loss))


# 保存模型
torch.save(cnn, "model.pth")

plt.xkcd()
plt.xlabel("Epoch ")
plt.ylabel("Loss")
plt.plot(losses)
plt.show()


# 验证
forecast.run()


