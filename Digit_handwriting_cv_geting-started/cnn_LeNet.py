import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable

train_data = pd.read_csv("D:\\DOWNLOAD\\digit-recognizer\\train.csv")
test_data = pd.read_csv("D:\\DOWNLOAD\\digit-recognizer\\test.csv")

# print(test_data.shape)
# print(train_data.shape)

class imgDataset2(Dataset):
    def __init__(self, datas: pd.DataFrame, transform=None, target_transform=None, loader=None):
        imgs = []
        # labels = []
        # for i in range(data.shape[0]):
        #     tmp = []
        #     if "label" in data.keys():
        #         labels.append(data.iloc[i, 0])
        #         for j in range(28):
        #             tmp.append(np.array(data.iloc[i, j+1:j+29])/255)
        #     else:
        #         labels.append(0)
        #         for j in range(28):
        #             tmp.append(np.array(data.iloc[i, j:j+28])/255)
        #     imgs.append(np.array([tmp]))
        if "label" in datas.keys():
            self.labels = datas.iloc[:, 0]
            for j in range(28):
                imgs.append(np.array(datas.iloc[:, j * 28 + 1:j * 28 + 29]).T.astype(np.float) / 255)
        else:
            self.labels = np.zeros(datas.shape[0])
            for j in range(28):
                imgs.append(np.array(datas.iloc[:, j * 28:j * 28 + 28]).T.astype(np.float) / 255)
        self.imgs = np.array([np.array(imgs).T]).swapaxes(0,1)
        # self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        if self.transform is not None:
            img = self.transform(self.imgs[index]).type(torch.FloatTensor)
        return img, self.labels[index]

    def __len__(self):
        return len(self.labels)


train_data = imgDataset2(train_data, transform=torch.from_numpy)
test_data = imgDataset2(test_data, transform=torch.from_numpy)
train_loader = DataLoader(train_data, batch_size=120, shuffle=True)
test_loader = DataLoader(test_data, batch_size=20)


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, padding=0)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)


        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(1, 6, 3),
        #     nn.BatchNorm2d(6),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(6, 16, 4),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(16 * 5 * 5, 120),
        #     nn.ReLU()
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(120, 90),
        #     nn.ReLU()
        # )
        # self.fc3 = nn.Linear(90, 40)
        # self.fc4 = nn.Linear(40, 10)

    def forward(self, x):
        # print(x.size())
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # print(x.size())
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # print(x.size())
        # 重新塑形，将多维数据重新塑造为二维数据
        # x = x.view(-1, self.num_flat_features(x))
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = self.fc4(x)
        # print(x.size())

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        return x

    # def num_flat_features(self, x):
    #     size = x.size()[1:]
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features


model = LeNet()
use_gpu = torch.cuda.is_available()
# batch_size = 40

if use_gpu:
    model = model.cuda()
    print("USE GPU")
else:
    print("USE CPU")
# 定义代价函数，使用交叉熵验证

criterion = nn.CrossEntropyLoss(size_average=False)
# 直接定义优化器，而不是调用backward
optimizer = torch.optim.Adam(model.parameters())

model.train()
epoch = 4
for i in range(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # 初始化，需清空梯度
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch:{} [{}/{} ({:.0f}%]\tLoss:{:.6f}'.format(
                i, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()
            ))
            print("accuracy:{}".format(np.mean(torch.max(output, 1)[1].cpu().numpy() == target.cpu().numpy())))

outcome = np.array([])
for _, (data, _) in enumerate(test_loader):
    if use_gpu:
        data = data.cuda()
    data = Variable(data)
    output = model(data)
    outcome = np.append(outcome, torch.max(output, 1)[1].cpu().numpy())
# np.savetxt("D:\\DOWNLOAD\\outcomes.csv", outcome, delimiter=',')
outcome = pd.DataFrame(np.array([outcome]).astype(np.int).T, columns=['Label'])
outcome.index += 1
outcome.to_csv("D:\\DOWNLOAD\\outcomes.csv")
