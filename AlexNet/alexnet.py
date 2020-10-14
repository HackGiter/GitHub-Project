import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PD.pre_data import ImgDateset
from torch.autograd import Variable


# some experiments like vgg have shown that local response normalisation is useless but cost a lot of computation and increase memories
class LRN(nn.Module):
    def __init__(self, N, k=2, a=10E-4, b=0.75):
        super(LRN, self).__init__()
        self.avgpool = nn.AvgPool3d(kernel_size=(N, 1, 1),
                                    stride=1,
                                    padding=(int((N-1.0)/2), 0, 0))
        self.k = k
        self.alfa = a
        self.beta = b

    def forward(self, x):
        div = self.avgpool(x.pow(2).unsqueeze(1)).squeeze(1)
        div = div.mul(self.alfa).add(self.k).mul(self.beta)
        x = x.div(div)
        return x



class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 48, 11, stride=4, padding=2), #without padding, output 124*124*48
            nn.ReLU(inplace=True),
            LRN(5),
            nn.MaxPool2d(3, stride=2, padding=1) #62*62
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(48, 128, 5, stride=2, padding=1), #30*30*128
            nn.ReLU(inplace=True),
            LRN(5),
            nn.MaxPool2d(3, stride=2, padding=1) #15*15
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 192, 3, stride=1), #13*13*192
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(192, 192, 3, stride=1), #11*11*192
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 128, 3, stride=1), #9*9*128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2), #4*4

        )
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 16),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(16, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc1(x.view(-1, 2048))
        x = self.fc2(x)
        x = self.fc3(x)
        return x


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = AlexNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
epoches = 10

train_set = ImgDateset()
test_set = ImgDateset(r"D:\DOWNLOAD\dogs-vs-cats\test1\test1")
train_loader = DataLoader(dataset=train_set, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=50)

if __name__ == "__main__":
    for per_epoch in range(epoches):
        for batch_idx, (data, target) in enumerate(train_loader):
            if DEVICE is "cuda":
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 40 == 0:
                print("epoch:{} || batch_idx:{} || loss:{} || acc:{}".format(
                    per_epoch,
                    batch_idx,
                    loss.item(),
                    (torch.max(output, 1)[1].cpu().numpy() == target.cpu().numpy()).mean()
                ))


