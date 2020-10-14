from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os


class Img(object):
    def __init__(self, img_list=None, dir=r"D:\DOWNLOAD\dogs-vs-cats\train\train"):
        self.dir = dir
        self.img_list = img_list
        self.transformer = transforms.Compose([
            transforms.Resize((600, 600), interpolation=0),
            transforms.ToTensor()
        ])

    def readList(self):
        self.img_list = os.listdir(self.dir)

    def readImg(self):
        img_tensor = []
        target = []
        for i in self.img_list:
            img = Image.open(self.dir+"\\"+i).convert('RGB')
            img_tensor.append(img)
            if 'cat' in i:
                target.append(1)
            else:
                target.append(0)
        return img_tensor, target


class ImgDateset(Dataset):
    def __init__(self, path=r"D:\DOWNLOAD\dogs-vs-cats\train\train", transformer=None):
        self.dir = path
        self.imgs = os.listdir(path)
        if transformer is None:
            self.transformer = transforms.Compose([
                transforms.Resize((500, 500), interpolation=0),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transformer = transformer

    def __getitem__(self, index):
        fn = self.imgs[index]
        if "cat" in fn:
            label = 1
        else:
            label = 0
        img = Image.open(self.dir+"\\"+fn).convert('RGB')
        if self.transformer is not None:
            img = self.transformer(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    # train_dataset = torchvision.datasets.ImageFolder(root='/kaggle/input/dogs-vs-cats/train',transform=transformer)
    train_set = ImgDateset()
    test_set = ImgDateset(r"D:\DOWNLOAD\dogs-vs-cats\test1\test1")
    train_loader = DataLoader(dataset=train_set, batch_size=100, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=20)





