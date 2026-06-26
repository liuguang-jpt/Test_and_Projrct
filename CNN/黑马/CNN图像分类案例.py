"""
案例：
    演示CNN的图像分类
步骤：
    1、准备数据集
    2、搭建神经网络
    3、训练模型
    4、测试模型

    卷积层参数计算公式：
        输入通道数*卷积核尺寸*卷积核数量+卷积核数量
"""
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import time

BATCH_SIZE=32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dataset():
    train_transform=transforms.Compose([
        transforms.RandomCrop(32,padding=4), #随机裁剪
        transforms.RandomHorizontalFlip(), #随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    #参1：数据集路径 参2：是否是训练集 参3：数据集预处理->张量数据 参4：是否联网下载
    train_dataset=CIFAR10(root='data',train=True,transform=train_transform)
    test_dataset = CIFAR10(root='data',train=False,transform=test_transform)

    return train_dataset,test_dataset

class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel,self).__init__()
        #输入 3*32*32
        self.conv1 = nn.Conv2d(3, 64,3,1,1) #64*16*16
        self.pool1= nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(64, 128, 3,1,1) #128*8*8
        self.pool2= nn.MaxPool2d(2,2)
        self.conv3=nn.Conv2d(128, 256, 3,1,1) #256*4*4
        self.pool3= nn.MaxPool2d(2,2)

        self.linear1 = nn.Linear(256*4*4, 1024)
        self.dropout1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(p=0.5)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
            #卷积层 + 激活函数（激励层）+ 池化层
            x=self.pool1(torch.relu(self.conv1(x)))
            x=self.pool2(torch.relu(self.conv2(x)))
            x=self.pool3(torch.relu(self.conv3(x)))
            #全连接层只能处理二维数据（8，16，6，6）
            x=x.reshape(x.size(0),-1)
            x=self.dropout1(torch.relu(self.linear1(x)))
            x=self.dropout2(torch.relu(self.linear2(x)))
            x=self.dropout3(torch.relu(self.linear3(x)))
            return self.output(x)

def train_model(train_dataset):
    #创建数据加载器
    dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    #创建模型对象
    model = ImageModel()
    model=model.to(device)

    #创建损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    #循环遍历epoch，开始每轮的训练工作
    epochs=100
    for epoch in range(epochs):
        total_loss,total_correct,total_samples,start=0,0,0,time.time()
        for x,y in dataloader:
            x=x.to(device)
            y=y.to(device)

            model.train()
            y_pred=model(x)
            loss=criterion(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_correct += (torch.argmax(y_pred,dim=-1)==y).sum().item()
            total_loss+=loss.item()*len(y) #第一批平均损失*第一批样本个数
            total_samples+=len(y)
        print(f'epoch:{epoch+1},loss:{total_loss},acc:{total_correct/total_samples},time:{time.time()-start}')

        #保存模型
        torch.save(model.state_dict(), '../model.pth')

def evaluate(test_dataset):
    dataloader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)

    model = ImageModel()
    model.load_state_dict(torch.load('../model.pth', map_location=device))
    model=model.to(device)

    total_correct,total_samples,start=0,0,time.time()
    for x,y in dataloader:
        model.eval()
        x=x.to(device)
        y=y.to(device)
        total_correct += (torch.argmax(model(x),dim=-1)==y).sum().item()
        total_samples+=len(y)

    print(f'Acc:{total_correct/total_samples},time:{time.time()-start}')

if __name__=='__main__':
    train_dataset,test_dataset=create_dataset()
    train_model(train_dataset)
    evaluate(test_dataset)
