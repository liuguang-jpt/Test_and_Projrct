import torch
import torch.nn as nn

class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4,**kwargs):
        super(Inception,self).__init__(**kwargs)
        #路径1 1x1卷积层
        self.p1_1=nn.Conv2d(in_channels,c1,kernel_size=1,stride=1,padding=0)
        #路径2 1x1卷积层后连接3x3卷积层
        self.p2_1=nn.Conv2d(in_channels,c2[0],kernel_size=1,stride=1,padding=0)
        self.p2_2=nn.Conv2d(in_channels,c2[1],kernel_size=3,stride=1,padding=1)
        #路径3 1x1卷积层后连接5x5卷积层
        self.p3_1=nn.Conv2d(in_channels,c3[0],kernel_size=1,stride=1,padding=0)
        self.p3_2=nn.Conv2d(in_channels,c3[1],kernel_size=5,stride=1,padding=2)
        #路径4 3x3最大池化层后连接1x1卷积层
        self.p4_1=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.p4_2=nn.Conv2d(in_channels,c4,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        p1=nn.ReLU(self.p1_1(x))
        p2=nn.ReLU(self.p2_2(nn.ReLU(self.p2_1(x))))
        p3=nn.ReLU(self.p3_2(nn.ReLU(self.p3_1(x))))
        p4=nn.ReLU(self.p4_2(self.p4_1(x)))
        return torch.cat((p1,p2,p3,p4),dim=1)

b1=nn.Sequential(nn.Conv2d(1,64,7,2,3),
                 nn.ReLU(),nn.MaxPool2d(3,2,1))
b2=nn.Sequential(nn.Conv2d(64,64,1,1,1),nn.ReLU(),
                 nn.Conv2d(64,192,3,1,1),nn.ReLU(),
                 nn.MaxPool2d(3,2,1))
b3=nn.Sequential(Inception(192,64,(96,128),(16,32),32),
                 Inception(256,128,(128,192),(32,96),64),
                 nn.MaxPool2d(3,2,1))
b4=nn.Sequential(Inception(480,192,(96,208),(16,48),64),
                 Inception(512,160,(112,224),(24,64),64),
                 Inception(512,128,(128,256),(24,64),64),
                 Inception(512,112,(144,288),(32,64),64),
                 Inception(528,256,(160,320),(32,128),128),
                 nn.MaxPool2d(3,2,1))
b5=nn.Sequential(Inception(832,256,(160,320),(32,128),128),
                 Inception(832,384,(192,384),(48,128),128),
                 nn.AdaptiveAvgPool2d((1,1)),nn.Flatten())
net=nn.Sequential(b1,b2,b3,b4,b5,nn.Linear(1024,10))
