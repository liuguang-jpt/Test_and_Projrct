import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,random_split
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.backends.cudnn.benchmark=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(degrees=(-10,10)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

data_path='data/dogs-vs-cats-redux-kernels-edition/train'
full_dataset=datasets.ImageFolder(data_path,transform=None)

train_size=int(len(full_dataset)*0.8)
val_size=len(full_dataset) - train_size
train_dataset,val_dataset=random_split(full_dataset,[train_size,val_size])

train_dataset.dataset.transform=train_transforms
val_dataset.dataset.transform=val_transforms

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=4)
val_loader=DataLoader(val_dataset,batch_size=32,shuffle=False,num_workers=4)

#ResNet-50
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(Bottleneck, self).__init__()
        #1x1卷积降维
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        #3x3卷积
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        #1x1卷积
        self.conv3=nn.Conv2d(out_channels,out_channels * self.expansion ,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(out_channels * self.expansion)
        self.relu=nn.ReLU()
        self.downsample=downsample

    def forward(self,x):
        identity=x
        if self.downsample is not None:
            identity=self.downsample(x)
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.relu(self.bn2(self.conv2(out)))
        out=self.bn3(self.conv3(out))
        out+=identity
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=2):
        super(ResNet,self).__init__()
        self.in_channels=64

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.pool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #4个残差层
        self.layer1=self.make_layer(block,64,layers[0],stride=1)
        self.layer2=self.make_layer(block,128,layers[1],stride=2)
        self.layer3=self.make_layer(block,256,layers[2],stride=2)
        self.layer4=self.make_layer(block,512,layers[3],stride=2)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion,num_classes) #512*4=2048

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def make_layer(self,block,out_channels,blocks,stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,out_channels*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels*block.expansion),
            )
        layers=[]
        layers.append(block(self.in_channels,out_channels,stride,downsample))
        self.in_channels=out_channels*block.expansion
        for i in range(1,blocks):
            layers.append(block(self.in_channels,out_channels))
        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.pool(self.relu(self.bn1(self.conv1(x))))
        x=self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x=self.fc(torch.flatten(self.avgpool(x),1))
        return x

model=ResNet(Bottleneck,[3,4,6,3],2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0005)
scheduler=ReduceLROnPlateau(optimizer,'min',patience=2,factor=0.5)
epochs=10

def validate():
    model.eval()
    total_loss,correct,total=0,0,0
    with torch.no_grad():
        for x,y in val_loader:
            x,y=x.to(device),y.to(device)
            output=model(x)
            loss=criterion(output,y)
            total_loss+=loss.item()*x.size(0)
            predicted=torch.argmax(output,dim=-1)
            correct+=torch.sum(predicted==y).item()
            total+=y.size(0)
        return total_loss/total,correct/total

def train():
    best_acc=0
    for epoch in range(epochs):
        model.train()
        total_loss,total_correct,total_size,start=0,0,0,time.time()
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            output=model(x)
            loss=criterion(output,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
            total_size+=x.size(0)
            total_correct+=torch.sum(torch.argmax(output,-1)==y).item()

        val_loss,val_acc=validate()
        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc=val_acc
            torch.save(model.state_dict(),'./model.pth')

        print(f'epoch:{epoch+1} total_loss:{total_loss:.4f} Acc:{total_correct/total_size:.4f} time:{time.time()-start:.4f}')
        print(f'Val_loss:{val_loss:.4f} Acc:{val_acc:.4f}')

if __name__ == '__main__':
    train()

