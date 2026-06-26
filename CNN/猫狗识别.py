import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,random_split
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.backends.cudnn.benchmark=True
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms=transforms.Compose([
    transforms.Resize([224,224]),
    #数据增强
    transforms.RandomRotation(degrees=(-10,10)), #随机旋转
    transforms.RandomHorizontalFlip(p=0.5), #随机水平翻转 选择一个概率
    #transforms.RandomVerticalFlip(p=0.5) 随机垂直翻转
    #transforms.RandomPerspective(distortion_scale=0.6,p=1.0), #随机视角
    #transforms.GaussianBlur(kernel_size=(3,3),sigma=(0.1,0.1)), #随机选择的高斯模糊图像
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
val_test_transforms=transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

data_path='data/dogs-vs-cats-redux-kernels-edition/train'
full_dataset=datasets.ImageFolder(root=data_path,transform=None)

train_size=int(len(full_dataset)*0.8)
val_size=len(full_dataset) - train_size
train_dataset,val_dataset=random_split(full_dataset,[train_size,val_size])

train_dataset.dataset.transform=train_transforms
val_dataset.dataset.transform=val_test_transforms

train_loader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,num_workers=4)
val_loader=DataLoader(dataset=val_dataset,batch_size=32,shuffle=False,num_workers=4)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #224
        self.conv1=nn.Conv2d(3,32,kernel_size=3,stride=1)
        self.conv2=nn.Conv2d(32,64,kernel_size=3,stride=1)
        self.conv3=nn.Conv2d(64,128,kernel_size=3,stride=1)
        self.conv4=nn.Conv2d(128,256,kernel_size=3,stride=1)
        self.linear1=nn.Linear(256*12*12,512)
        self.linear2=nn.Linear(512,128)
        self.output=nn.Linear(128,2)
        self.dropout=nn.Dropout(0.5)
        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(kernel_size=2)
    def forward(self,x):
        x=self.pool(self.relu(self.conv1(x)))
        x=self.pool(self.relu(self.conv2(x)))
        x=self.pool(self.relu(self.conv3(x)))
        x=self.pool(self.relu(self.conv4(x)))
        x=x.view(x.size(0),-1)
        x=self.dropout(self.relu(self.linear1(x)))
        x=self.dropout(self.relu(self.linear2(x)))
        output=self.output(x)
        return output

model=Net().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.0005)
scheduler=ReduceLROnPlateau(optimizer,'max',patience=2,factor=0.5)
epoches=10

def validate():
    model.to(device)
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item() * x.size(0)
            predicted = torch.argmax(output, dim=-1)
            correct += torch.sum(predicted == y).item()
            total += y.size(0)
    return total_loss / total, correct / total

def train():
    best_acc = 0
    model.to(device)
    for epoch in range(epoches):
        model.train()
        total_loss,total_correct,total_size,start=0,0,0,time.time()
        for x,y in train_loader:

            x,y=x.to(device),y.to(device)
            y_predict=model(x)
            loss=criterion(y_predict,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
            total_size+=y.size(0)
            total_correct+=torch.sum(torch.argmax(y_predict,dim=-1)==y).item()

        val_loss, val_acc = validate()
        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), './model.pth')

        print(f'epoch:{epoch+1} total_loss:{total_loss:.4f} Acc:{total_correct/total_size:.4f} time:{time.time()-start:.4f}s')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

if __name__ == '__main__':
    train()
