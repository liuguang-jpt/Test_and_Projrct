import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_dataset():
    train_dataset=CIFAR10(root='./data',train=True,transform=ToTensor())
    test_dataset = CIFAR10(root='./data',train=False,transform=ToTensor())
    return train_dataset,test_dataset

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(3,6,5,1)
        self.pool1=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5,1)
        self.pool2=nn.MaxPool2d(2,2)

        self.linear1=nn.Linear(16*5*5,120)
        self.linear2=nn.Linear(120,84)
        self.output=nn.Linear(84,10)
    def forward(self,x):
        x=self.pool1(torch.relu(self.conv1(x)))
        x=self.pool2(torch.relu(self.conv2(x)))
        x=x.view(-1,16*5*5)
        x=torch.relu(self.linear1(x))
        x=torch.relu(self.linear2(x))
        x=self.output(x)
        return x
def train(train_dataset):
    train_loader = DataLoader(dataset=train_dataset,batch_size=8,shuffle=True)
    model = CNN()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    epoches=10
    model.train()
    for epoch in range(epoches):
        total_loss,total_correct,total_samples,start=0,0,0,time.time()
        for x,y in train_loader:
            x=x.to(device)
            y=y.to(device)
            y_pred=model(x)
            loss=criterion(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()*len(y)
            total_samples+=len(y)
            total_correct+=(torch.argmax(y_pred,dim=-1)==y).sum().item()
        print(f'epoch:{epoch+1} total_loss:{total_loss} Acc:{total_correct/total_samples} Time:{time.time()-start}')
    torch.save(model.state_dict(), '../model.pth')

def evaluate(test_dataset):
    test_loader=DataLoader(dataset=test_dataset,batch_size=8,shuffle=False)
    model=CNN()
    model.load_state_dict(torch.load('../model.pth'))
    model.to(device)
    total_correct,total_samples,=0,0
    for x,y in test_loader:
        model.eval()
        x=x.to(device)
        y=y.to(device)
        y_pred=model(x)
        total_correct+=(torch.argmax(y_pred,dim=-1)==y).sum().item()
        total_samples+=len(y)
    print(f'Acc:{total_correct/total_samples}')


if __name__ == '__main__':
    train_dataset,test_dataset=create_dataset()
    #train(train_dataset)
    evaluate(test_dataset)