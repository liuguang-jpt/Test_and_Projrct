import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms_train=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transforms_test=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset=datasets.CIFAR10(root='./data',train=True,transform=transforms_train)
test_dataset=datasets.CIFAR10(root='./data',train=False,transform=transforms_test)
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True,num_workers=2)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False,num_workers=2)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1)
        self.conv2=nn.Conv2d(64,64,kernel_size=3,stride=1)
        self.conv3=nn.Conv2d(64,128,kernel_size=3,stride=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.linear1=nn.Linear(128*2*2,512)
        self.linear2=nn.Linear(512,10)
        self.dropout=nn.Dropout(0.5)
    def forward(self,x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(-1,128*2*2)
        x=self.dropout(torch.relu(self.linear1(x)))
        output=self.linear2(x)
        return output

model=Net().to(device)
Epochs=20
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)

def train(train_loader,model,epochs,criterion,optimizer,device):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        total_loss,total_correct,total_samples,start_time=0,0,0,time.time()
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            y_predict=model(x)

            loss=criterion(y_predict,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
            total_samples+=y.size(0)
            total_correct+=torch.sum(torch.argmax(y_predict,dim=-1)==y).item()
        print(f'total_loss:{total_loss:.4f}  Accuracy:{total_correct/total_samples:.4f}  time:{time.time()-start_time}')
    torch.save(model.state_dict(),'model.pth')

def evaluate(test_loader,model,criterion,device):
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    model.to(device)
    total_loss,total_correct,total_samples,start_time=0,0,0,time.time()
    with torch.no_grad():
        for x,y in test_loader:
            x,y=x.to(device),y.to(device)
            y_predict=model(x)
            loss=criterion(y_predict,y)
            total_loss+=loss.item()
            total_samples+=y.size(0)
            total_correct+=torch.sum(torch.argmax(y_predict,dim=-1)==y).item()
        print(f'total_loss:{total_loss}  Accuracy:{total_correct/total_samples:.4f}  time:{time.time()-start_time}')
    torch.save(model.state_dict(),'model.pth')
if __name__ == '__main__':
    #train(train_loader,model,Epochs,criterion,optimizer,device)
    evaluate(test_loader,model,criterion,device)




