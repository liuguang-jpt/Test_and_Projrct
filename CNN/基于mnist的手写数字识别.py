import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform,download=False)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform,download=False)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

class MNist(nn.Module):
    def __init__(self):
        super().__init__()
        #(batch_size,channels,height,width) (batch,1,28,28) [批次，通道，高，宽]
        self.conv1=nn.Conv2d(1,32,3,1)  #(batch,32,26,26)
        self.conv2=nn.Conv2d(32,64,3,1) #(batch,64,24,24)
        self.pool=nn.MaxPool2d(2,2) #(batch,64,12,12)
        self.relu=nn.ReLU()
        self.linear1=nn.Linear(64*12*12,128)
        self.linear2=nn.Linear(128,10)

    def forward(self,x):
        x=self.relu(self.conv1(x))
        x=self.pool(self.relu(self.conv2(x)))
        x=x.view(-1,64*12*12) #(batch,64*12*12)
        x=self.relu(self.linear1(x))
        output=self.linear2(x)
        return output

model = MNist().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

def train(train_loader,model,criterion,optimizer,epochs):
    model.train()
    for epoch in range(epochs):
        total_loss,total_correct,total_number,start_time=0,0,0,time.time()
        for i,(x,y) in enumerate(train_loader):
            x,y=x.to(device),y.to(device)

            y_predict=model(x)
            loss=criterion(y_predict,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
            total_number+=y.size(0)
            total_correct += (y_predict.argmax(dim=-1) == y).sum().item()
        print(f'epoch:{epoch+1}  loss:{total_loss:.6f}  Acc:{total_correct/total_number:.6f}  time:{time.time()-start_time}')
    torch.save(model.state_dict(),'./model.pth')

def evaluate(test_loader, model, criterion, epoches):
        model.load_state_dict(torch.load('./model.pth'))
        model.eval()
        with torch.no_grad():
            total_loss, total_correct, total_number, start_time = 0, 0, 0, time.time()
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                y_predict = model(x)
                total_loss += criterion(y_predict, y).item()
                total_correct += (y_predict.argmax(dim=-1) == y).sum().item()
                total_number += y.size(0)
            print(f'平均损失:{total_loss / total_number}  Acc:{total_correct / total_number:.6f} time:{time.time()-start_time}')

if __name__ == '__main__':
    train(train_loader,model,criterion,optimizer,epochs)
    evaluate(test_loader,model,criterion,epochs)