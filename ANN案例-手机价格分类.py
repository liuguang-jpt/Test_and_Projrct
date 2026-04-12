"""
案例：
    ANN(人工神经网络)案例：手机价格分类

背景：
    基于手机20列特征 --> 预测手机的价格区间(4个区域)

ANN案例的实现步骤：
    1、构建数据集
    2、搭建神经网络
    3、模型训练
    4、模型测试

优化思路：
    1、优化方法从SGD-->Adam
    2、学习率变成0.0001
    3、对数据进行标准化
    4、增加网络的深度，每层的神经元个数，增加轮数
"""
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset  # 数据集对象， 数据-->tensor-->数据集-->数据加载器
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression  #数据集和测试集的划分
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from torchsummary import summary


def create_dataset():
    #1、加载CSV文件数据集
    data=pd.read_csv('手机价格预测(1).csv') #2000,21

    #2、获取x特征列 和 y标签列
    x,y=data.iloc[:,:-1],data.iloc[:,-1]

    #3、把特征列转成浮点型
    x=x.astype('float32')

    # 4、切分数据集和测试集
    # 参1：特征，参2：标签，参3：测试集所占比例，参4：随机种子，参5：样本分布特征（参考y的类别进行抽取数据）
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)

    #优化：数据标准化
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)


    #5、把数据集封装成张量数据集，数据-->张量-->数据集TensorDataset-->数据加载器DataLoader
    train_dataset=TensorDataset(torch.from_numpy(x_train),torch.from_numpy(y_train.values))
    test_dataset=TensorDataset(torch.from_numpy(x_test),torch.from_numpy(y_test.values))

    #6、返回结果                         20充当输入特征数    去重之后输出标签数
    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y))

class PhonePriceModel(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(PhonePriceModel,self).__init__()
        self.linear1=nn.Linear(input_dim,128)
        self.linear2=nn.Linear(128,256)
        self.linear3=nn.Linear(256,512)
        self.linear4=nn.Linear(512,256)
        self.linear5=nn.Linear(256,output_dim)

    def forward(self,x):
        x=torch.relu(self.linear1(x))
        x=torch.relu(self.linear2(x))
        output=self.linear3(x)
        return output

def train(train_dataset,input_dim,output_dim):
    #创建数据加载器
    #参1：数据集对象，参2：每批次的数据条数，每轮要训练100批，参3：是否打乱数据
    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)

    #创建神经网络模型对象
    model = PhonePriceModel(input_dim, output_dim)

    #创建损失函数 多分类交叉熵损失函数CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    #创建优化器
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    #模型训练
    #定义总轮数
    epochs = 200
    for epoch in range(epochs):
        total_loss,batch_num=0.0,0
        start_time=time.time()
        for x,y in train_loader:
            #切换模型状态
            model.train() #训练模式
            y_pred=model(x)
            loss=criterion(y_pred,y)
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            total_loss+=loss.item()
            batch_num=batch_num+1
        #本轮训练结束，打印训练结果
        #print(f'epoch:{epoch+1},loss:{total_loss/batch_num:.4f},time:{time.time()-start_time:.2f}')

    #多轮训练结束，保存模型参数
    #参1：模型对象参数（权重矩阵，偏置矩阵），参2：模型保存的文件名
    #print(f'\n\n模型的参数信息：{model.state_dict()}')
    torch.save(model.state_dict(),'model.pth')

#模型测试
def evaluate(test_dataset,input_dim,output_dim):
    model = PhonePriceModel(input_dim, output_dim)
    model.load_state_dict(torch.load('model.pth'))

    test_loader = DataLoader(test_dataset,batch_size=8,shuffle=True)

    #定义变量，记录预测正确的样本个数
    correct=0

    for x,y in test_loader:
        model.eval()
        y_pred=torch.argmax(model(x),dim=1)
        #print(f'y_pred:{y_pred},y:{y}')
        #根据加权求和，得到类别，用argmax()获取最大值的下标，定义model的时候没有定义softmax()
        correct+=torch.sum(y==y_pred)

    print(f'准确率：{correct/len(test_dataset)}')



if __name__ == '__main__':
    train_dataset,test_dataset,input_dim,output_dim=create_dataset()
    train(train_dataset,input_dim,output_dim)
    evaluate(test_dataset,input_dim,output_dim)
    #计算模型参数
    #summary(model,input_size=(16,input_dim,),device='cpu')


