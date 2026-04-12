"""
神经网络搭建流程：
    1、定义一个类，继承：nn.Module
    2、在_init_方法中，搭建神经网络
    3、在forward()方法中，完成：前向传播

深度学习的4个步骤：
    1、准备数据
    2、搭建神经网络
    3、模型训练
    4、模型测试

"""

import torch
import torch.nn as nn
from torchsummary import summary  #计算模型参数

#搭建神经网络
class ModelDemo(nn.Module):
    #在init魔法方法中，完成初始化，父类成员及神经网络搭建
    def __init__(self):
        #初始化父类成员
        super().__init__()
        #搭建神经网络-->隐藏层 + 输出层
        self.linder1=nn.Linear(3,3)
        self.linder2=nn.Linear(3,2)
        self.output=nn.Linear(2,2)

        #对隐藏层初始化
        nn.init.xavier_normal_(self.linder1.weight)
        nn.init.zeros_(self.linder1.bias)

        nn.init.kaiming_normal_(self.linder2.weight)
        nn.init.zeros_(self.linder2.bias)

    #前向传播
    def forward(self,x):
        #第一层 隐藏层计算： 加权求和 + 激活函数(Sigmoid)
        x=self.linder1(x)  #加权求和
        x=torch.sigmoid(x) #激活函数

        #x=torch.sigmoid(self.linder1(x))

        #第二层 隐藏层计算
        x=torch.relu(self.linder2(x))

        #第三层
        x=torch.softmax(self.output(x),dim=-1) #按行计算

        #返回预测值
        return x

#模型训练、
def train():
    #创建模型对象
    model = ModelDemo()

    #创建数据集样本
    data=torch.randn(size=(5,3))
    print(f'data: {data}')
    print(f'data.shape: {data.shape}')
    print(f'data.requires_grad: {data.requires_grad}')

    #调用神经网络模型，进行模型训练
    output=model(data)  #自动调用forward方法，进行前向传播
    print(f'output: {output}')
    print(f'output.shape: {output.shape}')
    print(f'output.requires_grad: {output.requires_grad}')


    #打印 计算 和 查看模型参数
    print("========== 计算模型参数 ==========")
    #神经网路模型参数 输入数据维度


if __name__ == '__main__':
    train()





