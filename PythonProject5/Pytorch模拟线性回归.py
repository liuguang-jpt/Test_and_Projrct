"""
    Pytorch模拟线性回归

1、准备训练集数据
2、构建要使用的模型
3、设置损失函数和优化器
4、模型训练

API
    nn.MESLoss() 平方损失函数
    data.DataLoader  数据加载器
    optim.SGD  优化器
    nn.Linder  假设函数

    numpy对象 --> 张量Tensor --> 数据集对象TensorDataset --> 数据加载器DataLoader

"""

import torch
from sklearn.datasets import make_regression
from torch.utils.data import TensorDataset        #构建数据集对象
from torch.utils.data import DataLoader           #数据加载器
from torch import nn                              #nn模块中有平方损失函数和假设函数
from torch import optim                           #optim模块中有优化器函数
from sklearn.linear_model import LinearRegression #创建线性回归模拟数据集
import matplotlib.pyplot as plt                   #可视化

plt.rcParams['font.sans-serif']=['SimHei']        #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False          #用来正常显示负号

#1、定义函数，创建线性回归样本数据集
def create_dataset():
    #创建数据集对象
    x,y,coef=make_regression(
        n_samples=100,   #100条样本(100个样本点)
        n_features=1,    #1个特征
        noise=10,        #噪声，噪声越大，样本点越散
        coef=True,       #是否返回系数，默认为False，返回值为None
        random_state=3   #随机种子
    )

    #把上述数据封装为Tensor
    x=torch.tensor(x,dtype=torch.float)  #特征
    y=torch.tensor(y,dtype=torch.float)  #标签

    #返回结果
    return x,y,coef
#2、定义函数，表示模型训练
def train(x,y,coef):
    # 1、创建数据集对象，把tensor-->数据集对象-->数据加载器
    dataset=TensorDataset(x,y)

    #2、创建数据加载器对象
    #参数1：数据集对象，参数2：批次大小，参数3：是否打乱数据(训练集打乱，测试集不打乱)
    dataloader=DataLoader(dataset,batch_size=16,shuffle=True)

    #3、创建初始的线性回归模型
    #参数1：输入特征维度，参数2：输出特征维度
    model=nn.Linear(1,1)

    #4、创建损失函数对象
    criterion = nn.MSELoss()

    #5、创建优化器对象
    #参数1：模型参数，参数2：学习率
    optimizer=optim.SGD(model.parameters(),lr=0.01)

    #6、具体的训练过程
    #6.1 定义变量，分别表示：训练轮数，每轮的(平均)损失值，训练的总损失数，训练的样本数
    epochs,loss_list,total_loss,total_samples=100,[],0,0
    #6.2 开始训练，按轮训练
    for epoch in range(epochs):
        #6.3 每轮是分批次训练的，所以从数据加载器中获取批次数据
        for train_x,train_y in dataloader:  # 7批 [16,16,16,16,16,16,4],每轮的数据不一样
            #6.4 模型的预测
            y_pared = model(train_x)
            #6.5 计算每批的平均损失
            loss=criterion(y_pared,train_y.reshape(-1,1))  #-1 自动计算 n行1列
            #6.6 计算总损失 和 样本(批次)数
            total_loss+=loss.item()  #每批平均损失
            total_samples+=1         #+1而不是+16
            #6.7 梯度清零 + 反向传播
            optimizer.zero_grad() #底层已经判断非None
            loss.sum().backward()
            optimizer.step()      #梯度更新

        #6.8 把本轮的平均损失值，添加到列表中
        loss_list.append(total_loss/total_samples)
        print(f'轮数：{epoch+1},平均损失值：{total_loss/total_samples}')

    #7、打印最终的训练结果
    print(f'{epochs}轮的平均损失值分别为L{loss_list}')
    print(f'模型参数，权重：{model.weight},偏置：{model.bias}')

    #8、绘制损失曲线
    #              100轮       每轮的平均损失值
    plt.plot(range(epochs),loss_list)
    plt.title("损失值曲线变化图")
    plt.grid(True)  #绘制网格线
    plt.show()
    #9、绘制预测值和真实值的关系

    #9.1 绘制样本点分布情况
    plt.scatter(x.numpy(),y.numpy())
    #9.2 绘制训练模型的预测值
    #x:100个样本点的特征
    y_pred=model(x).detach().numpy()
    #9.3 计算真实值
    y_true=(x*coef).detach().numpy()
    #9.4 绘制预测值和真实值的折线图
    plt.plot(x,y_pred,color='red',label='预测值')
    plt.plot(x,y_true,color='green',label='真实值')
    #9.5 图例，网格
    plt.legend()
    plt.grid()
    #9.6 显示图像
    plt.show()


if __name__ == '__main__':
    #构造数据集
    x,y,coef=create_dataset()
    train(x,y,coef)