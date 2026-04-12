"""
案例：
    演示学习率衰减策略

学习率衰减策略介绍：
    目的：
        较之于AdaGrad,RMSProp,Adam方式，我们可以通过 等间隔，指定间隔，指数等方式，来手动控制学习率的调整

    分类：
        等间隔学习率衰减
        指定间隔学习率衰减
        指数学习率衰减

    1、等间隔学习率衰减：
        step_size: 间隔的轮数，即：多少轮调整一次学习率
        gamma：     学习率衰减系数，即 lr新 = lr旧 * gamma
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

def dm01():
    lr,epochs,iteration = 0.1,200,10

    y_true=torch.tensor([0])
    x=torch.tensor([1,0],dtype=torch.float)
    w=torch.tensor([1,0],dtype=torch.float,requires_grad=True)

    optimizer=optim.SGD([w],lr=lr,momentum=0.9)

    scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)

    lr_list,epoch_list=[],[]

    for epoch in range(epochs):
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_lr())

        for batch_num in range(iteration):
            y_pre= w*x
            loss=(y_pre-y_true)**2

            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

        scheduler.step()

    print(f'lr_list:{lr_list},epoch_list:{epoch_list}')

    plt.plot(epoch_list,lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    dm01()