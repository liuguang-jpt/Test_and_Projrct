"""
梯度下降优化方法


            SGD算法

概述：
    梯度下降是结合 本次损失函数的导数(作为梯度), 基于学习率  来更新权重
公式：
    W新 = W旧 - 学习率 * 梯度
存在的问题：
    1、遇到平滑区域，梯度下降(权重更新)可能会慢
    2、可能会遇到鞍点 （梯度为0）
    3、可能会遇到 局部最小值
解决思路：
    从上述的 学习率 或者 梯度入手，进行优化，于是有了：动量法Momentum，自适应学习率AdaGrad，RMSProp，综合衡量：Adam

1、动量法Momentum
    公式：
        St = β * St-1 + (1-β) * Gt
    解释：
        St：本次的指数移动加权平均结果
        β：调节权重系数，越大，数据越平稳，历史指数移动加权平均 比重越大，本次梯度权重越小
        St-1： 历史的指数移动加权平均结果
        Gt： 本次计算出的梯度

    加入动量法后的 梯度更新公式：
        W新 = W旧 - 学习率 * St
2、自适应学习率：AdaGrad
    公式：
        累计平方梯度：
            St = St-1 + Gt * Gt
            解释：
            St: 累计平方梯度
            St-1: 历史累计平方梯度
            Gt: 本次的梯度

        学习率 = 学习率 / (sqrt(St) + 小常数)

         梯度下降公式：
             W新 = W旧 - 调整后的学习率 * Gt
         缺点：
             可能会导致学习率过早过量的降低，导致模型后期学习率太小，较难找到最优解
3、自适应学习率：RMSProp  -->  可以看成是对AdaGrad做的优化，加入  调和权重系数
    公式：
        指数加权平均 累计平方梯度：
            St =β * St-1 + (1-β) * Gt * Gt
            解释：
            St: 累计平方梯度
            St-1: 历史累计平方梯度
            Gt: 本次的梯度

        学习率 = 学习率 / (sqrt(St) + 小常数)

         梯度下降公式：
             W新 = W旧 - 调整后的学习率 * Gt
         优点：
             RMSProp通过引入  衰减系数β，控制历史梯度 对 历史梯度信息获取的多少
4、自适应矩估计：Adam(Adaptive Moment Estimation)
    思路：
        优化学习率和梯度
    公式：
        一阶段：算均值
            Mt = β1 * Mt-1 + (1-β) * Gt      充当梯度
            St = β2 * St-1 + (1-β) * Gt * Gt 充当学习率
        二阶段：算方差
            Mt^ = Mt / (1-β1^t)
            St^ = St / (1-β2^t)
        权重更新公式：
            W新 = W旧 - 学习率 / (sqrt(St^) + 小常数) * Mt^
        大白话翻译：
            Adam = RMSProp + Momentum

总结：如何选择梯度下降优化方法
    简单任务和较小的模型
        SGD，动量法
    复杂任务或者有大量数据
        Adam
    需要处理稀疏数据或者文本数据
        AdaGrad，RMSProp
"""
import torch
import torch.nn as nn

#动量法Momentum
def dm01():
    #初始化权重参数
    w=torch.tensor([1,0],requires_grad=True,dtype=torch.float)
    #定义损失函数
    criterion=((w**2)/2.0)
    #创建优化器（函数对象）--> 基于SGD（随机梯度下降），加入参数 momentum 就是动量法
    optimizer=torch.optim.SGD([w],lr=0.01,momentum=0.9)   #momentum=0（默认），只考虑吧本次梯度
    #计算梯度值:梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")

    #重复上述步骤，再次更新权重
    criterion=((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")

def dm02():
    # 初始化权重参数
    w = torch.tensor([1, 0], requires_grad=True, dtype=torch.float)
    # 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 创建优化器（函数对象）--> 基于SGD（随机梯度下降），加入参数 momentum 就是动量法
    optimizer = torch.optim.Adagrad([w], lr=0.01)
    # 计算梯度值:梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")

    # 重复上述步骤，再次更新权重
    criterion = ((w ** 2) / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")

def dm03():
    # 初始化权重参数
    w = torch.tensor([1, 0], requires_grad=True, dtype=torch.float)
    # 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 创建优化器（函数对象）--> 基于SGD（随机梯度下降），加入参数 momentum 就是动量法
    optimizer = torch.optim.RMSprop([w], lr=0.01,alpha=0.99)  #alpha 权重系数
    # 计算梯度值:梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")

    # 重复上述步骤，再次更新权重
    criterion = ((w ** 2) / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")

def dm04():
    # 初始化权重参数
    w = torch.tensor([1, 0], requires_grad=True, dtype=torch.float)
    # 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 创建优化器（函数对象）--> 基于SGD（随机梯度下降），加入参数 momentum 就是动量法
    optimizer = torch.optim.Adam([w], lr=0.01, betas=(0.9,0.99))  # (梯度用的衰减系数，学习率用的衰减系数)
    # 计算梯度值:梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")

    # 重复上述步骤，再次更新权重
    criterion = ((w ** 2) / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")

if __name__ == '__main__':
    dm01()
    dm02()
    dm03()
    dm04()



