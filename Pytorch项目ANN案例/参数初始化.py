"""
演示参数初始化的7种方式：

    参数初始化的目的：
    1、防止梯度消失，或者  梯度爆炸
    2、提高收益速度
    3、打破对称性

参数初始化的方式：
    无法打破对称性的：
        全0，全1
    可以打破对称性的：
        随机初始化，正态分布初始化，kaiming初始化，xavier初始化

    总结：
        1、记忆 kaiming初始化，xavier初始化，全0初始化
        2、关于参数初始化选择上:
            激活函数是ReLU，优先使用kaiming
            激活函数非ReLU 优先使用xavier
            如果是浅层神经网络：可以考虑使用 随机初始化
"""

import torch.nn as nn

#均匀分布初始化
def dm01():
    #创建一个线性层，输入维度为5，输出维度为3
    linear=nn.Linear(in_features=5,out_features=3)
    #对权重w进行随机初始化，从0~1均匀随机产生参数
    nn.init.uniform_(linear.weight)
    #对偏置b
    nn.init.uniform_(linear.bias)

    print(linear.weight.data)
    print(linear.bias.data)

if __name__ == '__main__':
    dm01()