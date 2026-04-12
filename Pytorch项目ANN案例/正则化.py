"""
案例：
    代码演示随即失活

正则化的作用：
    缓解模型的过拟合

正则化的方式：
    L1正则化:权重可以变成0，相当于 降维
    L2正则化:权重可以无限接近于0
    Dropout:随机失活，每批次样本训练时，随机让一部分神经元死亡，防止一些特征对结果的影响较大(过拟合)
    BN(批量归一化):
        先对数据标准化(丢失一些数据),然后再对数据做 缩放(λ，理解为：w权重) 和 平移(β，理解为:b偏置),再找补回一些信息
        应用场景：
            批量归一化在计算机视觉领域使用较多

        BatchNorm1d:主要应用于全连接层或处理一维数据的网络，例如文本数据，它接受形状为(N, num_features)张量作为输入
        BarchNorm2d:主要应用于卷积神经网络，处理二维图像数据或者特征图，它接受形状为(N,C,H,M)的张量作为输入
        BatchNorm3d:主要应用于三维卷积神经网络(3D CNN),处理三维数据，例如视频或医学图像，它接受形状为(N,C,D,H,W)的张量作为输入

"""

import torch
import torch.optim as optim
import torch.nn as nn

def dm01():
    #创建隐藏层输出结果
    t1=torch.randint(1,10,(1,4)).float()
    print(f't1:{t1}')
    #进行下一层 加权求和和激活函数
    linear1=nn.Linear(4,5)

    l1=linear1(t1)
    print(f'l1:{l1}')
    output=torch.relu(l1)
    print(f'output:{output}')
    #对激活值进行随机失活dropout处理 --> 只有训练阶段有，测试阶段没有
    dropout=nn.Dropout(0.4)

    d1=dropout(output)
    print(f'd1:{d1}')
def dm02():
    #创建图像样本数据
    input_2d=torch.randn((1,2,3,4))
    print(f'input_2d:{input_2d}')

    #创建批量归一化层（BN层）
    #参1：输入特征数 = 图片的通道数
    #参2：噪声值（小常量），默认为1e-5
    #参3：动量值，用于计算平均移动平局统计量的  动量值
    #参4：表示使用可学习的变换参数（λ，β）对归一化（标准化）后的数据进行 缩放和平移
    bn2d=nn.BatchNorm2d(2,1e-5,0.1,affine=True)

    output=bn2d(input_2d)
    print(f'output:{output}')
def dm03():
    input_1d=torch.randn((2,2))
    linear1=nn.Linear(2,4)
    output=linear1(input_1d)
    print(f'output:{output}')

    bn1d=nn.BatchNorm1d(4)
    output_1d=bn1d(output)
    print(f'output_1d:{output_1d}')

if __name__ == '__main__':
    dm03()