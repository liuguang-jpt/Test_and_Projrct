"""
    卷积层API 用于获取图像的局部特征，获取特征图（Feature Map）

    卷积神经网络：
        Convolutional neural network

    组成：
        卷积层(Convolutional)
            用于提取图像的局部特征，结合卷积核（每个卷积核=一个神经元） 实现，处理后的结果叫特征图
        池化层(Pooling)
            用于降维，降采样
        全连接层(Full Connected,fc,linear)
            用于预测结果，并输出结果
    特征图的计算方式：
        N = ( W -F + 2 * P ) / S + 1
        W: 输入的图像的大小
        F: 卷积核的大小
        P: 填充的大小
        S: 步长
        N: 输出图像的大小

    池化：降维，不会改变数据的通道数
        nn.MaxPool2d(kernel_size=2, stride=2) 最大池化
        nn.AvgPool2d(kernel_size=2, stride=2) 平均池化
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from altair import ChainedWhen

def dm02():

    img = plt.imread('微信图片_20260412220329_248924_1.png')  # (1531,2737,4) HWC
    #print(f'img:{img},shape:{img.shape}')

    #把图像的形状从 HWC -> CHW  img->tensor->转换维度
    img2=torch.tensor(img,dtype=torch.float32)

    img2=img2.permute(2,0,1)  #(4,1531,2737)

    # 因为这里只有一张图，所以再增加一个维度，从CHW -> (1,C,H,W),1张3通道的640*640像素 (1,4,1531,2737)
    img3=img2.unsqueeze(0)

    #输入通道数  输出通道数（几个特征图，卷积核个数）  卷积核大小  步长  填充
    conv=nn.Conv2d(4,4,3,3,0)

    conv_img=conv(img3)

    #print(f'conv_img:{conv_img},shape:{conv_img.shape}')

    img4=conv_img[0]  #CHW

    img5=img4.permute(1,2,0)

    #可视化 第一个通道的可视化
    feature1=img5[:,:,0].detach().numpy() #第一通道的像素图
    feature2=img5[:,:,1].detach().numpy()
    feature3=img5[:,:,2].detach().numpy()
    feature4=img5[:,:,3].detach().numpy()

    plt.imshow(feature1)
    plt.show()
    plt.imshow(feature2)
    plt.show()
    plt.imshow(feature3)
    plt.show()
    plt.imshow(feature4)
    plt.show()

def dm03():
    inputs=torch.tensor([[
        [0,1,2],[3,4,5],[6,7,8]
    ]])
    pool1=nn.MaxPool2d(kernel_size=2, stride=1)
    outputs=pool1(inputs)
    print(outputs)

    pool2=nn.AvgPool2d(kernel_size=2, stride=1)
    outputs2=pool2(inputs)
    print(outputs2)


if __name__ == '__main__':
    dm03()