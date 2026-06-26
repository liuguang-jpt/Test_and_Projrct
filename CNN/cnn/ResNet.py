import torch
import torch.nn as nn

"""
ResNet ”阶梯式“特征提取主干
ResNet从低分辨率、低通道的浅层特征，逐步提取到高语义、高通道的深层特征
ResNet的主体结构遵循：1个初始卷积池化模块+4个残差阶段+1个分类头
输入图像 → b1(初始卷积池化) → b2(残差阶段1) → b3(残差阶段2) → b4(残差阶段3) → b5(残差阶段4) → 池化 → 全连接 → 分类输出

ResNet-18:每个残差阶段创建2个残差块  num_residuals=2
ResNet-34:每个残差阶段创建3/4/6/3个残差块  num_residuals=(3,4,6,3)
ResNet-50/101/152 ... 
"""
#残差块
class Residual_Block(nn.Module):
    # 恒等映射 + 残差学习
    def __init__(self,input_channels,num_channels,use_1x1conv=False,strides=1):
        super(Residual_Block,self).__init__()
        self.conv1=nn.Conv2d(input_channels,num_channels,3,strides,1)
        self.conv2=nn.Conv2d(num_channels,num_channels,3,padding=1)
        if use_1x1conv:
            #投影映射 改变维使数据无损传播
            self.conv3=nn.Conv2d(input_channels,num_channels,1,strides)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)
        self.rule=nn.ReLU(inplace=True)

    def forward(self,x):
        """
        残差学习  网络学习的不是输入到输出的完整映射，而是输入和输出之间的 "差值"（残差）
        学习残差更容易
        """
        y=torch.relu(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))
        if self.conv3:
            x=self.conv3(x)
        """
        恒等映射 残差块的最终输出 = 卷积学习到的特征 + 原始输入
        use_1x1conv==False时(第一个残差阶段的全部和除第一个残差阶段的除第一个残差块的全部)x没有经过任何卷积处理,直接加到了卷积输出y上
        作用：
            1、解决梯度消失问题
              反向传播时，梯度可以直接通过y += x这条路径，从最后一层无损地传到第一层
            2、解决深度网络退化问题
              网络深度增加时，性能至少不会下降，只会保持不变或提升
            3、降低学习难度
              网络不需要从零开始学习一个复杂的函数，只需要在原始输入的基础上做微小的调整
        """
        y+=x
        return torch.relu(y)

#ResNet第一个预处理卷积池化模块
b1=nn.Sequential(
    nn.Conv2d(1,64,7,2,3),
    nn.BatchNorm2d(64),nn.ReLU(),
    nn.MaxPool2d(3,2,1),
)

def resnet_block_period(input_channels,num_channels,num_residuals,first_block=False):
    blk=[]
    for i in range(num_residuals):
        if i==0 and not first_block:
            #if i==0 and not first_block 非第一个残差阶段的第一个残差块
            blk.append(Residual_Block(input_channels,num_channels,use_1x1conv=True,strides=2))
            """
            这是每个残差阶段的第一个残差块(除第一阶段的b2),它承担了ResNet的两个最核心的功能
            1、下采样,特征图尺寸减半
            strides=2:让特征图高和宽都变为原来的1/2
                降低计算量：特征图尺寸减半，后续卷积的计算量变为原来的1/4
                扩大感受野：深层神经元能看到更大的图像区域，提取更全局的视野
                符合视觉认知规律：从边缘、纹理等局部特征，逐步过渡到物体、场景等全局特征
            2、通道升维(存储更多的特征信息：深层网络需要更多的通道来表示复杂的语义特征)，特征维度翻倍
                输入通道：input_channels(上一个阶段的输出通道)
                输出通道：num_channels(通常是input_channels的2倍)
            3、1x1卷积解决维度不匹配的问题
            """
        else:
            blk.append(Residual_Block(input_channels,num_channels))

    return blk

#ResNet 每个残差阶段都有两个残差块
b2=nn.Sequential(*resnet_block_period(64,64,2,first_block=True))
"""
第一个残差阶段不需要下采样
    前面的b1模块已经做了两次下采样，如果b2再下采样，会丢失很多信息
"""
b3=nn.Sequential(*resnet_block_period(64,128,2))
b4=nn.Sequential(*resnet_block_period(128,256,2))
b5=nn.Sequential(*resnet_block_period(256,512,2))

net=nn.Sequential(b1,b2,b3,b4,b5,
                  nn.AdaptiveAvgPool2d((1,1)),
                  nn.Flatten(),nn.Linear(512,10))

