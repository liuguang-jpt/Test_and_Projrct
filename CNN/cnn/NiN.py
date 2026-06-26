import torch.nn as nn

def nin_block(input_channels,output_channels,kernel_size,strides,padding):
    return nn.Sequential(
        nn.Conv2d(input_channels,output_channels,kernel_size,strides,padding),nn.ReLU(),
        nn.Conv2d(output_channels,output_channels,kernel_size=1,stride=1,padding=0),nn.ReLU(),
        nn.Conv2d(output_channels,output_channels,kernel_size=1,stride=1,padding=0),nn.ReLU())

net=nn.Sequential(
    nin_block(1,96,11,4,0),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
    nin_block(96,256,5,1,1),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
    nin_block(256,384,3,1,1),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
    nin_block(384,10,3,1,1),
    nn.AdaptiveAvgPool2d(output_size=(1,1)),nn.Flatten())

