import torch.nn as nn

def vgg_block(num_conv,in_channels,out_channels):
    layers=[]
    for i in range(num_conv):
        layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

conv_arch=((1,64),(1,128),(2,256),(2,512),(2,512))

def vgg_achieve(conv_arch):
    conv_blocks=[]
    input_channels=1
    for (num_conv,out_channels) in conv_arch:
        conv_blocks.append(vgg_block(num_conv,input_channels,out_channels))
        input_channels = out_channels

    return nn.Sequential(*conv_blocks,nn.Flatten(),nn.Linear(input_channels*7*7,4096),
                         nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),nn.Linear(4096,10))

net=vgg_achieve(conv_arch)