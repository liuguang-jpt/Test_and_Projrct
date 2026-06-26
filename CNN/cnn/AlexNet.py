
import torch.nn as nn

net=nn.Sequential(
    nn.Conv2d(1,96,11,4,1),nn.ReLU(),
    nn.MaxPool2d(3,2),
    nn.Conv2d(96,256,5,1,2),nn.ReLU(),
    nn.MaxPool2d(3,2),
    nn.Conv2d(256,384,3,1,1),nn.ReLU(),
    nn.Conv2d(384,512,3,1,1),nn.ReLU(),
    nn.Conv2d(512,512,3,1,1),nn.ReLU(),
    nn.MaxPool2d(3,2),nn.Flatten(),
    nn.Linear(512*6*6,4096),nn.ReLU(),nn.Dropout(0.5),
    nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),
    nn.Linear(4096,1000)
)



