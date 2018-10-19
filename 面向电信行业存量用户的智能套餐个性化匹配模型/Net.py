# -*-coding:utf-8-*-
from torch import nn
import torch
import torch.nn.functional as F
import math

#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.hidden = nn.Linear(25, 30)   # hidden layer
#         self.out = nn.Linear(30, 10)   # output layer
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))      # activation function for hidden layer
#         x = self.out(x)
#         return x
#
# class Net(nn.Module):  # 网络结构
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(25, 1, 1)
#         self.conv2 = nn.Conv2d(25, 16, 1)
#         self.fc1 = nn.Linear(16, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(x.size()[0], -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 96, 11, 4, 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(256,384, 3, 1, 1),
            torch.nn.ReLU(),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(384,384, 3, 1, 1),
            torch.nn.ReLU(),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(384,256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2)
        )
        # self.dense = torch.nn.Sequential(
        #     torch.nn.Linear(9216, 4096),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.Linear(4096, 4096),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.Linear(4096, 10)
        # )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(6400, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 10)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        res = conv5_out.view(conv5_out.size(0), -1)
        out = self.dense(res)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
