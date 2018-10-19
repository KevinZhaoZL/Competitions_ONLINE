# -*-coding:utf-8-*-
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.autograd import Variable
from Net import Net
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
from torch import nn
import random

CUDA_LAUNCH_BLOCKING = 1

epoch_num = 50
lr = 0.05
batch_size = 65
net = torchvision.models.resnet50(pretrained=True)
for param in net.parameters():
    param.requires_grad = False

net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
net.fc = nn.Linear(8192, 11)
print net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 第一行代码
# net.to(device)  # 第二行代码
net = net.cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted


class Preprocess():
    testData_dir = '/home/yuanye/ubuntu-windows/data/toubuntudektop/republish_test.csv'
    trainData_dir = '/home/yuanye/ubuntu-windows/data/toubuntudektop/train_all.csv'
    rows_train = pd.read_csv(trainData_dir)
    rows_test = pd.read_csv(trainData_dir)

    def getBasicInfo(self, rows=rows_train):
        # field names
        fieldNames = list(rows.columns.values)
        # labels
        labelSet = set(rows['current_service'])
        # shape
        row_num = len(rows['current_service'])
        column_num = len(list(rows.columns.values))
        return fieldNames, labelSet, row_num, column_num

    def rows2tensor(self, rows=rows_train):
        fieldNames, labelSet, row_num, column_num = self.getBasicInfo(rows=rows)
        returnMat = np.zeros((row_num, column_num - 2))
        index = 0
        returnLabelVector = list(rows['current_service'])
        for i in range(len(returnLabelVector)):
            if returnLabelVector[i] == 99999830:
                returnLabelVector[i] = 0
            elif returnLabelVector[i] == 90063345:
                returnLabelVector[i] = 1
            elif returnLabelVector[i] == 90155946:
                returnLabelVector[i] = 2
            elif returnLabelVector[i] == 99999825:
                returnLabelVector[i] = 3
            elif returnLabelVector[i] == 99999826:
                returnLabelVector[i] = 4
            elif returnLabelVector[i] == 99999827:
                returnLabelVector[i] = 5
            elif returnLabelVector[i] == 99999828:
                returnLabelVector[i] = 6
            elif returnLabelVector[i] == 89950166:
                returnLabelVector[i] = 7
            elif returnLabelVector[i] == 89950167:
                returnLabelVector[i] = 8
            elif returnLabelVector[i] == 89950168:
                returnLabelVector[i] = 9
            else:
                returnLabelVector[i] = 10

        for i in range(column_num - 2):
            tmp = list(rows[fieldNames[i]])
            for j in range(len(tmp)):
                if tmp[j] == '\N' or tmp[j] == '#VALUE!':
                    tmp[j] = 0
                if tmp[j] == 'nan':
                    tmp[j] = 255
                print i, j, tmp[j]
                tmp[j] = float(tmp[j])
            returnMat[:, index] = tmp[0:]
            index = index + 1

        retTrainData = []
        # for i in range(743990):

        for epoch in range(5):
            loss_sum = 0
            for i in range(743990):
                if (i + 1) % 65 == 0:
                    retMat = np.zeros((65, 1, 225, 225))
                    retLabel = []
                    for j in range(65):
                        randindex = random.randint(0, 743989)
                        retMat[j] = np.tile(returnMat[randindex], 2025).reshape((1, 225, 225))
                        retLabel.append(returnLabelVector[randindex])
                        tempM = torch.from_numpy(retMat)
                        tempL = torch.Tensor(retLabel)
                        feature = tempM.type(torch.FloatTensor)
                        label = tempL.type(torch.LongTensor)
                        feature_v = Variable(feature)
                        label_v = Variable(label)
                    feature_v = feature_v.to(device)
                    label_v = label_v.to(device)

                    out = net(feature_v)  # input x and predict based on x
                    loss = loss_func(out,
                                     label_v)  # must be (1. nn output, 2. target), the target label is NOT one-hotted
                    loss_sum += loss.data[0]
                    optimizer.zero_grad()  # clear gradients for next train
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradient
                    # prediction = torch.max(F.softmax(out), 1)[1]
                    # print iter
                    # print prediction,label_v,loss

            print 'Epoch: %d, Loss: %f' % (epoch + 1, loss_sum.item() / 11446)
            torch.save(net.state_dict(), '%d_Tele_params_all_batch65Shffule_Original_lr1e-05.pkl'%(epoch+1))
            print "XXXXXXXX"
            for epoch in range(5):
                loss_sum = 0
                for i in range(743990):
                    if (i + 1) % 65 == 0:
                        retMat = np.zeros((65, 1, 225, 225))
                        retLabel = []
                        for j in range(65):
                            # randindex = random.randint(0, 743989)
                            retMat[j] = np.tile(returnMat[i], 2025).reshape((1, 225, 225))
                            retLabel.append(returnLabelVector[i])
                            tempM = torch.from_numpy(retMat)
                            tempL = torch.Tensor(retLabel)
                            feature = tempM.type(torch.FloatTensor)
                            label = tempL.type(torch.LongTensor)
                            feature_v = Variable(feature)
                            label_v = Variable(label)
                        feature_v = feature_v.to(device)
                        label_v = label_v.to(device)

                        out = net(feature_v)  # input x and predict based on x
                        loss = loss_func(out,
                                         label_v)  # must be (1. nn output, 2. target), the target label is NOT one-hotted
                        loss_sum += loss.data[0]
                        optimizer.zero_grad()  # clear gradients for next train
                        loss.backward()  # backpropagation, compute gradients
                        optimizer.step()  # apply gradient
                        # prediction = torch.max(F.softmax(out), 1)[1]
                        # print iter
                        # print prediction,label_v,loss

                print 'Epoch: %d, Loss: %f' % (epoch + 1, loss_sum.item() / 11446)
                torch.save(net.state_dict(), '%d_Tele_params_all_batch65NoShffule_Original_lr1e-05.pkl' % (epoch + 6))


    # requires_grad = True
    # .reshape((743990, 2))


if __name__ == '__main__':
    test = Preprocess()
    test.rows2tensor(test.rows_train)
