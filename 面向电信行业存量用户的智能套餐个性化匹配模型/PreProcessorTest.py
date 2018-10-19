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

CUDA_LAUNCH_BLOCKING = 1

net = torchvision.models.resnet50(pretrained=True)
for param in net.parameters():
    param.requires_grad = False

net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
net.fc = nn.Linear(8192, 11)

net.load_state_dict(torch.load('8_Tele_params_all_batch65NoShffule_Original_lr1e-05.pkl'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 第一行代码
net = net.cuda()
print net
txtName = "8_result.txt"
f = file(txtName, "a+")


class Preprocess():
    testData_dir = '/home/yuanye/ubuntu-windows/data/toubuntudektop/republish_test_tmp_tmp.csv'
    trainData_dir = '/home/yuanye/ubuntu-windows/data/toubuntudektop/train_all.csv'
    rows_train = pd.read_csv(trainData_dir)
    rows_test = pd.read_csv(testData_dir)

    def getBasicInfo(self, rows=rows_train):
        # field names
        fieldNames = list(rows.columns.values)
        # shape
        row_num = len(rows['service_type'])
        column_num = len(list(rows.columns.values))
        return fieldNames, row_num, column_num

    def rows2tensor(self, rows=rows_test):
        fieldNames, row_num, column_num = self.getBasicInfo(rows=rows)
        returnMat = np.zeros((row_num, column_num - 1))

        index = 0
        for i in range(column_num - 1):
            tmp = list(rows[fieldNames[i]])
            for j in range(len(tmp)):
                if tmp[j] == '\N' or tmp[j] == '#VALUE!':
                    tmp[j] = 0
                if tmp[j] == 'nan':
                    tmp[j] = 255
                tmp[j] = float(tmp[j])
            returnMat[:, index] = tmp[0:]
            index = index + 1

        retMat = np.zeros((1, 1, 225, 225))
        for i in range(200000):
            print str(i) + ' begin'
            # retTrainData.append((returnMat[i].repeat(2025).reshape((1,225,225)), returnLabelVector[i]))
            # retTrainData.append(np.tile(returnMat[i],2025).reshape((1, 225, 225)))
            retMat[0] = np.tile(returnMat[i], 2025).reshape((1, 1, 225, 225))
            temp = torch.from_numpy(retMat)
            feature = temp.type(torch.FloatTensor)
            feature_v = Variable(feature)
            feature_v = feature_v.to(device)

            out = net(feature_v)  # input x and predict based on x
            prediction = torch.max(F.softmax(out), 1)[1]
            print i, prediction.item()

            if prediction.item() == 0:
                tmp = 99999830
            elif prediction.item() == 1:
                tmp = 90063345
            elif prediction.item() == 2:
                tmp = 90155946
            elif prediction.item() == 3:
                tmp = 99999825
            elif prediction.item() == 4:
                tmp = 99999826
            elif prediction.item() == 5:
                tmp = 99999827
            elif prediction.item() == 6:
                tmp = 99999828
            elif prediction.item() == 7:
                tmp = 89950166
            elif prediction.item() == 8:
                tmp = 89950167
            elif prediction.item() == 9:
                tmp = 89950168
            else:
                tmp = 90109916
            f.write(str(tmp) + '\n')

            print 'end'

        f.close()

    # requires_grad = True
    # .reshape((743990, 2))


if __name__ == '__main__':
    test = Preprocess()
    print 'XXXX'
    test.rows2tensor(test.rows_test)
