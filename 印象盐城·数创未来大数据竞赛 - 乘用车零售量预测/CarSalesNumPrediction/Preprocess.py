import pandas as pd
from numpy import *
from sklearn.model_selection import train_test_split


class PrePro:
    def __init__(self):
        self.dir = 'data/'
        self.train = pd.read_csv(self.dir + '[new] yancheng_train_20171226.csv', engine='python')
        self.test = pd.read_csv(self.dir + 'yancheng_testA_20171225.csv', engine='python')

    def process_train(self):
        self.train = self.train.iloc[:, 0:3]
        self.train['sale_date'] = self.train['sale_date'].astype('str')
        self.train['class_id'] = self.train['class_id'].astype('float64')
        self.train['sale_quantity'] = self.train['sale_quantity'].astype('float64')
        return self.train.as_matrix(columns=None)

    def generate(self,matrix):
        for i in range(matrix.shape[0]):
            temp=matrix[i][0]
            matrix[i][0]=temp[-2:]
        matrix[:,0]=matrix[:,0].astype('float64')
        traindata=matrix[:,0:2]
        labelVector=matrix[:,-1]
        feature_train, feature_test, label_train, label_test = train_test_split(traindata, labelVector, test_size=0.3, random_state=0)
        return feature_train, feature_test, label_train, label_test,traindata,labelVector


