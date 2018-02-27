import pandas as pd
import matplotlib.pyplot as plt


class Preprocessor:
    def __init__(self):
        self.dir = 'data/'
        self.train = pd.read_table(self.dir + 'train_20171215.txt', engine='python')
        self.test_A = pd.read_table(self.dir + 'test_A_20171225.txt', engine='python')
        self.sample_A = pd.read_table(self.dir + 'sample_A_20171225.txt', engine='python')

    '''
    数据初步处理函数
    1.合并date和day_of_week
    2.date对365求余，使date成为1-365的序列
    3.将周一到周五与周六日的数据分离
    4.分别建立训练集和测试集以方便局部加权线性回归算法的计算
    '''

    def process_pre1(self):
        self.train = self.train.groupby(['date', 'day_of_week'], as_index=False).cnt.sum()
        weekend = [6,7]
        train_weekday = pd.DataFrame(columns=['date', 'day_of_week', 'cnt'])
        train_weekend = pd.DataFrame(columns=['date', 'day_of_week', 'cnt'])

        for i in range(len(self.train['date'])):
            self.train['date'][i] = self.train['date'][i] % 365
            if self.train['date'][i] == 0:
                self.train['date'][i] = 365
            a = self.train['day_of_week'][i]
            if a in weekend:
                #删除异常数据
                if self.train['cnt'][i]<800:
                    new = pd.DataFrame({"date": self.train['date'][i], "day_of_week": self.train['day_of_week'][i],
                                    "cnt": self.train['cnt'][i]}, index=["0"])
                    train_weekend = train_weekend.append(new, ignore_index=True)
                    train_weekend['date']=train_weekend['date'].astype('float64')
                    train_weekend['day_of_week']=train_weekend['day_of_week'].astype('float64')
                    train_weekend['cnt']=train_weekend['cnt'].astype('float64')
            else:
                if self.train['cnt'][i]<3800:
                    new = pd.DataFrame({"date": self.train['date'][i], "day_of_week": self.train['day_of_week'][i],
                                    "cnt": self.train['cnt'][i]}, index=["0"])
                    train_weekday = train_weekday.append(new, ignore_index=True)
                    train_weekday['date']=train_weekday['date'].astype('float64')
                    train_weekday['day_of_week']=train_weekday['day_of_week'].astype('float64')
                    train_weekday['cnt']=train_weekday['cnt'].astype('float64')
        weekday_train = train_weekday[train_weekday.index < 600]
        weekday_test = train_weekday[train_weekday.index >= 600]
        weekend_train = train_weekend[train_weekend.index < 150]
        weekend_test = train_weekend[train_weekend.index >= 150]

        # return train_weekday, weekday_train, weekday_test, train_weekend, weekend_train, weekend_test
        return train_weekday.as_matrix(columns=None), weekday_train.as_matrix(columns=None), weekday_test.as_matrix(
            columns=None), train_weekend.as_matrix(columns=None), weekend_train.as_matrix(
            columns=None), weekend_test.as_matrix(columns=None)

    def process_pre2(self, matrix):
        predictVector = matrix[:, 0]
        returnMat = matrix[:, 1:]
        return predictVector, returnMat


if __name__ == '__main__':
    prepro = Preprocessor()
    t_wday_all, t_wday_train, t_wday_test, t_wend_all, t_wend_train, t_wend_test = prepro.process_pre1()

    plt.plot(t_wend_train['date'], t_wend_train['cnt'], 'o')
    plt.plot(t_wday_train['date'], t_wday_train['cnt'], 'x')
    plt.show()
