from scipy import linalg
from Preprocessor import Preprocessor
from numpy import *
from sklearn.metrics import mean_squared_error
import pandas as pd


class Train:
    def __init__(self):
        self.prepro = Preprocessor()
        self.t_weekday_all, self.t_weekday_train, self.t_weekday_test, self.t_weekend_all, self.t_weekend_train, self.t_weekend_test = self.prepro.process_pre1()

    def lwlr(self, testPoint, xArr, yArr, k):
        # xMat = np.mat(xArr)
        # yMat = np.mat(yArr).T
        xMat = xArr
        yMat = mat(yArr).T
        m = shape(xMat)[0]
        weights = mat(eye((m)))
        for j in range(m):
            diffMat = testPoint - xMat[j, :]
            diffMat = mat(diffMat)
            a = diffMat * diffMat.T / (-2.0 * k ** 2)
            weights[j, j] = math.exp(a)
        xTx = xMat.T * (weights * xMat)
        if linalg.det(xTx) == 0.0:
            print("This matrix is singular, cannot do inverse")
            return
        ws = xTx.I * (xMat.T * (weights * yMat))
        return testPoint * ws

    def lwlrTest(self, testArr, xArr, yArr, k=1.0):  # loops over all the data points and applies lwlr to each one
        m = shape(testArr)[0]
        yHat = zeros(m)
        for i in range(m):
            yHat[i] = self.lwlr(testArr[i], xArr, yArr, k)
        return yHat

    def predict_testA_pre(self):
        prero = Preprocessor()
        sample = prero.test_A
        weekend = [6, 7]
        test_weekday = pd.DataFrame(columns=['date', 'day_of_week'])
        test_weekend = pd.DataFrame(columns=['date', 'day_of_week'])

        for i in range(len(sample['date'])):
            sample['date'][i] = sample['date'][i] % 365
            if sample['date'][i] == 0:
                sample['date'][i] = 365
            a = sample['day_of_week'][i]
            if a in weekend:
                new = pd.DataFrame({"date": sample['date'][i], "day_of_week": sample['day_of_week'][i]}, index=["0"])
                test_weekend = test_weekend.append(new, ignore_index=True)
                test_weekend['date'] = test_weekend['date'].astype('float64')
                test_weekend['day_of_week'] = test_weekend['day_of_week'].astype('float64')
            else:
                new = pd.DataFrame({"date": sample['date'][i], "day_of_week": sample['day_of_week'][i]}, index=["0"])
                test_weekday = test_weekday.append(new, ignore_index=True)
                test_weekday['date'] = test_weekday['date'].astype('float64')
                test_weekday['day_of_week'] = test_weekday['day_of_week'].astype('float64')
        return test_weekday.as_matrix(columns=None), test_weekend.as_matrix(columns=None)

    def predict_A_weekday(self, test_xMat):
        y_weekday = zeros((test_xMat.shape[0], 2))
        for i in range(test_xMat.shape[0]):
            y_weekday[i][0] = int(test_xMat[i][0])
        trainer = Train()
        pre = Preprocessor()
        predictVector, xMat = pre.process_pre2(trainer.t_weekday_all)
        for i in range(xMat.shape[0]):
            xMat[i, 0] = xMat[i, 0] / 365
        for j in range(test_xMat.shape[0]):
            test_xMat[j, 0] = test_xMat[j, 0] / 365
        y_hat_weekday = trainer.lwlrTest(test_xMat, xMat, predictVector, k=0.3)
        for i in range(test_xMat.shape[0]):
            y_weekday[i][1] = int(y_hat_weekday[i])
        f_weekday = open('data/y_hat_weekday.txt', 'w+')
        for i in range(y_weekday.shape[0]):
            f_weekday.write(str(y_weekday[i][0]))
            f_weekday.write('\t')
            f_weekday.write(str(y_weekday[i][1]))
            f_weekday.write('\n')
        f_weekday.close()

    def predict_A_weekend(self, test_xMat):
        y_weekend = zeros((test_xMat.shape[0], 2))
        for i in range(test_xMat.shape[0]):
            y_weekend[i][0] = int(test_xMat[i][0])
        trainer = Train()
        pre = Preprocessor()
        predictVector, xMat = pre.process_pre2(trainer.t_weekend_all)
        for i in range(xMat.shape[0]):
            xMat[i, 0] = xMat[i, 0] / 365
        for j in range(test_xMat.shape[0]):
            test_xMat[j, 0] = test_xMat[j, 0] / 365
        y_hat_weekend = trainer.lwlrTest(test_xMat, xMat, predictVector, k=0.3)
        for i in range(test_xMat.shape[0]):
            y_weekend[i][1] = int(y_hat_weekend[i])
        f_weekend = open('data/y_hat_weekend.txt', 'w+')
        for i in range(y_weekend.shape[0]):
            f_weekend.write(str(y_weekend[i][0]))
            f_weekend.write('\t')
            f_weekend.write(str(y_weekend[i][1]))
            f_weekend.write('\n')
        f_weekend.close()


if __name__ == '__main__':
    p = Train()
    test_weekday_mat, test_weekend_mat = p.predict_testA_pre()
    p.predict_A_weekday(test_weekday_mat)
    p.predict_A_weekend(test_weekend_mat)
    '''
    trainer = Train()
    pre = Preprocessor()
    predictVector, xMat = pre.process_pre2(trainer.t_weekend_train)
    test_predictVector, test_xMat = pre.process_pre2(trainer.t_weekend_test)
    for i in range(xMat.shape[0]):
        xMat[i,0]=xMat[i,0]/365
    for j in range(test_xMat.shape[0]):
        test_xMat[j,0]=test_xMat[j,0]/365
    y_hat = trainer.lwlrTest(test_xMat, xMat, predictVector, k=0.3)
    print(mean_squared_error(test_predictVector,y_hat))
    '''
