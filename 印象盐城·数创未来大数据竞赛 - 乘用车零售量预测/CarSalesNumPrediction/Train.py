from Preprocess import *
from numpy import *
from scipy import linalg

class Train:
    def __init__(self):
        self.prepro = PrePro()
        self.matrix_complete = self.prepro.process_train()
        self.feature_train, self.feature_test, self.label_train, self.label_test, self.traindata, self.labelVector \
            = self.prepro.generate(self.matrix_complete)

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
        temp_d=zeros([2,2])
        temp_d[0][0]=xTx[0].tolist()[0][0]
        temp_d[0][1]=xTx[0].tolist()[0][1]
        temp_d[1][0]=xTx[1].tolist()[0][0]
        temp_d[1][1]=xTx[1].tolist()[0][1]
        xTx=temp_d
        xTx=mat(xTx)
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

if __name__ == '__main__':
    trainer = Train()
    temp=0
    y_hat = trainer.lwlrTest(trainer.feature_test, trainer.feature_train, trainer.label_train, k=0.3)
    for i in range(len(y_hat)):
        temp=temp+(y_hat[i]-trainer.label_test[i])**2
    score=temp/(len(y_hat))
    score=math.sqrt(score)
    print(score)
    #print(trainer.label_test,y_hat)
