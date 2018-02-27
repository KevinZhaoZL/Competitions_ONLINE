import numpy as np
from numpy import *

c=np.array([[100,200],[300,400]])
c=mat(c)
d=zeros([2,2])
d[0][0]=c[0].tolist()[0][0]
d[0][1]=c[0].tolist()[0][1]
d[1][0]=c[1].tolist()[0][0]
d[1][1]=c[1].tolist()[0][1]
d=mat(d)
print(linalg.det(c))
print(linalg.det(d))