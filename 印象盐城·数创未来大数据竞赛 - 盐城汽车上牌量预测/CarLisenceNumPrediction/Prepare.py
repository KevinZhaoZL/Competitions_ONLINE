# coding:utf-8

import pandas as pd
import matplotlib.pyplot as plt

dir = 'data/'
train = pd.read_table(dir + 'train_20171215.txt', engine='python')
test_A = pd.read_table(dir + 'test_A_20171225.txt', engine='python')
sample_A = pd.read_table(dir + 'sample_A_20171225.txt', engine='python')

# print(train.info())
# print(test_A.info())
# print(sample_A.info())

# print(train['day_of_week'].unique())
# print(test_A['day_of_week'].unique())

# 箱型图检测异常值，分布大致情况(左偏，右偏，对称，U形)
# plt.boxplot(train['cnt'])
# plt.show()

# 绘制分布图
'''
import seaborn as sns
color=sns.color_palette()
sns.set_style('darkgrid')
from scipy import stats
from scipy.stats import norm,skew
sns.distplot(train['cnt'],fit=norm)
plt.show()
'''

# date和cnt的关系
# plt.plot(train['date'],train['cnt'])
# plt.show()

# 目标数据的描述
'''
print(train['cnt'].describe())
from sklearn.metrics import mean_squared_error
train['25%']=221
train['50%']=351
train['75%']=496
train['median']=train['cnt'].median()
train['mean']=train['cnt'].mean()
print(mean_squared_error(train['cnt'],train['25%']))
print(mean_squared_error(train['cnt'],train['50%']))
print(mean_squared_error(train['cnt'],train['75%']))
print(mean_squared_error(train['cnt'],train['median']))
print(mean_squared_error(train['cnt'],train['mean']))
'''

# 按星期几统计
# monday = train[train['day_of_week']==1]
# plt.plot(range(len(monday)),monday['cnt'])
# plt.show()

# 按星期评分
'''
res = train.groupby(['day_of_week'],as_index=False).cnt.mean()
xx = train.merge(res,on=['day_of_week'])
print(xx.head())
from sklearn.metrics import mean_squared_error
print(mean_squared_error(xx['cnt_x'],xx['cnt_y']))
'''

# 合并品牌
# 因为第一赛季只是预测与时间相关的cnt的数量
# 所以可以对数据以dat和dow进行数据合并
train = train.groupby(['date', 'day_of_week'], as_index=False).cnt.sum()
plt.plot(train['date'], train['cnt'], '*')
plt.show()#

#dw和date
'''
for i in range(7):
    tmp = train[train['day_of_week']==i+1]
    plt.subplot(7, 1, i+1)
    plt.plot(tmp['date'],tmp['cnt'],'*')
plt.show()
'''
