## 肝癌影像分类任务
[比赛链接](https://www.datafountain.cn/competitions/335/details/rule)
*****
### 1. 目标
&emsp;&emsp;对肝癌病人的多张肝部医学影像进行分类，良性 or  恶性
### 2. 思路
* 将同一病人的多张肝部影像合并，每幅图像权值相同，在resnet，resnetv2，resenet和seresnet上进行分类，其中v2，得到了0.7的Score，初赛29/1397
* 在上基础上，对病人的没一张图象进行打分，即不融合当前图片的情况下分数变化，选出下降最多的十张，融合后，在已有模型上进行训练，或者用预训练模型进行训练，最好效果也是0.7
****
[pytorch 模型](https://pan.baidu.com/s/1WbJmG9SFlI1vdUju6s_soQ)    提取码: dkfe 
