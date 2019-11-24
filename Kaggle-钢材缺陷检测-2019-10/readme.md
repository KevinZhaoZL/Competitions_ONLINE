## 钢材缺陷检测

[比赛链接]([https://www.kaggle.com/c/severstal-steel-defect-detection/](https://www.kaggle.com/c/severstal-steel-defect-detection/)

---

### 1. 目标

&emsp;&emsp;对钢材中存在的缺陷进行分割并分类

### 2. 思路

* 将CSV格式存储的标注数据转化为Pascal PNG格式，使用DeepLabV3+模型进行训练，主干网络选用ResNet-50
* 第一次Kaggle试水，mIou为88.314%，虽然第一名90%左右，但中间还是差了1400多名，真.一分甩开无数人，md，继续加油~~
