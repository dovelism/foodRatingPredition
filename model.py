#!/usr/bin/python
# encoding:utf-8
'''
Author : Dovelism
Email:18222599593@163.com
Creat Time: 2019/11/29 15:33
Content: A simple demo for xgboost
'''
from __future__ import print_function
from sklearn.model_selection import train_test_split

from xgboost.sklearn import XGBClassifier
from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

feature = []
label = []
#fr = open('testData.csv')
fr = open('food_ratings.txt')
lines = fr.readlines()

for line in lines:
    lineArr = line.strip().split(',')
    feature.append([float(i) for i in lineArr[:3]])
    label.append(lineArr[-1])

feature = np.array(feature)
label = np.array(label)


# 加载样本数据集
X,y = feature,label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234) # 数据集分割

# 训练模型
model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=16, silent=True)
model.fit(X_train, y_train)

# 对测试集进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test,y_pred)
print("accuarcy: %.2f%%" % (accuracy*100.0))

# 显示重要特征
plot_importance(model)
plt.show()