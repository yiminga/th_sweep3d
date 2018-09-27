# -*- coding: UTF-8 -*-
# 该文件采用获取到的进程数和规模进行数据训练
# 传统的方法

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn import linear_model

#提取完特征后，对数据进行log缩放处理
def datalog(data):
    for i in range(len(data)):
        data[i]=np.log10(data[i])
    return data
# 数据获取
# 可以更改文件来读取其他数据，只要文件满足格式
DataPath='E:/hpc/th/train2.txt'
data=np.loadtxt(DataPath)
print(data.shape)
num=750
np.random.shuffle(data)
# 预留评估数据
#取特征
X = data[:num,[0,1,2,3]]
X_scale=datalog(X)
# 标签
y=data[:num,4]
y=datalog(y)
# svr训练
# 参数与svm.SVR()函数相同，模型的超参数
# 内部采用10折交叉验证，并将模型的评分存储在对应的score中
def svrTest(c=13,epsilon=0.2,kernel='linear',gamma='auto' ):
    global num
    # 将数据分为10份，每次取一份做测试集，其他做训练集
    for i in range(0,10):
        min=i*(int(num/10))
        max=i*(int(num/10))+int(num/10)
        X_train=np.delete(X_scale,np.s_[min:max],axis=0)
        y_train=np.delete(y,np.s_[min:max],axis=0)
        clf=svm.SVR(C=c,epsilon=epsilon,kernel=kernel,gamma=gamma)
        clf.fit(X_train,y_train)
    print (clf)
    joblib.dump(clf,'E:\\hpc\\th\\clf2.pkl')
# rf训练
# 参数与RandomForestRegressor()函数相同，模型的超参数
# 内部采用10折交叉验证，并将模型的评分存储在对应的score中
def randomForestTest(n_estimators=10,max_features='auto',max_depth=None,min_samples_split=2,min_samples_leaf=1,oob_score=False,random_state=0):

    global num
    for i in range(0,10):
        min = i * (int(num / 10))
        max = i * (int(num / 10)) + int(num / 10)
        X_train=np.delete(X_scale,np.s_[min:max],axis=0)
        y_train=np.delete(y,np.s_[min:max],axis=0)
        regr = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,oob_score=oob_score,random_state=0)
        regr.fit(X_train,y_train)
    joblib.dump(regr, 'E:\\hpc\\th\\regr2.pkl')
    print(regr)

def ridegTest(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, random_state=None, solver='auto', tol=0.001):
    global num
    # 将数据分为10份，每次取一份做测试集，其他做训练集
    for i in range(0, 10):
        min = i * (int(num / 10))
        max = i * (int(num / 10)) + int(num / 10)
        X_train = np.delete(X_scale, np.s_[min:max], axis=0)
        y_train = np.delete(y, np.s_[min:max], axis=0)
        clf =linear_model.Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
        clf.fit(X_train, y_train)
    print(clf)
    joblib.dump(clf, 'E:\\hpc\\th\\ridge2.pkl')

svrTest(c=500,epsilon=0,kernel='linear')
randomForestTest(n_estimators=4,max_features=4,min_samples_split=2,min_samples_leaf=2)
ridegTest(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, random_state=None, solver='auto', tol=0.001)
