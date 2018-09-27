# -*- coding: UTF-8 -*-
# 该文件用于数据训练
# 更改num可以选择不同的数据规模
# 更改DataPath可以读取不同的数据，包括top-15-master、NPB、sweep3d


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from sklearn import linear_model

#提取完特征后，对数据进行log缩放处理
def datalog(data):
    for i in range(len(data)):
        data[i]=np.log10(data[i])
    return data

# 数据获取
# 可以更改文件来读取其他数据，只要文件满足格式
DataPath='E:/hpc/th/train1.txt'
data_old=np.loadtxt(DataPath)
#过滤掉时间小于10s的数据
# 还剩1767组数据
# data=data_old[data_old[:,-1]>=10,:]
data=data_old
print(data.shape)
# 使用的数据总数
# 因为要进行交叉验证，取10的整数倍比较方便
#取70%的数据构建模型
num=750
# 将数据打乱顺序
np.random.shuffle(data)
# 预留评估数据
# global X_pre
# X_pre=data[num+1:]
# 计算第6到第15位模块数之和
# for i in range(7,16):
#     data[:,6]+=data[:,i]
# 取特征
X=data[:num,[0,1,2,3,4,5,6]]
# 添加p*x作为新特征
# p--进程数 x--对应模块数
# X=np.append(X,X[:,0:1]*data[:num,0:1],axis=1)
# X=np.append(X,X[:,1:2]*data[:num,0:1],axis=1)
# X=np.append(X,X[:,2:3]*data[:num,0:1],axis=1)
# X=np.append(X,X[:,3:4]*data[:num,0:1],axis=1)
# X=np.append(X,X[:,4:5]*data[:num,0:1],axis=1)
# X=np.append(X,X[:,5:6]*data[:num,0:1],axis=1)
# 缩小
X_scale = datalog(X)
# print (X_scale)
# 标签
y=data[:num,16]
#缩小
y = datalog(y)

# 用于泛化误差图像的横坐标
X_show=np.arange(1,101,10)
# 存储评价SVR模型好坏的相关评分，越靠近后面的数字表示拟合效果越好
# 目前只用到score2和score3，其他未用
score1_svr=np.array([])#explained_variance_score，1
score2_svr=np.array([])#mean_absolute_error，0
score3_svr=np.array([])#mean_squared_error，0
score4_svr=np.array([])#median_absolute_error，0
score5_svr=np.array([])#r2_score，1
#同上，用于存储随机森林模型的评分
score1_rf=np.array([])
score2_rf=np.array([])
score3_rf=np.array([])
score4_rf=np.array([])
score5_rf=np.array([])

# svr训练
# 参数与svm.SVR()函数相同，模型的超参数
# 内部采用10折交叉验证，并将模型的评分存储在对应的score中
def svrTest(c=1,epsilon=0,kernel='linear',gamma='auto' ):
    global score1_svr
    global score2_svr
    global score3_svr
    global score4_svr
    global score5_svr
    global num
    score=np.zeros(5)
    # 将数据分为10份，每次取一份做测试集，其他做训练集
    for i in range(0,10):
        min=i*(int(num/10))
        max=i*(int(num/10))+int(num/10)
        X_train=np.delete(X_scale,np.s_[min:max],axis=0)
        X_test=X_scale[min:max,:]
        y_train=np.delete(y,np.s_[min:max],axis=0)
        y_test=y[min:max]
        # print(X_train,X_test,y_train,y_test)
        clf=svm.SVR(C=c,epsilon=epsilon,kernel=kernel,gamma=gamma)
        clf.fit(X_train,y_train)
        y_pre=clf.predict(X_test)
        # print(y_test,y_pre)
        # score[0]+=explained_variance_score(y_test,y_pre)#best is 1
        score[1]+=mean_absolute_error(y_test,y_pre)#best is 0
        score[2]+=mean_squared_error(y_test,y_pre)#best is 0

        # 这两行可以用来表示训练误差，即模型对训练集的拟合情况
        # score[1] += mean_absolute_error(y_train, clf.predict(X_train))
        # score[2] += mean_squared_error(y_train, clf.predict(X_train))

        # score[3]+=median_absolute_error(y_test,y_pre)#best is 0
        # score[4]+=r2_score(y_test,y_pre)#best is 1
        # s1=pickle.dumps(clf)#模型持久化（方法一）
    print (clf)
    joblib.dump(clf,'E:\\hpc\\th\\clf1.pkl')
    # 取10次训练结果误差的均值
    score/=10
    print(score[1],score[2])
    # score1_svr=np.append(score1_svr,score[0])
    score2_svr = np.append(score2_svr, score[1])
    score3_svr = np.append(score3_svr, score[2])
    # score4_svr = np.append(score4_svr, score[3])
    # score5_svr = np.append(score5_svr, score[4])
    #return s1 #模型持久化方法一返回
# rf训练
# 参数与RandomForestRegressor()函数相同，模型的超参数
# 内部采用10折交叉验证，并将模型的评分存储在对应的score中
def randomForestTest(n_estimators=10,max_features='auto',max_depth=None,min_samples_split=2,min_samples_leaf=1,oob_score=False,random_state=0):
    global score1_rf
    global score2_rf
    global score3_rf
    global score4_rf
    global score5_rf
    global num
    score=np.zeros(5)
    for i in range(0,10):
        min = i * (int(num / 10))
        max = i * (int(num / 10)) + int(num / 10)
        X_train=np.delete(X_scale,np.s_[min:max],axis=0)
        X_test=X_scale[min:max,:]
        y_train=np.delete(y,np.s_[min:max],axis=0)
        y_test=y[min:max]
        # print(X_train,X_test,y_train,y_test)
        regr = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,oob_score=oob_score,random_state=0)
        regr.fit(X_train,y_train)
        y_pre=regr.predict(X_test)
        # score[0]+=explained_variance_score(y_test,y_pre)#best is 1
        score[1]+=mean_absolute_error(y_test,y_pre)#best is 0
        score[2]+=mean_squared_error(y_test,y_pre)#best is 0
        # score[3]+=median_absolute_error(y_test,y_pre)#best is 0
        # score[4]+=r2_score(y_test,y_pre)#best is 1
        #s2 = pickle.dumps(regr) #模型持久化
    joblib.dump(regr, 'E:\\hpc\\th\\regr1.pkl')
    print(regr)
    score/=10
    print(score[1], score[2])
    # score1_rf = np.append(score1_rf,score[0])
    score2_rf = np.append(score2_rf, score[1])
    score3_rf = np.append(score3_rf, score[2])
    # score4_rf = np.append(score4_rf, score[3])
    # score5_rf = np.append(score5_rf, score[4])
    #return s2

#岭回归
def ridegTest(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, random_state=None, solver='auto', tol=0.001):
    global score1_svr
    global score2_svr
    global score3_svr
    global score4_svr
    global score5_svr
    global num
    score = np.zeros(5)
    # 将数据分为10份，每次取一份做测试集，其他做训练集
    for i in range(0, 10):
        min = i * (int(num / 10))
        max = i * (int(num / 10)) + int(num / 10)
        X_train = np.delete(X_scale, np.s_[min:max], axis=0)
        X_test = X_scale[min:max, :]
        y_train = np.delete(y, np.s_[min:max], axis=0)
        y_test = y[min:max]
        # print(X_train,X_test,y_train,y_test)
        clf = linear_model.Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
        clf.fit(X_train, y_train)
        y_pre = clf.predict(X_test)

        score[1] += mean_absolute_error(y_test, y_pre)  # best is 0
        score[2] += mean_squared_error(y_test, y_pre)  # best is 0

    print(clf)
    joblib.dump(clf, 'E:\\hpc\\th\\ridge1.pkl')

    score /= 10
    print(score[1], score[2])
    score2_svr = np.append(score2_svr, score[1])
    score3_svr = np.append(score3_svr, score[2])

# 显示svr模型的误差与超参数变化的图像
def svrShow():
    plt.subplot(221)
    plt.title('score2:0')
    plt.plot(X_show, score2_svr, 'blue')
    plt.subplot(222)
    plt.title('score3:0')
    plt.plot(X_show, score3_svr, 'green')
    # plt.subplot(223)
    # plt.title('score4:0')
    # plt.plot(X_show, score4_svr, 'green')
    # plt.subplot(224)
    # plt.title('score5:0')
    # plt.plot(X_show, score5_svr, 'green')
    plt.show()

# 显示随机森林模型的误差与超参数变化的图像
def rfShow():
    plt.subplot(121)
    plt.title('score2:0')
    plt.plot(X_show, score2_rf, 'blue')
    plt.subplot(122)
    plt.title('score3:0')
    plt.plot(X_show, score3_rf, 'green')
    plt.show()

# 进行超参数的选择，也可使用网格搜索等其他方法
# 遍历范围大小需和X_show大小相同
# for i in range(1,13):
#     randomForestTest(n_estimators=i,max_features=8,min_samples_split=2,min_samples_leaf=2)
# rfShow()
# for i in range(1,101,10):
#     svrTest(c=i ,epsilon=0,kernel='linear')
# svrShow()

# #  wym改进后的sweep3d
svrTest(c=1,epsilon=0,kernel='linear')
ridegTest(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, random_state=None, solver='auto', tol=0.001)
randomForestTest(n_estimators=9,max_features=7,min_samples_split=2,min_samples_leaf=2)


# top15 1 未添加大规模的测试集（最开始的那个top-15-master）
# svrTest(c=33,epsilon=0.01) 1.60 27.7 rbf
# svrTest(c=0.2,epsilon=1.3) 1.02 1.43 linear
# svrTest(c=3.5,epsilon=0.9) 2.10 9.82 poly
# svrTest(c=1.1,epsilon=0.9) 1.74 19.01 sigmoid
# svrTest(c=0.29,epsilon=0.4) 1.06 1.63 linearSVR 'epsilon_insensitive'
# svrTest(c=0.29,epsilon=0.4) 1.06 1.63 linearSVR 'squared_epsilon_insensitive'
# randomForestTest(n_estimators=9,max_features=2,min_samples_split=2,min_samples_leaf=1,oob_score=False) 1.40 13.40
# randomForestTest(n_estimators=1,max_features=2,min_samples_split=2,min_samples_leaf=1,oob_score=True) 1.38 12.4

# sweep3d 改进后
# svrTest(c=13,epsilon=0.2,kernel='linear')1.96 14.83
# randomForestTest(n_estimators=4,max_features=11,min_samples_split=2,min_samples_leaf=2)2.01 17.91
# randomForestTest(n_estimators=19,max_features=8,min_samples_split=2,min_samples_leaf=2,oob_score=True)2.04 16.78

# sweep3d 未改进
# svrTest(c=200,epsilon=0.1,gamma=0.06)2.12 21.28 rbf
# svrTest(c=3,epsilon=20,kernel='linear')9.65 174.14
# svrTest(c=0.7,epsilon=1.5,gamma=0.2,kernel='sigmoid')14.26 359.54
# svrTest(c=20,epsilon=13,kernel='poly')22.03 1132.53
# randomForestTest(n_estimators=10,max_features=6,min_samples_leaf=2,min_samples_split=2)5.43 65.51 false
# randomForestTest(n_estimators=10,max_features=6,min_samples_split=4,min_samples_leaf=1,oob_score=True)4.63 71.82

# NPB数据
# svrTest(c=54,epsilon=0.75,gamma=0.002) 0.995 2.34 rbf
# svrTest(c=0.4,epsilon=0.4,kernel='linear') 0.92 2.22
# svrTest(c=1.5,epsilon=0.1,kernel='sigmoid',gamma=0.08) 2.15 38.67
# randomForestTest(n_estimators=18,max_features=1,min_samples_leaf=2,min_samples_split=2)1.52 29.25
# randomForestTest(n_estimators=8,max_features=2,min_samples_split=2,min_samples_leaf=1,oob_score=False)1.22 14.92

# top15 2 将大规模的数据加入进行训练（top-15-master 2）
# svrTest(c=20,epsilon=2.5) rbf 2.66 37.07
# svrTest(c=5,epsilon=5,kernel='sigmoid') 6.23 92.16
# svrTest(c=6,epsilon=7,kernel='poly')8.37 123.81
# randomForestTest(n_estimators=7,max_features=3,min_samples_split=3,min_samples_leaf=2,oob_score=True)2.71 37.24
# oobscore false 2.17 25.84
