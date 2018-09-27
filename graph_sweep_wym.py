#画图，时间图
#我们提出的方法

import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
# 对输入数据LOG处理
def datalog(data):
    for i in range(len(data)):
        data[i]=np.log10(data[i])
    return data
# 对时间进行归一化（1~0）
# def normalizedata(y):
#     y1=[]
#     for i in y:
#         i = float(i - np.min(y)) / (np.max(y) - np.min(y))
#         y1.append(i)
#     return y1

# # 数据获取
# # 可以更改文件来读取其他数据，只要文件满足格式
# DataPath='E:/hpc/loaddata.txt'
# data_old=np.loadtxt(DataPath)
# data=data_old[data_old[:,-1]>=10,:]
# num=351
# # 计算第6到第15位模块数之和
# for i in range(7,16):
#     data[:,6]+=data[:,i]
# # 取特征
# X=data[:num,[1,2,3,4,5,6]]

#数据获取2 ，将数据分为两部分：训练数据(70%=1230)和测试数据(30%=537)
DataPath='E:/hpc/th/test1.txt'
data=np.loadtxt(DataPath)
print(data.shape)
num=861
# data=sweep.X_pre
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
X_scale =datalog(X)
y=data[:num,16]#y是真实运行时间
y=sorted(y)
y=np.array(y)
# y1=normalizedata(y)

#SVR
clf=joblib.load("E:\\hpc\\th\\clf1.pkl")
y_svr=clf.predict(X_scale)
y_svr=pow(10,y_svr)  #y_pre是预测执行时间
y_pre1=sorted(y_svr)
# y2=normalizedata(y_pre1)
# y_pre1=np.array(y_pre1)
err1=abs(y_pre1-y)/y
print(err1.shape)
err1_mean=np.mean(err1)
print("svr:",err1_mean)
#RF
regr=joblib.load("E:\\hpc\\th\\regr1.pkl")
y_rf=regr.predict(X_scale)
y_rf=pow(10,y_rf)
y_pre2=sorted(y_rf)
# y3=normalizedata(y_pre2)
# y_pre2=np.array(y_pre2)
err2=abs(y_pre2-y)/y
err2_mean=np.mean(err2)
print("rf:",err2_mean)
#ridge
ridge=joblib.load("E:\\hpc\\th\\ridge1.pkl")
y_rd=ridge.predict(X_scale)
y_rd=pow(10,y_rd)
y_pre3=sorted(y_rd)
# y4=normalizedata(y_pre3)
# y_pre3=np.array(y_pre3)
err3=abs(y_pre3-y)/y
err3_mean=np.mean(err3)
print("ridge:",err3_mean)
#画图
#刻度设置,Title为标题。Axis为坐标轴，Label为坐标轴标注。Tick为刻度线，Tick Label为刻度注释。：
plt.tick_params(labelsize=20,width=2)
#添加网格
plt.grid(True, linestyle='-.')
#画图
A,=plt.plot(np.arange(len(y)), y, 'r.', label='Actual Time')
B,=plt.plot(np.arange(len(y)), y_pre1, 'c+', label='Predicted Time (SVR)')
C,=plt.plot(np.arange(len(y)), y_pre2, 'g+', label='Predicted Time (RF)')
D,=plt.plot(np.arange(len(y)), y_pre3, 'b+', label='Predicted Time (Ridge)')
# 标题
plt.title('Actual and Predicted Runtime',size=20)
#X/Y轴标签
plt.xlabel('Testing Samples',size=20)
plt.ylabel('Runtime',size=20)
font1 = {'family' : 'Times New Roman','weight' : 'normal','size' : 20,}
plt.legend(handles=[A,B,C,D],prop=font1)
plt.show()

#刻度设置,Title为标题。Axis为坐标轴，Label为坐标轴标注。Tick为刻度线，Tick Label为刻度注释。
plt.tick_params(labelsize=20,width=2)
#添加网格
plt.grid(True, linestyle='-.')
#画图
E,=plt.plot(np.arange(len(y)), err1, 'c.', label='SVR error rate')
F,=plt.plot(np.arange(len(y)), err2, 'g.', label='RF error rate')
G,=plt.plot(np.arange(len(y)), err3, 'b.', label='Ridge error rate')
plt.title('Predicted Runtime Error Rate',size=20)
#X/Y轴标签
plt.xlabel('Testing Samples',size=20)
plt.ylabel('Error Rate',size=20)
font1 = {'family' : 'Times New Roman','weight' : 'normal','size' : 20,}
plt.legend(handles=[E,F,G],prop=font1)
plt.show()
