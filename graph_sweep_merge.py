#画图，时间图
#将两种方法合并

import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
# 对输入数据LOG处理
def datalog(data):
    for i in range(len(data)):
        data[i]=np.log10(data[i])
    return data

#获取测试数据
DataPath='E:/hpc/th/test3.txt'
data=np.loadtxt(DataPath)
num=861
# 取特征
X=data[:num,[0,1,2,3,4,6,16,17,18]]
# 缩小
X_scale =datalog(X)
#y是真实运行时间
y=data[:num,19]
y=sorted(y)
y=np.array(y)
#SVR
clf=joblib.load("E:\\hpc\\th\\clf3.pkl")
print (clf)
y_svr=clf.predict(X_scale)
y_svr=pow(10,y_svr)  #y_pre是预测执行时间
y_pre1=sorted(y_svr)
err1=abs(y_pre1-y)/y
print(err1.shape)
err1_mean=np.mean(err1)
print("svr:",err1_mean)
#RF
regr=joblib.load("E:\\hpc\\th\\regr3.pkl")
y_rf=regr.predict(X_scale)
y_rf=pow(10,y_rf)
y_pre2=sorted(y_rf)
err2=abs(y_pre2-y)/y
err2_mean=np.mean(err2)
print("rf:",err2_mean)
#ridge
ridge=joblib.load("E:\\hpc\\th\\ridge3.pkl")
y_rd=ridge.predict(X_scale)
y_rd=pow(10,y_rd)
y_pre3=sorted(y_rd)
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
