from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split,cross_val_score
import matplotlib.pylab as plt
%matplotlib inline
import numpy as np
#%%單層感知機
x,y=make_classification(n_samples=1000,#樣本數
                        n_features=2,#樣本特徵數
                        n_redundant=0,#冗餘資訊
                        n_informative=1,#特徵的隨機縣性組合
                        n_clusters_per_class=1,#一個類別只有一群
                        random_state=22)#隨機亂樹種子

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

positive_x1=[x[i,0] for i in range(1000) if y[i]==1]
positive_x2=[x[i,1] for i in range(1000) if y[i]==1]
negetive_x1=[x[i,0] for i in range(1000) if y[i]==0]
negetive_x2=[x[i,1] for i in range(1000) if y[i]==0]
plt.figure()
plt.scatter(positive_x1,positive_x2,c='red')
plt.scatter(negetive_x1,negetive_x2,c='blue')


from sklearn.linear_model import Perceptron
clf=Perceptron(fit_intercept=False,shuffle=False)
clf.fit(x_train,y_train)
print(clf.coef_)
acc=clf.score(x_test,y_test)
print(acc)

plt.figure()
plt.scatter(positive_x1,positive_x2,c='red')
plt.scatter(negetive_x1,negetive_x2,c='blue')
line_x=np.arange(-4,4)
line_y=line_x*(clf.coef_[0][0]/clf.coef_[0][1])+clf.intercept_
plt.plot(line_x,line_y,lw=3,c='black')

#%%神經網路
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,cross_val_score
import  matplotlib.pylab as plt

iris=load_iris()
x=iris.data[:,:2]
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

def plot_sample(ax,x,y,n_classes=3):
    plot_colors='bry'
    for i,color in zip(range(n_classes),plot_colors):
        idx=np.where(y==i)
        ax.scatter(x[idx,0],x[idx,1],c=color,label=iris.target_names[i],
                   cmap=plt.cm.Paired)

def plot_clf(ax,clf,x_min,x_max,y_min,y_max):
    step=0.02
    x,y=np.meshgrid(np.arange(x_min,x_max,step),np.arange(y_min,y_max,step))
    z=clf.predict(np.c_[x.ravel(),y.ravel()])
    z=z.reshape(x.shape)
    ax.contourf(x,y,z,cmap=plt.cm.Paired)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plot_sample(ax,x,y)

def mplnn_iris():
    plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    clf=MLPClassifier(hidden_layer_sizes=(30,),activation='relu',
                      max_iter=10000)
    clf.fit(x_train,y_train)
    train_score=clf.score(x_train,y_train)
    test_score=clf.score(x_test,y_test)
    x1_min,x1_max=min(x_train[:,0]-1),max(x_train[:,0]+1)
    x2_min,x2_max=min(x_train[:,0]-1),max(x_train[:,0]+1)
    plot_clf(ax,clf,x1_min,x1_max,x2_min,x2_max)
    plot_sample(ax,x_train,y_train)
    ax.legend(loc='best')
    ax.set_xlabel('sepal length(cm)')
    ax.set_ylabel('sepal width(cm)')
    ax.set_title('準確率:訓練集%f,測試集%f'%(train_score,test_score))

mplnn_iris()


#%%
#啟動函數
x=tf.constant([-1.0,2.0,3.0])
x_relu=tf.nn.relu(x)
x_sigmoid=tf.math.sigmoid(x)
x_tanh=tf.math.tanh(x)



























