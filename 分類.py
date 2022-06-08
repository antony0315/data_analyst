import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pylab as plt
%matplotlib inline
#%%KNN
iris=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch15/iris.csv')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report
x=iris.iloc[:,0:4]
y=iris.iloc[:,4]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
iris_knn=KNeighborsClassifier(algorithm='kd_tree')
iris_knn.fit(x_train,y_train)
answer=iris_knn.predict(x)
answer_array=np.array([y,answer])
answer_mat=np.matrix(answer_array).T
result=pd.DataFrame(answer_mat)
result.columns=['真實','預測']
print(result)

print("KNN測試集測試結果")
print(classification_report(y_test,iris_knn.predict(x_test)))
print(53*"-")

scores=cross_val_score(iris_knn,x,y,cv=5)
cross=pd.DataFrame(scores)
cross.columns=['K-Fold結果']
print(cross)



#%%決策樹
iris=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch15/iris.csv')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report
x=iris.iloc[:,0:4]
y=iris.iloc[:,4]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
iris_tree=DecisionTreeClassifier(criterion='entropy')
iris_tree.fit(x_train,y_train)
pred=iris_tree.predict(x_test)
answer_array=np.array([y_test,pred])
answer_mat=np.matrix(answer_array).T
result=pd.DataFrame(answer_mat)
result.columns=['真實','預測']
print(result)

from sklearn import tree
fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)#dpi解析度
tree.plot_tree(iris_tree,
               feature_names = fn, 
               class_names=cn,
               filled = True);
#fig.savefig('imagename.png')
#驗證
print('測試集')
print(classification_report(y_test,iris_tree.predict(x_test)))
print(20*'-','全部',20*'-')
print(classification_report(y,iris_tree.predict(x)))
#交叉驗證
score=cross_val_score(iris_tree,x,y,cv=5)
cross=pd.DataFrame(score)
cross.columns=['K-Fold=5']
print(cross)
#%%隨機森林
from sklearn.ensemble import RandomForestClassifier
iris_rf=RandomForestClassifier(n_estimators=100)  #樹木數量
iris_rf.fit(x_train,y_train)
answer=np.array([y,iris_rf.predict(x)])
answer=answer.T
answer=pd.DataFrame(answer)
answer.columns=['真實','預測']

print('test predict result')
print(classification_report(y_test,iris_rf.predict(x_test)))
print('total predict result')
print(classification_report(y,iris_rf.predict(x)))

#K-FOLD
score=cross_val_score(iris_rf,x,y)
cross=pd.DataFrame(score)
cross.columns=['K-Fold5']
print(cross)
#%%SVM
from sklearn import svm
iris_svm=svm.SVC()
iris_svm.fit(x_train,y_train)
answer=iris_svm.predict(x)
answer_array=np.array([y,answer])
result=pd.DataFrame(np.matrix(answer_array).T)
result.columns=['真實',"預測"]
print(result)

print(classification_report(y_test,iris_svm.predict(x_test)))
print(classification_report(y,iris_svm.predict(x)))

#%%LDA
mouse_d=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch15/mouse_discrim.csv')
def getminindex(lis):
    lis_copy=lis[:]
    lis_copy.sort()
    minvalue=lis_copy[0]
    minindex=lis.index(monvalue)
    return minindex
from sklearn.preprocessing import scale
from scipy.spatial.distance import mahalanobis

def mahalanuobis_discrim(x_test,x_train,train_label):
    final_result=[]
    colname=x_train.columns
    test_n=x_test.shape[0]
    train_n=x_train.shape[0]
    m=x_train.shape[1]
    n=test_n+train_n
    data_x=x_train.append(x_test)
    data_x_scale=scale(data_x)
    data_x_scale=pd.DataFrame(data_x_scale[:train_n])
    x_train_scale=pd.DataFrame(data_x_scale[:train_n])
    x_test_scale=pd.DataFrame(data_x_scale[train_n:])
    data_train=x_train_scale.join(train_label)
    label_name=data_train.columns[-1]
    miu=data_train.groupby(label_name).mean()
    miu=np.array(miu)
    print('類中心:',pd.DataFrame(miu))
    print()
    label=train_label.drop_duplicates()
    label=label.iloc[:,0]
    label=list(label)
    label_len=len(label)
    x_test_array=np.array(x_test_scale)
    x_train_array=np.array(x_train_scale)
    data_x_scale_array=np.array(data_x_scale)
    cov=np.cov(data_x_scale_array.T)
    for i in range(n):
        dist=[]
        for j in range(label_len):
            d=float(mahalanobis(data_x_scale[i],miu[j],np.mat(cov)))
            dist.append(d)
        min_dist_index=getminindex(dist)
        result-label[min_dist_index]
        final_result.append(result)
    print('分類結果:')
    return final_result
x_train=mouse_d.iloc[0:13,1:6]
x_test=mouse_d.iloc[13:15,1:6]
y_train=mouse_d.iloc[0:13,6:7]
predict_mahalanobis=mahalanuobis_discrim(x_test, x_train, y_train)
print(predict_mahalanobis)




