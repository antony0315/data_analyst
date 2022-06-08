import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hc
#%%系統聚類
#方法1
mc=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch14/mouse_cluster.csv')
z=hc.linkage(mc.iloc[:,1:],method='average')#single,complete,average,weighted,centroid,median,ward
dd=hc.dendrogram(z,orientation='right',labels=list(mc.iloc[:,0]))#top,bottom,left,right
hc.fcluster(z,2.6,criterion='distance')   #依照2.6閾值分類
hc.fclusterdata(mc.iloc[:,1:],2.6,criterion='distance',metric='euclidean',method='average')


#方法2
from sklearn.cluster import AgglomerativeClustering as AC
cm=AC(n_clusters=3,linkage='average',affinity='euclidean')
cl=cm.fit(mc.iloc[:,1:])
print(cl.labels_)
dd_ward=hc.dendrogram((hc.linkage(mc.iloc[:,1:],method='ward')),
                      orientation='right',labels=list(mc.iloc[:,0]))


#%%K-MEANS
#方法1(每次結果不同)
from scipy.cluster.vq import kmeans2
kmeans2(mc.iloc[:,1:],3)
from scipy.cluster.vq import kmeans,whiten
kmeans(whiten(mc.iloc[:,1:]),3)
#方法2
from sklearn.cluster import KMeans
mc_km=KMeans(n_clusters=3).fit(mc.iloc[:,1:])
print(mc_km.labels_)
cluster_by_Kmeans=pd.concat([mc['brand'],pd.DataFrame(mc_km.labels_,columns=['cluster'])],axis=1)
print(cluster_by_Kmeans)
print(mc_km.cluster_centers_)

#%%DBSCAN
from sklearn.cluster import DBSCAN
mc_DB=DBSCAN(eps=1.5,min_samples=1).fit(mc.iloc[:,1:])
cluster_by_DBSCAN=pd.concat([mc['brand'],pd.DataFrame(mc_DB.labels_,columns=['cluster'])],axis=1)
print(cluster_by_DBSCAN)










