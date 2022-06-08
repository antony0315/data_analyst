import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
dc_sales=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch7/dc_sales.csv')
dc_sales['pixel']=dc_sales['pixel'].astype('category')
dc_sales['pixel'].cat.categories=['500P','500-600P','600-800P','800-1000P','1000P']
pd.pivot_table(dc_sales,index=['pixel'],columns=['market'],values=['sales'],aggfunc='sum')


#====圖像化
G=dc_sales['pixel'].unique()
args=[]
for i in list(G):
    args.append(dc_sales[dc_sales['pixel']==i]['sales'])
dc_sales_plot=plt.boxplot(args,vert=True,patch_artist=True)
colors=['pink','lightblue','lightgreen','cyan','lightyellow']
for patch,color in zip(dc_sales_plot['boxes'],colors):
    patch.set_facecolor(color)
fig=plt.gcf()
combinbox=plt.subplot(111)
combinbox.set_xticklabels(G)

#%%一元方差分析 one-way ANOVA   (比較不同名目、尺度 的樣本平均值)
#H0:u1=u2=u3=u4  H1:u1,u2,u3,u4    1.母體為常態  2.獨立性  3.var同質性
#step1:VAR同質性檢驗   (Var需相等)
stats.levene(*args)   #p-value大 滿足方差齊性假設
#step2:F檢定
stats.f_oneway(*args)      #p-value<0 變數顯著影響  各組平均不同 reject H0
#查看ANOVA表
from statsmodels.formula.api import ols
dc_sales_anova=sm.stats.anova_lm(ols('sales~C(pixel)',dc_sales).fit())
print(dc_sales_anova)            #P-value<0.05   拒絕均值相等假設

#%%多重比較檢驗
#兩兩對比 檢查哪種影響較大
from statsmodels.stats.multicomp import pairwise_tukeyhsd
dc_sales_anova_post=pairwise_tukeyhsd(dc_sales['sales'],
                                      dc_sales['pixel'],alpha=0.05)
print(dc_sales_anova_post.summary())

#%% ANOVA模型  估計與預測
#含截距向
formula='sales~C(pixel)'
dc_sales_est=ols(formula,dc_sales).fit()
print(dc_sales_est.summary2())

#不含截距向
formula2='sales~C(pixel)'
dc_sales_est2=ols(formula2,dc_sales).fit()
print(dc_sales_est2.summary2())

#%%方差分析模型預測
print(dc_sales_est.fittedvalues)

dc_sales_influence=dc_sales_est.get_influence()
print(dc_sales_influence.summary_table())


#%%多因素方差分析
house=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch7/house.csv')
house['education']=house['education'].astype('category')
house['education'].cat.categories=['初中&以下','高中','大學','研究所&以上']
house['unit']=house['unit'].astype('category')
house['unit'].cat.categories=['國營企業','行政','學校','私人公司','unemployee','other']
house['income']=house['income'].astype('category')
house['income'].cat.categories=['<10000','10000-25000','25000-50000','50000-75000','>75000']
house['type']=house['type'].astype('category')
house['type'].cat.categories=['1R3L','2R1L','2R2L','3R1L','3R2L','3R3L','4R2L1B','4R2L2B','4R3L1B','4R3L2B','更大']
#anova 檢驗
formula='space~C(education)+C(unit)+C(income)+C(type)'
house_anova=sm.stats.anova_lm(ols(formula,data=house).fit(),typ=3)#type 3型檢驗
print(house_anova)
#education unit不顯著去除
formula2='space~C(income)+C(type)'
house_anova2=sm.stats.anova_lm(ols(formula2,data=house).fit(),typ=3)#type 3型檢驗
print(house_anova2)

#多重比較檢驗
house_anova_post=pairwise_tukeyhsd(house['space'],house['income'],alpha=0.05)
print(house_anova_post.summary())

#OLS估計式評估模型
house_anova_est=ols(formula2,data=house).fit()
print(house_anova_est.summary2())

#交互效應 多因素方差分析
formula3='space~C(income)*C(type)'
house_anova_inter=sm.stats.anova_lm(ols(formula3,data=house).fit())
print(house_anova_inter)

ols(formula3,data=house).fit().rsquared

#===交互影響圖
from statsmodels.graphics.api import interaction_plot
plt.figure(figsize=(12,6))
fig=interaction_plot(np.array(house['income']),np.array(house['type']),
                     house['space'],ax=plt.gca())
fig_adj=plt.subplot(111)
plt.legend(prop={'family':'SimHei','size':10.5},loc='upper left',
           frameon=False)
fig_adj.set_xticklabels(house['income'].unique(),fontproprties=myfront)

#%%協方差分析(類別型變數+連續型變數)
sale_points=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch7/sale_points.csv')
sale_points['market']=sale_points['market'].astype('category')
sale_points['market'].cat.categories=['market1','market2','market3']
sale_points['warranty']=sale_points['warranty'].astype('category')
sale_points['warranty'].cat.categories=['1year','3year']
#變數:  尺度:market、warranty    連續:points
formula='sales~points+C(market)*C(warranty)'
sale_points_anova_cov=sm.stats.anova_lm(ols(formula,data=sale_points).fit())
print(sale_points_anova_cov)
#參數估計與假設
sale_points_anova_cov_est=ols(formula,data=sale_points).fit()
print(sale_points_anova_cov_est.summary())

















































