import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

#%%-------單總體參數區間估計-------------------------------
  #大樣本  n>=30   Z檢定
  #小樣本  n<30     標準差已知:Z檢定       標準差未知:T檢定
#Z區間估計
moisture=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch6/moisture.csv')
sm.stats.DescrStatsW(moisture['moisture']).zconfint_mean(alpha=0.05)

#T區間估計
sm.stats.DescrStatsW(moisture['moisture']).tconfint_mean(alpha=0.05)
#T分布-法2   alpha=0.95
df_mean,df_var,df_std=stats.bayes_mvs(moisture['moisture'],alpha=0.95)
print(df_mean)  #母體平均 與 置信區間
#df_var  變異數與置信區間
#df_std  標準差與置信區間

#mean var std 信賴區間Z
mean,var,std=stats.mvsdist(moisture)
mean.interval(0.95)
var.interval(0.95)
std.interval(0.95)

#伯努力
#抽樣100個 5個不合格  99%信心水準  產品合格率區間
sm.stats.proportion_confint(95,100,alpha=0.01,method='normal')
                                 #normal sgresti_coull beta wilson jeffrey binom_test

#%%-------------單總體的假設檢定-------------------------------------
   #大樣本 n>=30 使用:Z統計量    無母體std 使用樣本std代替
   #小樣本 n<30  母體std已知:Z統計量
#============================Z統計量===================================
#EX1:n>=30    H0:u<=4     H1:u>4      
sm.stats.DescrStatsW(moisture['moisture']).ztest_mean(value=4,alternative='larger')
                                    #two-sided   larger   smaller
                                    #傳回 Z-score,p-value   >不拒絕H0
#============================T統計量====================================
#EX2:n<30 population std unknow      H0:u<=82   H1:u>82
mobile=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch6/mobile.csv')
sm.stats.DescrStatsW(mobile['csi']).ttest_mean(value=82,alternative='larger')
                                       #p-value>0.05   不拒絕H0
       #使用scipy.stats計算
stats.ttest_1samp(a=mobile['csi'],popmean=82)
           #H1:<  t>=0 p=1-p/2；  t<0  p=p/2
           #H1:>  t>=0 p=p/2  ；  t<0  p=1-p/2
#============================比例檢定==================================
#H0:p<=0.97   H1:P>0.97   
stats.binom_test(95,100,p=0.97,alternative='greater')  #two-sided   less greater
                              #p>0.05  不拒絕H0   無法證明良率>0.97
sm.stats.binom_test(95,100,prop=0.97,alternative='larger')
sm.stats.proportions_ztest(95,100,value=0.97,alternative='larger')


#%%------------兩總體假設檢定
#=====================================獨立樣本平均差檢定====================================
#先對兩樣本std檢定     再做T檢定(母體為常態分布)
#=====兩樣本標準差檢定   H0:var相等     Ha:var不相等
battery=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch6/battery.csv')
#方法1
stats.bartlett(battery[battery['tech']==1]['Endurance'],
               battery[battery['tech']==2]['Endurance'])
                                   #p-vale>0.01不拒絕  std相等
#方法2
stats.levene(battery[battery['tech']==1]['Endurance'],
               battery[battery['tech']==2]['Endurance'])
#====T檢定     
#直接用統計量檢驗(mean1,std1,樣本數1,mean2,std1,樣本數2,equal_var=True)
stats.ttest_ind_from_stats(3.7257,0.2994,35,3.9829,0.4112,35)
#H0:u1-u2=0     H1:u1-u2!=0
stats.ttest_ind(battery[battery['tech']==1]['Endurance'],
               battery[battery['tech']==2]['Endurance'],equal_var=True)  #p<0.01 拒絕H0
stats.ttest_ind_from_stats(mean1,std1,n1,mean2,std2,n2)
sm.stats.ttest_ind(df1,df2,alternative='two-sided',uservar='pooled',value=0)
                                       #uservar='pooled','unequal'  指定std是否相等  #value=(u1-u2=0)
#H0:u1-u2>=-0.1   u1-u2<-0.1         u2比u1多0.1
sm.stats.ttest_ind(battery[battery['tech']==1]['Endurance'],
               battery[battery['tech']==2]['Endurance'],
               alternative='smaller',usevar='pooled',value=0.1)
                             #拒絕H0   u1比u2 小0.1


#========================獨立樣本比例檢定=========================================
#H0=p1-p2<=-0.3    H1:p1-p2>-0.3
magzine=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch6/magzine.csv')
magzine['name']=magzine['name'].astype('category')
magzine['name'].cat.categories=['Fashin','Cosmetic']

magzine['gender']=magzine['gender'].astype('category')
magzine['gender'].cat.categories=['Male','Female']

female=magzine[magzine['gender']=='Female']['name'].value_counts()
magzines=magzine['name'].value_counts()


sm.stats.proportions_ztest(np.array(female),np.array(magzines),value=0.3,alternative='smaller',prop_var=False)
#p-value>0.05 不拒絕H0

#========================成對樣本比例檢定==============================================
  #2015年  與 2016年 資料變化  是否有提升  2016年建立在2015年基礎上
#H0=u1-u2>=0     H0:u1-u2<0
happiness=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch6/happiness.csv')
stats.ttest_rel(happiness['Year2015'],happiness['Year2016'])
sm.stats.ttost_paired(happiness['Year2015'],happiness['Year2016'],-0,0)#上下界[-0,0]













































































