import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


#%%  Wilcoxon 中位數檢定(分布未知)     H0:M=M0
water=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch8/water.csv')
water['Net'].median()
def wilcoxon_signed_rank_test(samp,mu0=0):
    temp=pd.DataFrame(np.asarray(samp),columns=['origin_data'])
    temp['D']=temp['origin_data']-mu0
    temp['rank']=abs(temp['D']).rank()
    posW=np.sum(temp.loc[temp.D>0,['rank']])
    negW=np.sum(temp.loc[temp.D<0,['rank']])
    n=temp.loc[temp.D!=0,['rank']].count()
    z=(posW-n*(n+1)/4)/np.sqrt((n*(n+1)*(2*n+1))/24)
    p=(1-stats.norm.cdf(abs(z)))*2
    return z,p
wilcoxon_signed_rank_test(water['Net'],mu0=600)
#H1:<   單側P-value=p-value/2    <a=0.05  reject   H0

#%%分布檢驗    K-S檢驗  
#H0:F(x)=F0   H1:F(x)!=F0
ks=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch8/ks.csv')
#常態分佈檢驗
sm.stats.diagnostic.kstest_normal(ks['observation'])
sm.stats.diagnostic.lilliefors(ks['observation'])
#stats.kstest 可檢驗上百種分布
stats.kstest(ks['observation'],'norm',(ks['observation'].mean(),ks['observation'].std()))
#shapiro-wilk
stats.shapiro(ks['observation'])
#anderson(常態分布、指數分布.....)
stats.anderson(ks['observation'],dist='norm')
#QQplot
import seaborn as sns
sns.distplot(ks['observation'], fit=stats.norm);
fig = plt.figure()
res = stats.probplot(ks['observation'], plot=plt)

#%%隨機性檢驗   H0:資料室隨機的
runs=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch8/runs.csv')
sm.stats.runstest_1samp(np.asarray(runs['economics']),cutoff='median')
#%%中位數相等    H0:M1=M2
sales_district=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch8/sales_district.csv')
stats.ranksums(sales_district.loc[sales_district.district==1,['Sales']],
               sales_district.loc[sales_district.district==2,['Sales']])

stats.mannwhitneyu(sales_district.loc[sales_district.district==1,['Sales']],
               sales_district.loc[sales_district.district==2,['Sales']],
               alternative='two-sided')
#%%獨立樣本分布檢驗   H0:F1=F2   H1:F1!=F2
from scipy import stats
rng = np.random.default_rng()
n1 = 200  # size of first sample
n2 = 300  # size of second sample
rvs1 = stats.norm.rvs(size=n1, loc=0., scale=1, random_state=rng)
rvs2 = stats.norm.rvs(size=n2, loc=0.5, scale=1.5, random_state=rng)
stats.ks_2samp(rvs1, rvs2)

cafe_scale=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch8/cafe_scale.csv')
cafe1=cafe_scale.loc[cafe_scale.city==1,['Computers']].values.flatten()
cafe2=cafe_scale.loc[cafe_scale.city==2,['Computers']].values.flatten()
stats.ks_2samp(cafe1,cafe2)

plt.hist(cafe1,bins=50,density=True,histtype='step',cumulative=True)
plt.hist(cafe2,bins=50,density=True,histtype='step',cumulative=True)
plt.show()
#%% 成對樣本爭位數檢驗    H0:M(去年)>=M(今年)     
stats.wilcoxon(df1,df2)
#%%兩樣本連串檢定    H0:兩樣本來自同個總體   Wald-wolfowitz
runs2=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch8/runs2.csv')
sm.stats.runstest_2samp(runs2['score'].astype('float64'),groups=runs2['group'])



































