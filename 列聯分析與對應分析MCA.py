import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']
#%%列聯分析
sc=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch13/salary_reform.csv')
sc['department']=sc['department'].astype('category')
sc['department'].cat.categories=['發展戰略部','客戶服務部','市場部','研發中心','綜合部']
sc['attitude']=sc['attitude'].astype('category')
sc['attitude'].cat.categories=['支持','反對']
#列聯表
sc_contingencytable=pd.crosstab(sc['attitude'],sc['department'],margins=True)
print(sc_contingencytable)
#列聯表總人數%
sc_contingencytable/sc_contingencytable.loc['All']['All']
#列的%
def percent_observed(data):
    return data/data[-1]
pd.crosstab(sc['attitude'],sc['department'],margins=True).apply(percent_observed,axis=1)

#欄的%
pd.crosstab(sc['attitude'],sc['department'],margins=True).apply(percent_observed,axis=0)

#列聯表期望值
from scipy.stats import contingency
pd.DataFrame(contingency.expected_freq(sc_contingencytable),
             columns=sc_contingencytable.columns,
             index=sc_contingencytable.index)
#%%卡方檢驗    H0:各部門 支持反對獨立   H1:各部門  不獨立
#分兩類:最小的期望值應要>5
#分兩類以上:期望值小於<5的比例<20%
stats.chi2_contingency(sc_contingencytable.iloc[:-1,:-1])  #return:(chi2,p-value,df,expect)
#p>0.05不顯著接受H0
#2*2列聯
stats.fisher_exact(sc_contingencytable.iloc[:-1,:-4])
#%%對應分析
import prince
cp=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch13/CellPhone.csv')
cp['Element']=cp['Element'].astype('category')
cp['Element'].cat.categories=['價格','待機時間','外觀','功能','I/O介面','網路相容性','記憶體大小','品牌','鏡頭','品質','作業系統']
cp['Income']=cp['Income'].astype('category')
cp['Income'].cat.categories=['<1000元','1000-3000元','3000-5000元','5000-8000元','8000-10000元','>10000元']
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']
mca = prince.MCA()
mca.fit(cp)
mca.plot_coordinates(cp,row_points_alpha=0.2,figsize=(10,6),show_column_labels=True)
mca.eigenvalues_
mca.total_inertia_
mca.explained_inertia_


