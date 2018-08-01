"""
Clustering Analysis using E-M algorithm
"""
import matplotlib as mpl
from Code.PreAnalyzing.EM_Algo import EM_Algo_1D
mpl.use("TKAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sc
import sklearn.preprocessing as sp

src = pd.read_csv('data/last_anonym_2017_vartime.csv', sep=';', index_col=0, parse_dates=True)
outplot_path = ('Output/plot/')

# Drop out the dates that has consisten value, 70865.614814...
src_new = src[~ ((src['l'].values < 70865.615) & (src['l'].values > 70865.614))]

# ============test
# src_new['l'].plot()
#
# test = src_new[(src_new['l'].values < 70865.615) & (src_new['l'].values > 70865.614)]
#
# src_new['l'].value_counts()
# src['l'].value_counts()
# ============end test

# hf_load = load_curve[(load_curve > 70865) & (load_curve < 70866)]


# The exogenous variables can be separated into 2 groups
group_1 = src_new.columns[1:19]
group_2 = src_new.columns[19:44]
group_3 = src_new.columns[44:]

# schedule_group_1 = src_new.loc[:, group_1].values.reshape(1,-1).tolist()[0]
schedule_group_1 = src_new.loc[:, group_1].sum(axis=1).values.reshape(1, -1).tolist()[0]
# schedule_group_1_values = [ele for ele in schedule_group_1 if ele !=0]
# print(len(schedule_group_1_values))
# plt.hist(schedule_group_1_values,bins=50)

# schedule_group_2 = src.loc[:, group_2].values.reshape(1,-1).tolist()[0]
schedule_group_2 = src_new.loc[:, group_2].sum(axis=1).values.reshape(1, -1).tolist()[0]
# schedule_group_2_values = [ele for ele in schedule_group_2 if ele !=0]
# print(len(schedule_group_2_values))
# plt.hist(schedule_group_2_values,bins=100)

# schedule_group_3 = src.loc[:, group_3].values.reshape(1,-1).tolist()[0]
schedule_group_3 = src_new.loc[:, group_3].sum(axis=1).values.reshape(1, -1).tolist()[0]
# schedule_group_3_values = [ele for ele in schedule_group_3 if ele !=0]
# print(len(schedule_group_3_values))
# plt.hist(schedule_group_3_values,bins=50)

# Combine 3 groups of exogenous variables and load curve
gmm_df_4d = pd.DataFrame()
gmm_df_4d['load'] = src_new['l']
gmm_df_4d['schedule_1'] = schedule_group_1
gmm_df_4d['schedule_2'] = schedule_group_2
gmm_df_4d['schedule_3'] = schedule_group_3

# ===Check===
# gmm_df_4d['load'].plot()


# ========================EM Algorithm=======================
# On 1-dimensional clustering
np.random.seed(666)
X_test = gmm_df_4d['load'][0:10000]
cluster_num = 6
transform_flag = False

if transform_flag:
    scaler = sp.MinMaxScaler()
    X_scaled = scaler.fit_transform(X_test.values.reshape(-1,1))
    ran_idx = np.random.choice(len(X_scaled), cluster_num)
    sigma_init = np.random.random(cluster_num)
    mu_init = X_scaled[ran_idx].reshape(1, -1)[0]

    # plt.hist(X_scaled, bins=500)
else:
    X_scaled = X_test.values
    ran_idx = np.random.choice(len(X_scaled), cluster_num)
    sigma_init = np.random.randint(1000, 10000, cluster_num)
    mu_init = X_scaled[ran_idx]




em = EM_Algo_1D(X=X_scaled, K=cluster_num, mu_init=mu_init, sigma_init=sigma_init, stop_criteria=0.0001)
em.fit(iterate_num=5000)


# ====================compute the probability of sample that belongs to each distribution

mu_est = em.mu
X_test['']

