# ENTER LIST OF LOG FILENAMES HERE:
filepaths = ['../log-files/Pendulum-v0/May-15_18:41:08/log.csv']  # Pendulum-v0  LunarLanderContinuous-v2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotting import df_plot
plt.rcParams["figure.figsize"] = (10,5)
plt.style.use('ggplot')

dataframes = []
names = []
for filepath in filepaths:
    names.append(filepath.split('/')[-2])
    dataframes.append(pd.read_csv(filepath))
data = list(zip(dataframes, names))
for df, name in data:
    ma = np.convolve(df['_MeanReward'], np.ones((5,))/5, mode='valid')
    print('{}: Maximum 100 Episode Average = {:.3g} around Episode {}'
          .format(name, np.max(ma), np.argmax(ma)*20))

# Mean Reward, D_KL(pi_old || pi_new), and Value Function Explained Variance
df_plot(data, '_Episode', ['_MeanReward'])
df_plot(data, '_Episode', ['KL'], ylim=(0,0.2))
df_plot(data, '_Episode', ['ExplainedVarOld'], ylim=(0,1))

# Policy Entropy
df_plot(data, '_Episode', ['PolicyEntropy'])

# Observations Statistics (after scaling)
df_plot(data, '_Episode', ['_mean_obs'])
df_plot(data, '_Episode', ['_max_obs', '_min_obs'], ylim=(-5, 5))
df_plot(data, '_Episode', ['_std_obs'], ylim=(0, 0.5))

# Actions Statistics
df_plot(data, '_Episode', ['_mean_act', '_max_act', '_min_act'])
df_plot(data, '_Episode', ['_std_act'])

# Discounted Reward Statistics
df_plot(data, '_Episode', ['_mean_discrew'])
df_plot(data, '_Episode', ['_std_discrew'])

# Policy and Value Function Training Loss
df_plot(data, '_Episode', ['ValFuncLoss'])
df_plot(data, '_Episode', ['PolicyLoss'])

