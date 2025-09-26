import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats

import os

from renormalization import entropy_derivative
from analysis_rc import neurogym_hidden_states, NEUROGYM_TASKS, network_label_mapping, NEUROGYM_tasks_abbr, NEUROGYM_tasks_name, palette_deep, palette_light, pvalue_label
from performance_figure import save_fig

from data_connectome import Connectome
connectome = Connectome()
alphas = (0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0)
models = [2, 32, 53, 51, 52]
network_names = [network_label_mapping[connectome.conn_name(m)] for m in models]

tau_search = np.power(10, np.linspace(-4, 4, 401))
all_entropy_derivative = np.load(os.path.join('results', 'neurogym', 'entropy_derivative.npy'))
all_entropy = np.load(os.path.join('results', 'neurogym', 'entropy.npy'))
print(all_entropy_derivative.shape)
entropy_variance = np.var(all_entropy, axis=1)
mean_entropy_derivative = np.nanmean(all_entropy_derivative, axis=(1, 3))
mean_entropy = np.nanmean(all_entropy, axis=(1, 3))

peak_sigma = [[[tau_search[np.nanargmax(k)]
                 for k in j] 
                for j in i] 
               for i in entropy_variance]
peak_sigma = np.array(peak_sigma)
peak_sigma_logarithm = np.log(peak_sigma)
sigma_variance = np.var(peak_sigma_logarithm, axis=-1) # 对任务取方差
sigma_mean = np.mean(peak_sigma, axis=-1) # 对任务取平均
print(sigma_mean.shape)

peak = [[[[tau_search[np.nanargmax(l)]
                  for l in k] 
                 for k in j] 
                for j in i] 
               for i in all_entropy_derivative]
peak = np.array(peak)
print(peak.shape)
peak_logarithm = np.log(peak)
peak_variance = np.var(peak_logarithm, axis=-1) # 对任务求方差

# Fig 5 A
# Network average of the entropy parameter
sns.set_theme(font='Arial', context='paper', style='ticks', font_scale=2)
for j in range(1, 5):
    fig, ax = plt.subplots(figsize=(2.2, 2))
    ax.set_xscale('log')
    i = 0
    ax.plot(tau_search, np.nanmean(entropy_variance, axis=2)[i, 4] * 246, c=palette_deep[i], lw=3, label=network_names[i])
    ax.plot(tau_search, np.nanmean(entropy_variance, axis=2)[j, 4] * 246, c=palette_deep[j], lw=3, label=network_names[j])
    ax.set_ylabel(r'$\Sigma$')
    ax.set_xlabel(r'$\tau$')
    # fig.savefig(os.path.join('visualization', 'renormalization', f'{j}.png',), transparent=True, bbox_inches='tight')
    save_fig(fig, 'visualization', 'Fig 5 inset renormalization',  transparent=True)

sns.set_theme(font='Arial', context='paper', style='ticks', font_scale=2)
for j in range(1, 5):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xscale('log')
    ax1 = ax.twinx()
    i = 0
    ax.plot(tau_search, mean_entropy[i, 4], c=palette_deep[i], lw=3, label=network_names[i])
    ax.plot(tau_search, mean_entropy[j, 4], c=palette_deep[j], lw=3, label=network_names[j])
    ax1.plot(tau_search, mean_entropy_derivative[i, 4], '--', c=palette_deep[i], alpha=.4, lw=3)
    ax1.plot(tau_search, mean_entropy_derivative[j, 4], '--', c=palette_deep[j], alpha=.4, lw=3)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$S$')
    ax1.set_ylabel(r'$C$',)
    ymin, ymax = ax.get_ylim()
    ax.vlines(peak_sigma.mean(2)[i, 4], linestyles='--', lw=1.5, ymin=ymin, ymax=ymax, colors=palette_light[i])
    ax.vlines(peak_sigma.mean(2)[j, 4], linestyles='--', lw=1.5, ymin=ymin, ymax=ymax, colors=palette_light[j])
    ax.set_ylim(ymin, ymax)
    ax.legend(loc='center left', frameon=False)
    save_fig(fig, 'visualization', f'Fig 5 Renormalization entropy and derivative {network_names[j]}')

# Fig 5 B
# The time point of maximal variability versus spectrum radius
df = pd.DataFrame(
    sigma_mean, columns=alphas, index=network_names
).stack().reset_index(name=r'$\tau$ of maximal variability').rename(columns={'level_1': 'Spectrum radius', 'level_0': 'Network type'})
sns.set_theme(font='Arial', context='paper', style='ticks', font_scale=2)
fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.pointplot(
    data=df, 
    x="Spectrum radius", 
    y=r'$\tau$ of maximal variability',
    hue='Network type', 
    errorbar='sd',
    dodge=.1,
    palette=palette_deep,
    alpha=.7,
    lw=3.,
    ax=ax,
)
ax.legend(frameon=False, loc='lower left')
ax.set_yscale('log')
sns.despine(ax=ax)
save_fig(fig, 'visualization', 'Fig 5 Renormalization tau of maximal variability')

# Fig 5 C
# Variance of peak time points for task
df = []
y_name = r'Variance of $\ln\tau$ of peak $C$ for tasks'
for i, n in enumerate(network_names):
    df_tmp = pd.DataFrame(
        peak_variance[i], columns=alphas
    ).stack().reset_index(name=y_name)
    df_tmp = df_tmp.rename(columns={'level_1': 'Spectrum radius', 'level_0': 'id'})
    df_tmp['Network type'] = n
    df.append(df_tmp)
df = pd.concat(df)
df_tmp = df[df['Spectrum radius']==1.0]
sns.set_theme(font='Arial', context='paper', style='ticks', font_scale=2)
fig, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(
    data=df_tmp, 
    x="Spectrum radius", 
    y=y_name,
    hue="Network type",
    hue_order=network_names,
    # split=True, 
    inner="quart",
    density_norm = 'width',
    # fill=False,
    palette = palette_light,
#    palette={"Yes": "g", "No": ".35"}
    saturation=1.0,
    gap=.1,
    ax=ax,
)
ax.legend(frameon=False, loc='best')
sns.despine(bottom=True)
ax.tick_params(bottom=False)
k = len(ax.collections)
quarter_color = (.4,.4,.4,1)
median_color = (.3,.3,.3,1)
for j in range(k):
    for line in (ax.lines[j * 3 + 0],  ax.lines[j * 3 + 2]): # 0下四分位线 和 2上四分位线
        line.set_color(quarter_color)
        line.set_linestyle('--')
    line = ax.lines[j * 3 + 1] # 中位线
    line.set_color(median_color)
    line.set_linestyle('-')
    
    if j:
        x = df_tmp[(df_tmp['Network type'] == network_names[0])][y_name]
        y = df_tmp[(df_tmp['Network type'] == network_names[j])][y_name]
        pvalue = scipy.stats.mannwhitneyu(x, y, alternative='less').pvalue
        ax.text(line.get_xdata().mean(), 1.4, pvalue_label(pvalue), horizontalalignment='center')
save_fig(fig, 'visualization', 'Fig 5 Variance of peak time points for tasks')