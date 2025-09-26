import bct
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm
from PIL import Image
import scipy.stats

import os
from sys import getsizeof

from analysis_rc import neurogym_hidden_states, NEUROGYM_TASKS, network_label_mapping, pvalue_label, palette_deep, palette_light

import sklearn
from sklearn.preprocessing import minmax_scale
from scipy.stats import zscore

from data_connectome import Connectome
from performance_figure import save_fig

connectome = Connectome()
models = [2, 32, 53, 51, 52]
network_names = [network_label_mapping[connectome.conn_name(m)] for m in models]
alphas = (0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0)
tasks = NEUROGYM_TASKS


h_tv_alphas = np.load(os.path.join('results', 'neurogym', 'task_variance.npy'))[0]
print(h_tv_alphas.shape)
tv_corr = []
for a in alphas:
# for a in alphas[:1]:
    h_tv_stable = h_tv_alphas[:, alphas.index(a)]
    mat_alphas = [np.corrcoef(t) for t in h_tv_stable]
    mat = np.mean(mat_alphas, axis=0)
    tv_corr.append(mat)

# # Fig 4 A
# # Correlation matrix of task variance
# left_right = np.hstack((np.arange(0, 246, 2), np.arange(1, 246, 2)))
# yeo_7 = np.hstack([np.arange(246)[connectome.regions_to_labels(i)] for i in connectome.yeo_7network[1:]])
# yeo_7_left_right = np.hstack(([i for i in yeo_7 if i in np.arange(0, 246, 2)], [i for i in yeo_7 if i in np.arange(1, 246, 2)]))
# sns.set_theme(font='Arial', style='white', font_scale=1.5)
# fig, axes = plt.subplots(1, 3, figsize=(16, 4))
# selected_alphas = [0.1, 1., 10.]
# for i, a in enumerate(selected_alphas):
#     ax = axes.flatten()[i]
#     mat = tv_corr[alphas.index(a)]
#     mat = mat[left_right][:, left_right]
#     sns.heatmap(
#         mat,
#         # cmap=sns.diverging_palette(230, 20, as_cmap=True),
#         cmap='viridis',
#         square=True,
#         ax=ax,
#     )
#     ax.axis('off')
#     ax.set_title(f'Spectrum radii $={a}$')
# fig.subplots_adjust(wspace=.1)
# save_fig(fig, 'visualization', 'Fig 4 Correlation matrix of task variance')

# # Fig S 9
# # Correlation matrices of the task variance of the brain network
# left_right = np.hstack((np.arange(0, 246, 2), np.arange(1, 246, 2)))
# yeo_7 = np.hstack([np.arange(246)[connectome.regions_to_labels(i)] for i in connectome.yeo_7network[1:]])
# yeo_7_left_right = np.hstack(([i for i in yeo_7 if i in np.arange(0, 246, 2)], [i for i in yeo_7 if i in np.arange(1, 246, 2)]))
# sns.set_theme(font='Arial', style='white', font_scale=1.5)
# fig, axes = plt.subplots(3, 3, figsize=(20, 16))
# for i, a in enumerate(alphas):
#     ax = axes.flatten()[i]
#     mat = tv_corr[i]
#     mat = mat[left_right][:, left_right]
#     sns.heatmap(
#         mat,
#         # cmap=sns.diverging_palette(230, 20, as_cmap=True),
#         cmap='viridis',
#         square=True,
#         ax=ax,
#     )
#     ax.axis('off')
#     ax.set_title(f'Spectrum radii $={a}$')
# save_fig(fig, 'visualization', 'Fig 4 Correlation matrices of the task variance of the brain network')

df = pd.read_pickle(os.path.join('results', 'neurogym', 'task_variance_graph_attr.pkl'))
df_ami = pd.read_pickle(os.path.join('results', 'neurogym', 'modular_similarity_networks_ami.pkl'))
# # Fig 4 B and C
# sns.set_theme(font='Arial', context='paper', style='ticks', font_scale=2)
# fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
# selected_nets = [0, 1]
# df_tmp = df[df['Network type'].isin([network_names[i] for i in selected_nets])]
# # df_tmp = df_tmp[df_tmp['Spectrum radius'].isin([0.1, 0.5, 0.8, 1.0, 1.25, 2.0, 10.])]
# palette_tmp = [palette_deep[i] for i in selected_nets]

# # Fig 4 B Modularity
# # ax = sns.boxplot(
# #     data=df_tmp, x='Spectrum radius', y='Small-worldness', hue='network', 
# #     fill=False,
# #     palette=palette_tmp,
# #     gap=.05,
# #     ax=ax,
# #     )
# y_name = 'Modularity'
# ax = axes[0]
# sns.pointplot(
#     data=df_tmp, 
#     x='Spectrum radius', 
#     y=y_name, 
#     hue='Network type', 
#     estimator='median',
#     palette=palette_tmp,
#     ax=ax,
# )
# ax.legend(frameon=False, loc='lower left')
# sns.despine(ax=ax)
# for i, a in enumerate(alphas):
#     x = df_tmp[(df_tmp['Spectrum radius'] == a) & (df_tmp['Network type'] == network_names[0])][y_name]
#     y = df_tmp[(df_tmp['Spectrum radius'] == a) & (df_tmp['Network type'] == network_names[1])][y_name]
#     pvalue = scipy.stats.mannwhitneyu(x, y, alternative='greater').pvalue
#     pos_x = ax.get_xticklabels()[i].get_position()[0]
#     pos_y = ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) / 30
#     ax.text(pos_x, pos_y, pvalue_label(pvalue), horizontalalignment='center')
    
# # Fig 4 modular similarity with intrinsic networks
# y_name = 'AMI'
# df_tmp = df_ami
# ax = axes[1]
# sns.boxplot(
#     data=df_tmp,
#     x='Spectrum radius',
#     y=y_name,
#     fill=False,
#     hue='Network type',
#     hue_order=network_names[:2],
#     palette=palette_deep[:2],
#     # gap=.05,
#     ax=ax,
# )
# ax.legend(frameon=False)
# sns.despine(ax=ax)
# fig.subplots_adjust(wspace=.2)
# for i, a in enumerate(alphas):
#     x = df_tmp[(df_tmp['Spectrum radius'] == a) & (df_tmp['Network type'] == network_names[0])][y_name]
#     y = df_tmp[(df_tmp['Spectrum radius'] == a) & (df_tmp['Network type'] == network_names[1])][y_name]
#     pvalue = scipy.stats.mannwhitneyu(x, y, alternative='greater').pvalue
#     pos_x = ax.get_xticklabels()[i].get_position()[0]
#     pos_y = ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) / 30
#     ax.text(pos_x, pos_y, pvalue_label(pvalue), horizontalalignment='center')
# save_fig(fig, 'visualization', 'Fig 4 Modularity and similarity with 7 networks')

# # Fig S 8 modularity and small-worldness
# sns.set_theme(font = 'Arial', context='paper', style='ticks', font_scale=1.5)
# fig = plt.figure(figsize=(16, 14))
# gs = fig.add_gridspec(2, 1, height_ratios=(6, 8))
# ax = fig.add_subplot(gs[0])
# sns.boxplot(
#     data=df, 
#     x='Spectrum radius', 
#     y='Modularity', 
#     hue='network', 
#     hue_order=network_names,
#     fill=False,
#     palette=palette_deep,
#     gap=.05,
#     ax=ax,
# )
# ax.legend(frameon=False)
# ax = fig.add_subplot(gs[1])
# sns.boxplot(
#     data=df, 
#     x='Spectrum radius', 
#     y='Small-worldness', 
#     hue='network', 
#     hue_order=network_names,
#     fill=False,
#     palette=palette_deep,
#     gap=.05,
#     ax=ax,
# )
# ax.legend(frameon=False)
# sns.despine(fig=fig)
# save_fig(fig, 'visualization', 'Fig S Modularity and small-worldness')
# with open(os.path.join('visualization', 'Fig 4 Modularity and small-worldness.txt'), 'w') as fp:
#     tmp_df = df[df['Spectrum radius'] == 1.0]
#     fp.write('Modularity. Mann-Whitney U rank test, one-sided.\n')
#     for i in range(1, 5):
#         x = tmp_df[tmp_df['network']==network_names[0]]['Modularity']
#         y = tmp_df[tmp_df['network']==network_names[i]]['Modularity']
#         fp.write(f'Median: {network_names[0]} {np.median(x)}, {network_names[i]} {np.median(y)}\n')
#         fp.write(str(scipy.stats.mannwhitneyu(x, y, alternative='greater')) + '\n\n')
#     fp.write('Small-worldness. Mann-Whitney U rank test, one-sided.\n')
#     for i in range(1, 5):
#         x = tmp_df[tmp_df['network']==network_names[0]]['Small-worldness']
#         y = tmp_df[tmp_df['network']==network_names[i]]['Small-worldness']
#         fp.write(f'Median: {network_names[0]} {np.median(x)}, {network_names[i]} {np.median(y)}\n')
#         fp.write(str(scipy.stats.mannwhitneyu(x, y, alternative='greater')) + '\n\n')