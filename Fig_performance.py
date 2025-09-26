from matplotlib import style
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import label_binarize
# import statsmodels.stats
import seaborn as sns

import os
# from rc import DataSet
from performance_figure import save_fig
from data_connectome import Connectome
from analysis_rc import NEUROGYM_tasks_abbr, NEUROGYM_tasks_name, network_label_mapping, palette_light, palette_deep

connectome = Connectome()
task_abbr_mapping = dict(zip([''.join(i.split()) for i in NEUROGYM_tasks_name], NEUROGYM_tasks_abbr))
models = [2, 32, 53, 51, 52]
network_names = [network_label_mapping[connectome.conn_name(m)] for m in models]
label_mapping = network_label_mapping

# Results with added noise, modified specific values of alpha
metrics = 'F1 score'
# df_performance = []
# for m, name in zip(models, network_names):
#     disc = connectome.conn_name(m)
#     # fc_lists = []
#     for rand_id in range(0, 1000):
#         # fc_path = os.path.join('results', 'neurogym', disc, str(rand_id), f'fc_tasks_{rand_id}.npy')
#         # fc = np.load(fc_path, allow_pickle=True)
#         # fc_lists.append(fc)
#         csv_name = os.path.join('results', 'neurogym', disc, str(rand_id), f'alphas_performance.csv')
#         df_i = pd.read_csv(csv_name)
#         df_i['id'] = rand_id % 1000
#         df_i['Network type'] = name
#         df_performance.append(df_i)
# df_performance = pd.concat(df_performance)
# df_performance = df_performance.rename(columns={'f1_score': 'F1 score', 'alpha': 'Spectrum radius', 'task': 'Task'})
# df_performance = df_performance[['Spectrum radius', 'Task', 'Network type', 'F1 score', 'id']]
# df_performance['Task'] = df_performance['Task'].apply(lambda x: task_abbr_mapping[x])
# df_performance.to_pickle(os.path.join('results', 'neurogym', 'performance.pkl'))
all_df = pd.read_pickle(os.path.join('results', 'neurogym', 'performance.pkl'))
all_df['Spectrum radius'] = np.exp(np.round(np.log(all_df['Spectrum radius']), 3)) # Fix floating point error

# # Fig 1
# tmp_df = all_df[all_df['Network type'] == network_names[0]].copy()
# # tmp_df['Task'] = tmp_df['Task'].apply(func)
# task_colors = sns.color_palette("cubehelix", 15)[:14] + sns.color_palette("bwr", 8)[2:0:-1] + sns.color_palette("gist_stern", 16)[4:0:-1] + sns.color_palette("rocket", 6)[3:]
# tasks = all_df['Task'].unique()
# # c_name = 'BNA_matrix_bfr_randmized'
# # tmp_df = all_df[(all_df['Network type'] == c_name) & (all_df['ID'] == 85) & (all_df['Spectrum radius'] == 1.0)]
# sns.set_theme(font='Arial', style='ticks')
# fig = plt.figure(figsize=(12, 3))
# gs = fig.add_gridspec(1, 2, width_ratios=(2, 1))
# ax = fig.add_subplot(gs[0])
# lines = []
# # x = df_tmp['Spectrum radius'].unique()
# for i, t in enumerate(NEUROGYM_tasks_abbr):
#     tmp_series = tmp_df[tmp_df['Task'] == t].groupby(['Spectrum radius'])['F1 score'].mean()
#     x = tmp_series.keys().values
#     y = tmp_series.values
#     line = ax.plot(x, y, color=task_colors[i], lw=3., zorder=2)
#     lines.append(line[0])
# # ax.set_ylim(0.3, 1.0)
# # ax.set_yticks([0.5, 1], ['0.5', '1.0'], )
# # ax.set_xlim(1/3, 3)
# ax.set_ylim(0., 1.01)
# ax.set_yticks([0, 1], ['0', '1'], )
# ax.set_xlim(1/20, 20)
# ax.set_xscale('log')
# ax.axvline(x=1, color=(.5, .5, .5, .5), dashes=(6, 2), zorder=1, lw=2)
# ax.set_xlabel('Spectrum radius')
# ax.set_ylabel('F1 score')
# lg = fig.legend(lines, NEUROGYM_tasks_abbr, title='Task', ncol=2, labelspacing=0.3, loc=(.68, .14), frameon=False)
# sns.despine(ax=ax)
# save_fig(fig, 'visualization', 'Fig 1 mean task performance')

# # Fig 2
# # Select the best performing alpha
# tmp_df = all_df.groupby(['Network type', 'Spectrum radius', 'Task'])['F1 score'].mean().reset_index()
# tmp_df = tmp_df.iloc[tmp_df.groupby(['Network type', 'Task'])['F1 score'].idxmax()].drop(columns='F1 score')
# tmp_df = tmp_df.merge(all_df, on=['Network type', 'Spectrum radius', 'Task'], how='left')
# # tmp_df = all_df.groupby(['Network type', 'Spectrum radius', 'Task'])['F1 score'].mean().reset_index()
# # tmp_df = tmp_df.groupby(['Network type', 'Task'])['F1 score'].max().reset_index()
# tmp_df = tmp_df.groupby(['Network type', 'id', ])['F1 score'].mean().reset_index()
# sns.set_theme(font='Arial', style='ticks', font_scale=1.5)
# fig = plt.figure(figsize=(10, 5))
# gs = fig.add_gridspec(1, 2, width_ratios=(3, 2))
# ax = fig.add_subplot(gs[0])
# sns.boxplot(
#     data=tmp_df, 
#     y='F1 score', hue="Network type", 
#     hue_order=network_names,
#     ax=ax,
#     palette=palette_deep,
#     gap=.1,
#     linewidth=4,
#     fliersize=0,
#     zorder=2,
# )
# sns.stripplot(
#     data=tmp_df, 
#     y='F1 score', hue="Network type", 
#     hue_order=network_names,
#     ax=ax,
#     palette=palette_light, 
#     dodge=True, 
#     legend=False,
#     jitter=.32,
#     size=7,
#     alpha=.3,
#     zorder=1,
# )
# ax.legend(bbox_to_anchor=[0.98, 0.7], frameon=False)
# ax.set_xticks([])
# ax.set_xlabel('Network type')
# sns.despine(ax=ax)
# save_fig(fig, 'visualization', 'Fig 2 mean best performance')
# # Mann-Whitney U rank test
# with open(os.path.join('visualization', 'Fig 2 mean best performance test.txt'), 'w') as fp:
#     fp.write('Mann-Whitney U rank test, one-sided.\n')
#     for i in range(1, 5):
#         x = tmp_df[tmp_df['Network type']==network_names[0]]['F1 score']
#         y = tmp_df[tmp_df['Network type']==network_names[i]]['F1 score']
#         fp.write(f'Median: {network_names[0]} {np.median(x)}, {network_names[i]} {np.median(y)}\n')
#         fp.write(str(scipy.stats.mannwhitneyu(x, y, alternative='greater')) + '\n\n')

# # Fig 2
# # mean performance
# sns.set_theme(font='Arial', style='ticks', font_scale=1.5)
# palette=palette_deep
# fig, ax = plt.subplots(1, 1, figsize=(6, 5))
# ax = sns.lineplot(
#     x='Spectrum radius', y='F1 score', hue="Network type", 
#     hue_order=network_names,
#     # err_style=None,
#     data=tmp_df,
#     palette=palette,
#     ax = ax,
#     lw = 2.5,
#     zorder=2,
# )
# ax.set_xlim((0.05 / 1.2, 20 * 1.2))
# ax.set_xscale('log')
# ax.set_xlabel(r'Spectrum radius')
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels, frameon=False, loc='best')
# # ax.axvline(x=1, color=(1, 0, 1, .2), dashes=(6, 2)) # 
# ax.axvline(x=1, color=(.5, .3, 0.5, .4), dashes=(6, 2), lw=2.5, zorder=1,)
# sns.despine(ax=ax)
# save_fig(fig, 'visualization', 'Fig 2 mean performance')

# # Fig S best performance
# tmp_df = all_df.groupby(['Network type', 'Spectrum radius', 'Task'])['F1 score'].mean().reset_index()
# tmp_df = tmp_df.iloc[tmp_df.groupby(['Network type', 'Task'])['F1 score'].idxmax()].drop(columns='F1 score')
# tmp_df = tmp_df.merge(all_df, on=['Network type', 'Spectrum radius', 'Task'], how='left')
# # tmp_df = all_df.groupby(['Network type', 'Spectrum radius', 'Task'])['F1 score'].mean().reset_index()
# # tmp_df = tmp_df.groupby(['Network type', 'Task'])['F1 score'].max().reset_index()
# model_performance = tmp_df.groupby(['Task', 'Network type'])['F1 score'].median().unstack().values.round(3)
# bna_performance = model_performance[:, np.where(tmp_df.groupby(['Task', 'Network type'])['F1 score'].mean().unstack().columns == network_names[0])[0][0]]
# print('BNA achieved the best mean performance in {} tasks.'.format((bna_performance >= model_performance.max(1)).sum()))
# # Layout with 8 subplots per row, more compact
# sns.set_theme(font='Arial', style='ticks', font_scale=3)
# g = sns.catplot(
#     data=tmp_df, y="F1 score", 
#     hue="Network type",
#     hue_order=network_names,
#     kind="box",
#     col='Task',
#     col_wrap=8,
#     palette=palette_deep,
#     gap=.1,
#     linewidth=4,
#     fliersize=0,
#     zorder=2,
# )
# g.map_dataframe(
#     sns.stripplot, y="F1 score", hue="Network type", 
#     hue_order=network_names,
#     dodge=True, palette=palette_light, 
#     jitter=.3,
#     size=7,
#     alpha=.3,
#     zorder=1,
# )
# g.set_titles('{col_name}')
# g.set(ylim=(.0, 1.1), xticks=[])
# g.legend.set_bbox_to_anchor([0.87, 0.17])
# save_fig(g.figure, 'visualization', 'Fig S best performance')

# # Fig S performance
# # tmp_df = all_df.copy()
# tmp_df = all_df.groupby(['Network type', 'Task', 'Spectrum radius'])['F1 score'].mean().reset_index()
# tmp_df['Network type'] = tmp_df['Network type'].map(label_mapping)
# # tmp_df['Task'] = tmp_df['Task'].apply(lambda x: ''.join([i if i.isupper() else '' for i in x]))
# sns.set_theme(font='Arial', style='ticks', font_scale=3)
# palette=palette_deep
# g = sns.relplot(
#     data=tmp_df, x='Spectrum radius', y=metrics, 
#     hue='Network type', 
#     hue_order=network_names,
#     palette=palette, 
#     kind='line', 
#     col='Task',
#     col_wrap=8,
#     orient='x',
#     linewidth=4.,
#     alpha=.8,
#     zorder=2,
# )
# g.set_titles('{col_name}')
# for ax in g.axes.flatten():
#     ax.set_xscale('log')
#     ax.set_ylim(0., 1.05)
#     ax.grid(visible=False, which='both', axis='both')
#     ax.set_xlim(1/21, 21)
#     ax.axvline(x=1, color=(.5, .3, 0.5, .4), dashes=(6, 2), zorder=1,)
# # g.legend.set_bbox_to_anchor([0.86, 0.15])
# g.legend.set_bbox_to_anchor([0.88, 0.17])
# g.savefig(os.path.join('visualization', 'Fig S performance.png'), pad_inches=.21)