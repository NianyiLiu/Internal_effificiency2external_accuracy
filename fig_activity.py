import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib import colors

import os

from analysis_rc import neurogym_hidden_states, NEUROGYM_TASKS, network_label_mapping, NEUROGYM_tasks_abbr, NEUROGYM_tasks_name, palette_deep, palette_light, pvalue_label
from performance_figure import save_fig

from data_connectome import Connectome


def mean_fc_dist(fc, distance_matrix, norm_dist=False):
    # fc[fc < 0] = 0
    # return (fc * connectome.distance_matrix[:210, :210]).sum() / (210 * 210)
    mask = np.triu(np.ones(fc.shape[-2:], dtype=bool), 1)
    dist_vec = distance_matrix[mask]
    if norm_dist:
        dist_vec = (dist_vec - dist_vec.min()) / (dist_vec.max() - dist_vec.min())
    if fc.ndim == 2:
        return (abs(fc)[mask] * dist_vec).mean()
    else: # ndim >= 3
        return (abs(fc)[..., mask] * dist_vec).mean(-1)

def mean_fc(fc):
    mask = np.triu(np.ones(fc.shape[-2:], dtype=bool), 1)
    if fc.ndim == 2:
        return abs(fc)[mask].mean()
    else: # ndim == 3
        return abs(fc)[..., mask].mean(-1)
    
def fc_corr(fc, distance_matrix):
    mask = np.triu(np.ones(fc.shape[-2:], dtype=bool), 1)
    if fc.ndim == 2:
        corr = scipy.stats.spearmanr(abs(fc[mask]), distance_matrix[mask])[0]
    else:
        tmp_dist = np.expand_dims(distance_matrix, axis=tuple(range(fc.ndim-2)))
        for i, x in enumerate(fc.shape[:-2]):
            tmp_dist = np.repeat(tmp_dist, x, axis=i)
        s = fc.shape[:-2]
        corr = scipy.stats.spearmanr(abs(fc[..., mask].ravel()), tmp_dist[mask].ravel())[0]
    return corr

def round_sf_np(x,significant_figure=0):
    r=np.ceil(np.log(x)/np.log(10))
    f=significant_figure
    return np.round(x*(10**(f-r)),0)*(10**(r-f))

connectome = Connectome()
models = [2, 32, 53, 51, 52]
network_names = [network_label_mapping[connectome.conn_name(m)] for m in models]
task_abbr_mapping = dict(zip([''.join(i.split()) for i in NEUROGYM_tasks_name], NEUROGYM_tasks_abbr))
alphas = (0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0)
tasks = NEUROGYM_TASKS
mask = np.triu(np.ones((210, 210), dtype=bool), 1)
dist = connectome.distance_matrix[:210, :210]
mean_dist = dist[mask].mean()

# # Load data, load performance separately
# # df_performance = []
# # for m, name in zip(models, network_names):
# #     disc = connectome.conn_name(m)
# #     # fc_lists = []
# #     for rand_id in range(1000):
# #         # fc_path = os.path.join('results', 'neurogym', disc, str(rand_id), f'fc_tasks_{rand_id}.npy')
# #         # fc = np.load(fc_path, allow_pickle=True)
# #         # fc_lists.append(fc)
# #         csv_name = os.path.join('results', 'neurogym', disc, str(rand_id), f'alphas_performance.csv')
# #         df_i = pd.read_csv(csv_name)
# #         df_i['id'] = rand_id
# #         df_i['Network type'] = name
# #         df_performance.append(df_i)
# # df_performance = pd.concat(df_performance)
# # df_performance = df_performance.rename(columns={'f1_score': 'F1 score', 'alpha': 'Spectrum radius', 'task': 'Task'})
# # df_performance = df_performance[['Spectrum radius', 'Task', 'Network type', 'F1 score', 'id']]
# # df_performance['Task'] = df_performance['Task'].apply(lambda x: task_abbr_mapping[x])
# # df_performance.to_pickle(os.path.join('results', 'neurogym', 'performance.pkl'))
# df_performance = pd.read_pickle(os.path.join('results', 'neurogym', 'performance.pkl'))
# df_performance['Spectrum radius'] = df_performance['Spectrum radius'].apply(lambda x: np.round(round_sf_np(x, 2), 3))

# # # Load data and perform FC-related calculations
# # print('Calculating multiplication...')
# # df = []
# # for rand_id in range(1000):
# #     fc_list = []
# #     for m in models:
# #         disc = connectome.conn_name(m)
# #         fc_path = os.path.join('results', 'neurogym', disc, str(rand_id), f'fc_tasks_{rand_id}.npy')
# #         fc = np.load(fc_path, allow_pickle=True)[:, :, :210, :210]
# #         fc_list.append(fc)
# #     fc_list = np.asarray(fc_list)
# #     fc = mean_fc(fc_list)
# #     fc_dist = mean_fc_dist(fc_list, dist)
# #     # fc_norm_dist = mean_fc_dist(fc_list, dist, True)
# #     df_tmp = []
# #     for k, n in enumerate(network_names):
# #         for i, a in enumerate(alphas):
# #             for j, t in enumerate(tasks):
# #                 f_tmp = pd.DataFrame({
# #                     # 'id': rand_id,
# #                     'Task': t,
# #                     'Spectrum radius': a,
# #                     'Network type': n,
# #                     'FC multiply distance': fc_dist[k, i, j], 
# #                     # 'FC multiply normalized distance': fc_norm_dist[k, i, j], 
# #                     'Mean FC': fc[k, i, j],
# #                 }, index=[rand_id])
# #                 df_tmp.append(f_tmp)
# #     df_tmp = pd.concat(df_tmp)
# #     df.append(df_tmp)
# # df = pd.concat(df)
# # df.reset_index(names='id', inplace=True)
# # df['Task'] = df['Task'].apply(lambda x: task_abbr_mapping[x])
# # df.to_pickle(os.path.join('results', 'neurogym', 'distance_weighted_fc.pkl'))
# # print('Calculation over.')
# df = pd.read_pickle(os.path.join('results', 'neurogym', 'distance_weighted_fc.pkl'))

# print('Calculating correlatioin...')
# dist_vec = connectome.distance_matrix[:210, :210][mask]
# df_corr = []
# for m, n in zip(models, network_names):
#     disc = connectome.conn_name(m)
#     for rand_id in range(1000):
#         fc_path = os.path.join('results', 'neurogym', disc, str(rand_id), f'fc_tasks_{rand_id}.npy')
#         fc = np.load(fc_path, allow_pickle=True)
#         df_tmp = []
#         for i, a in enumerate(alphas):
#             for j, t in enumerate(tasks):
#                 # perf = [p[selected_alphas.index(a), j] for p in performances]
#                 f_vec = abs(fc[i, j][:210, :210][mask])
#                 # spearmanr = scipy.stats.spearmanr(f_vec, dist_vec)
#                 r, p = scipy.stats.pearsonr(f_vec, dist_vec)
#                 if r is np.nan:
#                     r = 0
#                     p = 1
#                 f_tmp = pd.DataFrame({
#                     'Task': t,
#                     'Spectrum radius': a,
#                     'Network type': n,
#                     # 'spearmanr': spearmanr[0],
#                     # 'spearmanp': spearmanr[1],
#                     'pearsonr': r,
#                     # 'pearsonp': p,
#                 }, index=[rand_id])
#                 df_tmp.append(f_tmp)
#         df_tmp = pd.concat(df_tmp)
#         df_corr.append(df_tmp)
# df_corr = pd.concat(df_corr)
# df_corr.reset_index(names='id', inplace=True)
# df_corr['Task'] = df_corr['Task'].apply(lambda x: task_abbr_mapping[x])
# df_corr.rename(columns={
#     'pearsonr': 'Correlation coefficient', 
#     # 'pearsonp': 'P-value',
# }, inplace=True)
# df_corr.to_pickle(os.path.join('results', 'neurogym', 'distance_fc_corr.pkl'))
# print('Calculation over.')
df_corr = pd.read_pickle(os.path.join('results', 'neurogym', 'distance_fc_corr.pkl'))

# # Fig 3 Correlation
# # correlation between affinity and distance
# # Mean correlation of tasks in different networks
# print('Ploting correlation...')
# sns.set_theme(font='Arial', context='paper', style='ticks', font_scale=2)
# palette_line_tmp = [palette_deep[0], palette_deep[1]]
# palette_face_tmp = [palette_light[0], palette_light[1]]
# df_tmp = df_corr[df_corr['Network type'].isin([network_names[i] for i in [0, 1]])]
# fig, ax = plt.subplots(figsize=(9, 5))
# sns.violinplot(
#     data=df_tmp, 
#     x="Spectrum radius", 
#     y="Correlation coefficient", 
#     hue="Network type",
#     split=True,
#     inner="quart",
#     density_norm = 'width',
#     gap=.2,
#     fill=True,
#     saturation=1.,
#     palette=palette_face_tmp,
#     ax=ax,
# )
# ax.legend(frameon=False, loc='best')
# ax.set_ylim((-.7, .15))
# sns.despine(ax=ax)
# # n = 2
# # face_alpha = .4
# # k = len(ax.collections) // n
# # for i in range(n):
# #     for j in range(i, n * k, n):
# #         ax.collections[j].set_facecolor((*palette_face_tmp[i], face_alpha))
# # For the legend, use ax.patches; for median and quartile lines, use ax.lines
# n = len(palette_line_tmp)
# k = len(ax.collections) // n
# quarter_color = (.4,.4,.4,1)
# median_color = (.3,.3,.3,1)
# for i in range(n):
#     # r, g, b = palette_line_tmp[i]
#     for j in range(i, n * k, n):
#         for line in (ax.lines[j * 3 + 0],  ax.lines[j * 3 + 2]):
#             line.set_color(quarter_color)
#             line.set_linestyle('--')
#         line = ax.lines[j * 3 + 1]
#         line.set_color(median_color)
#         line.set_linestyle('-')
# for i, a in enumerate(alphas):
#     x = df_tmp[(df_tmp['Spectrum radius'] == a) & (df_tmp['Network type'] == network_names[0])]['Correlation coefficient']
#     y = df_tmp[(df_tmp['Spectrum radius'] == a) & (df_tmp['Network type'] == network_names[1])]['Correlation coefficient']
#     pvalue = scipy.stats.mannwhitneyu(x, y, alternative='less').pvalue
#     if pvalue is np.nan: pvalue = 1.
#     ax.text(ax.get_xticklabels()[i].get_position()[0], .1, pvalue_label(pvalue), horizontalalignment='center')
# save_fig(ax.figure, 'visualization', 'Fig 3 correlation between affinity and distance')

mean_corr = np.load(os.path.join('results', 'neurogym', f'gradients_{4}_tasks_correlation.npy'))
# # Fig 3 gradients correlation matrix
# c_id = 1
# colors_list = [palette_deep[c_id], [0.95, 0.9, 0.5], [1.,1.,1.,], ]
# colors_list = [palette_deep[c_id], [1.,1.,1.,], palette_light[c_id], ]
# cmap = colors.LinearSegmentedColormap.from_list('How2matplotlib_custom', colors_list, N=100)
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
# mean_task_corr = mean_corr[c_id, :, 4].mean(0)
# sns.set_theme(font='Arial', style='white', font_scale=1.2)
# fig, ax = plt.subplots(figsize=(8, 6))
# df_tmp = pd.DataFrame(mean_task_corr, columns=NEUROGYM_tasks_abbr, index=NEUROGYM_tasks_abbr)
# sns.heatmap(
#     df_tmp, 
#     mask=np.tril(np.ones(mean_task_corr.shape)), 
#     # cmap=cmap,
#     # cmap=sns.cubehelix_palette(start=.5, rot=-.75, reverse=True, light=1., dark=0.5, gamma=.8, as_cmap=True),
#     cmap='crest_r',
#     square=True, 
#     vmax=0.9, 
#     vmin=0.1,
#     # cbar_kws={
#     #     # "shrink": 0.9,
#     #     "location": 'left',
#     #     },
#     ax=ax,
# )
# for i, t in enumerate(ax.get_yticklabels()):
#     # t.set(position=(i/23 + 0.065, 0), horizontalalignment='right')
#     t.set(position=(i/23 - 0.015, 0), horizontalalignment='center')
# ax.tick_params(left=False)
# ax.set_xticks([])
# # ax.text(-0.5, 22, network_names[0],)
# ax.text(7, -.5, network_names[c_id],)
# fig.savefig(os.path.join('visualization', 'Fig 3 grad tasks corr rewired BNA.png'), dpi=660, bbox_inches='tight', transparent=True)
# c_id = 0
# colors_list = [palette_deep[c_id], [0.95, 0.9, 0.5], [1.,1.,1.,], ]
# colors_list = [palette_deep[c_id], [1.,1.,1.,], [0.5, 0.9, 0.5], ]
# cmap = colors.LinearSegmentedColormap.from_list('How2matplotlib_custom', colors_list, N=100)
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
# mean_task_corr = mean_corr[c_id, :, 4].mean(0)
# sns.set_theme(font='Arial', style='white', font_scale=1.2)
# fig, ax = plt.subplots(figsize=(8, 6))
# df_tmp = pd.DataFrame(mean_task_corr.T, columns=NEUROGYM_tasks_abbr, index=NEUROGYM_tasks_abbr)
# sns.heatmap(
#     df_tmp, 
#     mask=np.triu(np.ones(mean_task_corr.shape)), 
#     # cmap=cmap,
#     # cmap=sns.cubehelix_palette(start=.5, rot=-.75, reverse=True, light=1., dark=0.5, gamma=.8, as_cmap=True),
#     cmap='crest_r',
#     square=True, 
#     vmax=0.9, 
#     vmin=0.1,
#     cbar_kws={
#         "location": 'left',
#         "pad": 0.05
#         },
#     ax=ax,
# )
# ax.set_xticks([])
# # for i, t in enumerate(ax.get_yticklabels()):
# #     t.set(position=(i/23 + 0.065, 0), horizontalalignment='right')
# ax.set_yticks([])
# ax.tick_params(left=False)
# # ax.text(-0.5, 22, network_names[1],)
# ax.text(7, 24, network_names[c_id],)
# fig.savefig(os.path.join('visualization', 'Fig 3 grad tasks corr BNA.png'), dpi=660, bbox_inches='tight', transparent=True)

# # Fig S
# # Explained variance of diffusion map
# lambdas = np.load(os.path.join('results', 'neurogym', 'gradients_explained_variance.npy')) # 5, 1000, 9, 244
# print(lambdas.shape)
# explained_variance = lambdas / np.sum(lambdas, -1, keepdims=True)
# sns.set_theme(font='Arial', context='paper', style='ticks', font_scale=1.5)
# fig, ax = plt.subplots(figsize=(6, 5))
# for i, n in enumerate(network_names):
#     ax.plot(np.arange(1, 13), explained_variance.mean(1)[i, 4, :12], '-', c=palette_deep[i], alpha=.6, lw=3, label=n)
#     ax.plot(np.arange(1, 13), explained_variance.mean(1)[i, 4, :12], 'o', c=palette_light[i], ms=7)
# ax.set_ylabel('Explained variance ratio')
# ax.set_xlabel('Component')
# ax.legend(frameon=False)
# fig.savefig(os.path.join('visualization', 'Fig S Explained variance of diffusion map.png'), dpi=660, bbox_inches='tight', transparent=True)

# # Fig S
# Correlation of main components between individual task and all tasks
# gradients_corr_between_all_and_tasks = np.load(os.path.join('results', 'neurogym', 'gradients_corr_between_all_and_tasks.npy'))
# sns.set_theme(font='Arial', context='paper', style='ticks', font_scale=1.5)
# n_comp = 6
# fig, axes = plt.subplots(n_comp, 1, figsize=(13, 14))
# for i in range(n_comp):
#     cmap = sns.diverging_palette(230, 20, as_cmap=True)
#     df_tmp = pd.DataFrame(
#         gradients_corr_between_all_and_tasks.mean(1)[:, 4, :, i], 
#         columns=NEUROGYM_tasks_abbr, 
#         index=['BNA', 'Rewired', 'ER', 'WS', 'BA'],
#     )
#     sns.heatmap(
#         df_tmp, 
#         square=True, 
#         cmap=cmap,
#         # cmap='crest_r',
#         vmax=1.0, 
#         vmin=0.1,
#         ax=axes[i],
#     )
#     if i != 5:
#         axes[i].set_xticks([])
#     axes[i].set_title(f'Component {i+1}')
# fig.savefig(os.path.join('visualization', 'Fig S Correlation of main components between individual task and all tasks.png'), dpi=660, bbox_inches='tight', transparent=True)

# # Fig S
# # Mean correlation of main task components in random networks
# mask_task = np.triu(np.ones((len(NEUROGYM_tasks_abbr), len(NEUROGYM_tasks_abbr)), dtype=bool), 1)
# mean_network_corr = mean_corr[:, :, :, mask_task].mean(-1)
# print(mean_network_corr.shape)
# df = []
# for i, n in enumerate(network_names):
#     df_tmp = pd.DataFrame(
#         mean_network_corr[i], columns=alphas
#     ).stack().reset_index(name='Mean correlation')
#     df_tmp = df_tmp.rename(columns={'level_1': 'Spectrum radius', 'level_0': 'id'})
#     df_tmp['Network type'] = n
#     df.append(df_tmp)
# df = pd.concat(df)
# sns.set_theme(font = 'Arial', context='paper', style='ticks', font_scale=2)
# fig, ax = plt.subplots(figsize=(12, 6))
# y_name = 'Mean correlation'
# sns.boxplot(
#     data=df, 
#     x='Spectrum radius', 
#     y=y_name, 
#     hue='Network type', 
#     hue_order=network_names,
#     fill=False,
#     palette=palette_deep,
#     gap=.1,
#     fliersize=0,
#     zorder=2,
#     ax=ax,
# )
# # for j, a in enumerate(alphas):
# #     pos_x = ax.containers[0].boxes[j].get_bbox().get_points()[:, 0].mean()
# #     x = df_tmp[(df_tmp['Spectrum radius'] == a) & (df_tmp['Network type'] == network_names[0])][y_name]
# #     y = df_tmp[(df_tmp['Spectrum radius'] == a) & (df_tmp['Network type'] == network_names[1])][y_name]
# #     pvalue = scipy.stats.mannwhitneyu(x, y, alternative='greater').pvalue
# #     ax.text(pos_x, ax.get_ylim()[1], pvalue_label(pvalue), horizontalalignment='center')
# sns.stripplot(
#     data=df, 
#     x='Spectrum radius', 
#     y=y_name, 
#     hue='Network type', 
#     hue_order=network_names,
#     ax=ax,
#     palette=palette_light, 
#     dodge=True, 
#     legend=False,
#     jitter=.32,
#     size=2,
#     alpha=.6,
#     zorder=1,
# )
# ax.legend(frameon=False)
# sns.despine(fig=fig)
# fig.savefig(os.path.join('visualization', 'Fig S Mean correlation of main task components in random networks.png'), dpi=660, bbox_inches='tight', transparent=True)

# # Fig S
# # Correlation coefficient of main components of task pairs
# color_map_names = ['Blues_r', 'Oranges_r', 'Purples_r', 'Greens_r', 'OrRd_r']
# sns.set_theme(font='Arial', style='white', font_scale=0.75)
# fig, axes = plt.subplots(1, 3, figsize=(16, 4))
# for idx, n in enumerate(network_names):
#     if idx in [0, 1]: continue
#     mean_task_corr = mean_corr[idx, :, 4].mean(0)
#     ax = axes[idx - 2]
#     df_tmp = pd.DataFrame(mean_task_corr, columns=NEUROGYM_tasks_abbr, index=NEUROGYM_tasks_abbr)
#     sns.heatmap(
#         df_tmp, 
#         mask=np.tril(np.ones(mean_task_corr.shape)), 
#         cmap=color_map_names[idx],
#         square=True, vmax=1, vmin=0,
#         # cbar_kws={"shrink": 0.5},
#         ax=ax,
#     )
#     ax.set_xticks([])
#     for i, t in enumerate(ax.get_yticklabels()):
#         t.set(position=(i/23 + 0.065, 0), horizontalalignment='right')
#     ax.tick_params(left=False)
#     ax.set_title(n)
#     ax.title.set_fontsize(ax.title.get_fontsize() * 2)
# fig.subplots_adjust(wspace=0.1)
# fig.savefig(os.path.join('visualization', 'Fig S grad tasks corr.png'), dpi=660, bbox_inches='tight')