from matplotlib import style
import numpy as np
# import torch
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
from data_connectome import Connectome

def load_rc_memory_performance(n, result_dir, task_dir, model_dir, spectrum_radius_range, *metrics, fname='performance.csv', spectrum_radius_range_tag = None, spectrum_radius_tag = None, readout_tag = None, number_tag = None):
    '''
    Args
        readout_tag: 如果有 readout_tag 那么分组取平均的时候就要根据 readout_tag 分读出脑区。其参数值可以任意给定，作为新的列名，一般将 'class' 重命名
        spectrum_radius_range_tag: 如果不是``None``，那么就要按照 `spectrum_radius_range` 的映射来分组平均。其参数值可以任意给定，作为新的列名。
        spectrum_radius_tag: 如果不是``None``，那么输出的结果中就会有一列记录 spectrum_radius 。其参数值可以任意给定，作为新的列名，一般将 'alpha' 重命名
        number_tag: 如果不是``None``，那么输出的结果中就会有一列记录参数 `n` 。其参数值可以任意给定，作为新的列名。注意 `n` 不一定从 ``0`` 开始。
    '''
    # 可用的 metrics 包括 mse_train,performance_train,capacity_train,mse_valid,performance_valid,capacity_valid
    df = pd.read_csv(os.path.join(
        result_dir, task_dir, model_dir, str(n), fname
    ))
    # if spectrum_radius_range_tag is not None:
    #     assert isinstance(spectrum_radius_range, dict)
    #     # assert isinstance(spectrum_radius_range, Mapping) # collections.abc.Mapping
    #     df.loc[:, spectrum_radius_range_tag] = df.apply(
    #     lambda x: spectrum_radius_range.get(x['alpha']), axis=1)
    groups = []
    columns = [*metrics]
    mapper = {}
    if spectrum_radius_range is None: # 所有alpha都需要
        spectrum_radius_range = np.unique(df['alpha'])
        if spectrum_radius_range_tag is not None:
            spectrum_radius_range = {a: 'Stable' if a < 1. else 'Chaotic' if a > 1. else 'Critical' for a in spectrum_radius_range}
    if readout_tag is not None:
        groups.append('class')
        mapper['class'] = readout_tag
    if spectrum_radius_range_tag is None:
        assert spectrum_radius_tag is not None
        groups.append('alpha')
        mapper['alpha'] = spectrum_radius_tag
    else:
        assert isinstance(spectrum_radius_range, dict)
        # assert isinstance(spectrum_radius_range, Mapping) # collections.abc.Mapping
        df.loc[:, spectrum_radius_range_tag] = df.apply(lambda x: spectrum_radius_range.get(x['alpha']), axis=1)
        if spectrum_radius_tag is not None:
            groups.append('alpha')
            mapper['alpha'] = spectrum_radius_tag
        groups.append(spectrum_radius_range_tag)

    # if spectrum_radius_tag is None:
    #     assert spectrum_radius_range_tag is not None
    #     groups.append(spectrum_radius_range_tag)
    # else:
    #     groups.append(spectrum_radius_tag)
            
    # 这里用query的方法会变慢，所以还是用循环的方法
    #df = df.query('alpha in @include_alpha')\
    df = df[df['alpha'].isin(spectrum_radius_range)]\
        .groupby(groups)[columns]\
        .mean().reset_index()
    df.rename(columns=mapper, inplace=True)
    if number_tag is not None:
        df[number_tag] = n
    return df
def load_performance(n, result_dir, task_dir, model_dir, spectrum_radius_range, *metrics):
    ''' 按照 spectrum radius (alpha) 分组 '''
    # df = pd.read_csv(os.path.join(
    #     result_dir, task_dir, model_dir, str(n), fname
    # ))
    # # df = df[df['alpha'].isin(spectrum_radius_range)]\
    # #     .groupby('alpha')[['alpha', *metrics]]\
    # #     .mean().reset_index(drop=True)
    # # df = df.round({'alpha': 2}) # .mean()有时候会有浮点数误差
    # df = df[df['alpha'].isin(spectrum_radius_range)]\
    #     .groupby('alpha')[[*metrics]]\
    #     .mean().reset_index() # 不要drop掉 alpha 就没有问题
    # return df
    return load_rc_memory_performance(n, result_dir, task_dir, model_dir, spectrum_radius_range, *metrics, spectrum_radius_tag='alpha')
def load_subregion_performance(n, result_dir, task_dir, model_dir, spectrum_radius_mapping, spectrum_radius_range_tag, readout_tag, *metrics):
    ''' 按照 dynamics (spectrum radius <, =, or > 1) 和 读出节点 来分组。 '''
    # df = pd.read_csv(os.path.join(
    #     result_dir, task_dir, model_dir, str(n), fname
    # ))
    # df.loc[:, spectrum_radius_range_tag] = df.apply(
    #     lambda x: spectrum_radius_mapping.get(x['alpha']), axis=1)
    # df = df[df['alpha'].isin(spectrum_radius_mapping)]\
    #     .groupby([readout_tag, spectrum_radius_range_tag])[[*metrics]]\
    #     .mean().reset_index()
    # # df = df.round({'alpha': 2}) # .mean()有时候会有浮点数误差
    # return df
    return load_rc_memory_performance(n, result_dir, task_dir, model_dir, spectrum_radius_mapping, *metrics, spectrum_radius_range_tag=spectrum_radius_range_tag, readout_tag=readout_tag)

def normalize_zero_one(df, col):
    ''' 原地修改 col 的值 '''
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

def set_line_style(ax, legend = True, face_alpha=0.3, line_alpha=0.7):
    rgbs = [sns.color_palette('Paired', 6)[i] for i in [5, 1, 3]] # 色盘
    # !!! 特别注意：因为seaborn的版本修改原因，这里需要重写
    # 每个版本对应 sns.boxplot 的参数也不一样
    # 第一版是老版本
    # ax.artists 是之前旧版本的方法，现在已经不适用了，会报错
    # for i in range(len(ax.artists)):
    #     patch = ax.artists[i]
    #     r, g, b, a = patch.get_facecolor()
    #     patch.set_facecolor((r, g, b, .3))
    #     r, g, b = rgbs[i % 3]
    #     patch.set_edgecolor((r, g, b, .7))
    #     for j in range(i*6,i*6+6): 
    #         line = ax.lines[j]
    #         line.set_color((r, g, b, .7))
    #         line.set_mfc((r, g, b, .7))
    #         line.set_mec((r, g, b, .7))
    #

    # 第二版是0.12.2
    # # for i in range(6, len(ax.patches)):
    # ax.patches[0].set_edgecolor((*rgbs[0], 1))
    # ax.patches[2].set_edgecolor((*rgbs[1], 1))
    # ax.patches[4].set_edgecolor((*rgbs[2], 1))
    # ax.patches[1].set_edgecolor((*rgbs[0], .7))
    # ax.patches[3].set_edgecolor((*rgbs[1], .7))
    # ax.patches[5].set_edgecolor((*rgbs[2], .7))
    # # ax.patches[7].set_edgecolor((*(ax.patches[-3].get_facecolor()[:3]), .3))
    # # ax.patches[8].set_edgecolor((*(ax.patches[-3].get_facecolor()[:3]), .3))
    # # for i in range(0,7):
    # for i in range(0,len(ax.patches)):
    #     patch = ax.patches[i]
    #     r, g, b, a = patch.get_facecolor()
    #     # print(r,g,b)
    #     patch.set_facecolor((r, g, b, .3))
    #     if i >= 6:
    #         r, g, b = rgbs[i % 3]
    #         # print(r,g,b)
    #         patch.set_edgecolor((r, g, b, .7))
    #     # print(i)
    # # for j in range(len(ax.lines)): 

    # for i in range(0, len(ax.patches)):
    #     r, g, b = rgbs[i % 3]
    #     if i*6+6 <= len(ax.lines):
    #         for j in range(i*6,i*6+6): 
    #             line = ax.lines[j]
    #             line.set_color((r, g, b, .7))
    #             line.set_mfc((r, g, b, .7))
    #             line.set_mec((r, g, b, .7))
    #             # line.set_color((r, g, b, .0))
    #             # line.set_mfc((r, g, b, .0))
    #             # line.set_mec((r, g, b, .0))
    # # print(len(ax.lines))

    # 第三版是0.13.0
    n = 3 # 三种颜色，对应三个
    k = len(ax.patches) // n - 1
    # 图上 box 的颜色
    for i in range(n):
        for j in range(i * k, (i + 1) * k):
            patch = ax.patches[j]
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, face_alpha))
            patch.set_edgecolor((*rgbs[i], line_alpha))
    # 图例的颜色
    if legend:
        for i in range(n):
            patch = ax.patches[i-n]
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, face_alpha))
            patch.set_edgecolor((*rgbs[i], line_alpha))
    # 图上 其他线和点 的颜色
    for i in range(n):
        r, g, b = rgbs[i]
        for j in range(i * k * 6, (i + 1) * k * 6):
            line = ax.lines[j]
            line.set_color((r, g, b, line_alpha))
            line.set_mfc((r, g, b, face_alpha)) # 设置外点里面填充的颜色
            line.set_mec((r, g, b, line_alpha)) # 设置外点边框的颜色

def set_subregion_line_style(ax, legend, face_alpha=0.2, line_alpha=0.8):
    # !!! 特别注意：因为seaborn的版本修改原因，这里需要重写
    # 每个版本对应 sns.boxplot 的参数也不一样
    # 第一版是老版本
    # # for patch in axis.artists:
    # #     r, g, b, a = patch.get_facecolor()
    # #     patch.set_facecolor((r, g, b, 0.9))
    # rgbs = [sns.color_palette('Paired', 6)[i] for i in [1, 5, 3]]
    #     for i in range(len(axis.artists)):
    #         patch = axis.artists[i]
    #         r, g, b, a = patch.get_facecolor()
    #         patch.set_facecolor((r, g, b, .3))
    #         r, g, b = rgbs[i % 3]
    #         patch.set_edgecolor((r, g, b, .7))
    #         for j in range(i*6,i*6+6): 
    #             line = axis.lines[j]
    #             line.set_color((r, g, b, .7))
    #             line.set_mfc((r, g, b, .7))
    #             line.set_mec((r, g, b, .7))
    
    # 第二版是0.12.2
    # sns.boxplot(
    #     x=x, 
    #     y=y,
    #     data=df_tmp,
    #     palette=[sns.color_palette('Paired', 6)[i] for i in np.array([4,0,2], dtype=int)],
    #     hue=model_type_tag,
    #     order=subregion_labels,
    #     orient=orient,
    #     width=width,
    #     linewidth=1,
    #     ax=ax,
    #     showfliers=True,
    # )
    # rgbs = [sns.color_palette('Paired', 6)[i] for i in [5, 1, 3]]
    # # for i in range(6, len(ax.patches)):
    # # 这三个是图例里的
    # ax.patches[0].set_edgecolor((*rgbs[0], 1))
    # ax.patches[2].set_edgecolor((*rgbs[1], 1))
    # ax.patches[4].set_edgecolor((*rgbs[2], 1))
    # # 这里是最左边的
    # ax.patches[1].set_edgecolor((*rgbs[0], .7))
    # ax.patches[3].set_edgecolor((*rgbs[1], .7))
    # ax.patches[5].set_edgecolor((*rgbs[2], .7))
    # # ax.patches[7].set_edgecolor((*(ax.patches[-3].get_facecolor()[:3]), .3))
    # # ax.patches[8].set_edgecolor((*(ax.patches[-3].get_facecolor()[:3]), .3))
    # # for i in range(0,7):
    # for i in range(0,len(ax.patches)):
    #     patch = ax.patches[i]
    #     r, g, b, a = patch.get_facecolor()
    #     # print(r,g,b)
    #     patch.set_facecolor((r, g, b, .4))
    #     if i >= 6:
    #         r, g, b = rgbs[i % 3]
    #         # print(r,g,b)
    #         patch.set_edgecolor((r, g, b, .7))
    #         # print(i)
    # # for j in range(len(ax.lines)): 
    # for i in range(0, len(ax.patches)):
    #     r, g, b = rgbs[i % 3]
    #     if i*6+6 <= len(ax.lines):
    #         for j in range(i*6,i*6+6): 
    #             line = ax.lines[j]
    #             line.set_color((r, g, b, .7))
    #             line.set_mfc((r, g, b, .7))
    #             line.set_mec((r, g, b, .7))
    #             # line.set_color((r, g, b, .0))
    #             # line.set_mfc((r, g, b, .0))
    #             # line.set_mec((r, g, b, .0))

    # 第三版是0.13.0
    # palette=[sns.color_palette('Paired', 6)[i] for i in np.array([4,0,2], dtype=int)]
    # legend = 'auto'
    # legend = False
    # sns.boxplot(
    #     x=x, 
    #     y=y,
    #     data=df_tmp,
    #     # palette=((.8,.8,.8), (0,0,0), (.4,.4,.4)),
    #     hue=model_type_tag,
    #     hue_order=model_type_names,
    #     order=subregion_labels,
    #     orient=orient,
    #     width=width,
    #     linewidth=1,
    #     ax=ax,
    #     showfliers=True,
    #     # legend='brief',
    #     legend=legend,
    #     # fill=False,
    #     # saturation=1,
    #     flierprops={"marker": "d"},
    # )

    n = 3 # 对应三种网络
    k = len(ax.patches) // n - 1 # 6 个脑区

    rgbs = sns.color_palette("husl", k) # 色盘，真实网络
    colour1 = (0.0, 0.0, 0.0) # 随机网络1
    colour2 = (0.6 ,0.6 ,0.6) # 随机网络2
    # 设置 empirical 6个subregion各自的颜色
    for i in range(0, k):
        r, g, b = rgbs[i]
        patch = ax.patches[i]
        patch.set_facecolor((r, g, b, face_alpha))
        patch.set_edgecolor((r, g, b, line_alpha))
        for j in range(i*6, i*6+6):
            line = ax.lines[j]
            line.set_color((r, g, b, line_alpha))
            line.set_mfc((r, g, b, face_alpha)) # 设置外点里面填充的颜色
            line.set_mec((r, g, b, line_alpha)) # 设置外点边框的颜色
    # 设置 剩余部分
    for i in range(k, len(ax.patches)):
        patch = ax.patches[i]
        patch.set_facecolor((1, 1, 1, 0)) # 先去掉填充颜色
        if i in range(k, 2 * k): # rewired 6个 subregion 的边框
            patch.set_edgecolor(colour1)
            for j in range(i*6, i*6+6):
                line = ax.lines[j]
                line.set_color(colour1)
                # line.set_mfc(*colour1, face_alpha))
                line.set_mec(colour1)
        if i in range(2 * k, 3 * k): # rotated 6个 subregion 的边框
            patch.set_edgecolor(colour2)
            for j in range(i*6, i*6+6):
                line = ax.lines[j]
                line.set_color(colour2)
                line.set_mec(colour2)
    # ax.patches 和 box 的数量有关，
    # 如果 fill=True ，那么会在 ax.patches 中增加18个对象，即图上的 18 个box
    # 如果 legend 不为 False ，那么会在 ax.patches 中的最后增加3个对象，即图例 legend 中的三个Rectangle对象
    # 调整 facecolor 是通过 ax.patches[i].set_facecolor .set_edgecolor 这个方法，
    # fill=True 时，调整其边框是通过 ax.patches[i].set_edgecolor 这个方法，
    # 注意对 legend 的调整要在设置完后调用 ax.legend 才能生效
    # ax.lines 和 fill 参数有关，
    # 当 fill=True ，ax.lines 有 108 个元素
    # 当 fill=False 时， ax.lines 有126个元素，（推测）包含了原先的108个对象加上 3*6=18 个box的边线

    if legend:
        ax.patches[-3].set_edgecolor((1., 1., 1.)) # 图例legend中 Empirical 的颜色（白色）
        ax.patches[-2].set_edgecolor(colour1) # 图例legend中 rewired 的颜色
        ax.patches[-1].set_edgecolor(colour2) # 图例legend中 rotated 的颜色
        ax.legend(frameon=True, ncol=1, loc='best')
        
    return ax

def set_box_style(ax, face_alpha=0.2, line_alpha=0.8):
    rgbs = [sns.color_palette('husl', 6)[i] for i in [0, 4, 5, 3, 2, 1]]
    for i in range(len(ax.patches)):
        patch = ax.patches[i]
        # r, g, b, a = patch.get_facecolor()
        r, g, b = rgbs[i]
        patch.set_facecolor((r, g, b, face_alpha))
        patch.set_edgecolor((r, g, b, line_alpha))
        for j in range(i*6,i*6+6): 
            line = ax.lines[j]
            line.set_color((r, g, b, line_alpha))
            line.set_mfc((r, g, b, face_alpha))
            line.set_mec((r, g, b, line_alpha))
    # for patch in axis.artists:
    #     r, g, b, a = patch.get_facecolor()
    #     patch.set_facecolor((r, g, b, 0.9))
    return ax

def dynamic_performance_fig(df, tag_name_mappings, metrics, hug_tag, normalize, variable = 'alpha'):
    if normalize == 1:
        normalize_zero_one(df, metrics)
        ylim=(0.0, 1.0)
    elif normalize == 0.5:
        ylim=(0.5, 1.0)
    else:
        ylim=(0., df[metrics].max())
    df.rename(columns=tag_name_mappings, inplace=True)

    sns.set(font='Times New Roman', style="ticks", font_scale=2.5)
    x = tag_name_mappings[variable]
    y = tag_name_mappings[metrics]

    figsize=(20,8)
    palette=[sns.color_palette('Paired', 6)[i] for i in np.array([4,0,2], dtype=int)]
    width=0.8
    orient='v'
    linewidth=1.2

    fig = plt.figure(figsize=figsize)
    ax = fig.subplots()
    sns.boxplot(
        x=x, 
        y=y,
        data=df,
        palette=palette,
        hue=hug_tag,
        orient=orient,
        width=width,
        linewidth=linewidth,
        ax=ax,
        flierprops={"marker": "d"},
    )
    set_line_style(ax)

    ax.legend(frameon=False, ncol=1, loc='best')
    # ax.set_title(title)
    # plt.suptitle(suptitle)
    # ax.set_ylim(xlim)
    ax.set_ylim(ylim)
    # ax.xaxis.set_major_locator(MultipleLocator(x_major_loc))
    # ax.yaxis.set_major_locator(MultipleLocator(y_major_loc))
    sns.despine(offset=10, trim='False')
    return fig

def critical_distribution_fig(df, metrics, model_type_tag, model_type_names, normalize_01):
    df = df.loc[df['alpha'] == 1.0, :] # 如果这里.loc中不是 : 而是一个Sequence来选出一些列，那么后面就不需要copy()
    if normalize_01:
        df = normalize_zero_one(df.copy(), metrics) # 这里还是需要copy一下的。否则会报warning
        # SettingWithCopyWarning: 
        # A value is trying to be set on a copy of a slice from a DataFrame.
        # Try using .loc[row_indexer,col_indexer] = value instead
        # 注意一定要指定 col_indexer ，不能空缺或是用 `:`。
    sns.set_theme(font='Arial', style="ticks", font_scale=2.0)
    fig = plt.figure(figsize=(10,7))
    ax = plt.subplot(111)
    rgbs = [sns.color_palette('Paired', 6)[i] for i in [5, 1, 3]]
    for model_type in model_type_names:
        sns.distplot(
            df.loc[df[model_type_tag] == model_type, metrics].values,
            bins=50,
            hist=False,
            kde=True,
            # kde_kws={'shade':True},
            kde_kws={'fill':True},
            label=model_type,
            color=rgbs[model_type_names.index(model_type)],
        )
    if normalize_01 == 0:
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.get_yaxis().set_visible(False)
    ax.legend(frameon=False, ncol=1, loc='upper right')
    # ax.set_xlim(0.7, 0.95)
    sns.despine(offset=10, left=True, trim=True)
    return fig

def subregions_performance_fig(df, tag_name_mappings, metrics, readout_tag, readout_order, spectrum_radius_tag, spectrum_radius_fields, model_type_tag, model_type_names):
    sns.set(font='Times New Roman', style="ticks", font_scale=2.5)
    x=readout_tag
    y=tag_name_mappings[metrics]
    orient='v'
    width=0.7 #0.8
    figsize=(20,20) #12,12
    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(3, 1)
    for i in range(len(spectrum_radius_fields)):
        d = spectrum_radius_fields[i]
        df_tmp = df[df[spectrum_radius_tag] == d].copy()
        df_tmp[readout_tag] = df_tmp.apply(lambda x: x[readout_tag].split()[0], axis=1)
        # df_tmp.rename(columns=tag_name_mappings, inplace=True)
        ax = axes[i]
        legend = 'auto'
        # legend = False
        sns.boxplot(
            x=x, 
            y=y,
            data=df_tmp,
            # palette=((.8,.8,.8), (0,0,0), (.4,.4,.4)),
            hue=model_type_tag,
            hue_order=model_type_names,
            order=readout_order,
            orient=orient,
            width=width,
            linewidth=1,
            ax=ax,
            showfliers=True,
            # legend='brief',
            legend=legend,
            # fill=False,
            # saturation=1,
            flierprops={"marker": "d"},
        )
        set_subregion_line_style(ax, legend=legend)
        ax.set_xlabel(None)
        # ax.set_ylim(xlim)
        # ax.xaxis.set_major_locator(MultipleLocator(x_major_loc))
        if i == 2:
            ax.set_ylim((0.0, 0.55))
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
        else:
            ax.set_ylim((0.0, 1.1))
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_title(d)
        sns.despine(ax=ax, offset=0, trim='False')
    axes[0].set(xticklabels=[])
    axes[1].set(xticklabels=[])
    return fig

def critical_subregions_fig(
    df, tag_name_mappings, 
    metrics, readout_tag, readout_order, 
    spectrum_radius_tag, spectrum_radius_fields, 
    model_type_tag, model_type_names
):
    sns.set(font='Times New Roman', style="ticks", font_scale=2)
    model_type = model_type_names[0]
    df_tmp = df[df[model_type_tag] == model_type]

    figsize=(10,20)
    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(3, 1)
    x=readout_tag
    y=tag_name_mappings[metrics]
    orient='v'
    width=0.4
    linewidth=1
    for i in range(len(spectrum_radius_fields)):
        d = spectrum_radius_fields[i]
        df_series = df_tmp.loc[df_tmp[spectrum_radius_tag] == d, [metrics, readout_tag]] # df_series type 是 pandas.core.series.Series
        df_series[metrics] = ((df_series[metrics]-df_series[metrics].min())/(df_series[metrics].max()-df_series[metrics].min()))
        df_series.rename(columns=tag_name_mappings, inplace=True)
        df_series[readout_tag] = df_series.apply(lambda x: x[readout_tag].split()[0], axis=1)

        ax = axes[i]
        sns.boxplot(
            x=x, 
            y=y,
            data=df_series,
            # palette=rgbs,
            hue=x,
            legend=False,
            order=readout_order,
            orient=orient,
            width=width,
            linewidth=linewidth,
            ax=ax,
            flierprops={"marker": "d"},
            # **kwargs
        )
        set_box_style(ax)
        ax.set_xlabel('')
        # ax.set_xticklabels(None)
        ax.set_title(d)
        sns.despine(ax=ax, offset=10, trim='False')
    axes[0].set(xticklabels=[])
    axes[1].set(xticklabels=[])
    return fig

# def load_multitask_df(n, result_dir, task_dir, model_dir, fname='performance.pt', ):
#     r = torch.load(os.path.join(
#         result_dir, task_dir, model_dir, str(n), fname
#     ))
#     df_mean = pd.DataFrame.from_records(r, columns=['spectrum radius', 'train accuracy', 'valid accuracy',])
#     # df_mean = df_mean[df_mean['spectrum radius'].isin(spectrum_radius_range)]\
#     #     .groupby('spectrum radius')[['spectrum radius', 'train accuracy', 'valid accuracy']]\
#     #     .mean().reset_index(drop=True)
#     df_all = pd.DataFrame.from_records(r, columns=('spectrum radius', 'train accuracy on tasks', 'valid accuracy on tasks', 'tasks')).apply(pd.Series.explode, ).reset_index(drop=True)
#     # df_all = df_all[df_all['spectrum radius'].isin(spectrum_radius_range)]\
#     #     .groupby('spectrum radius')[['spectrum radius', 'train accuracy on tasks', 'valid accuracy on tasks', 'tasks']]\
#     #     .mean().reset_index(drop=True)
#     return df_mean, df_all

def select_rows(df, row, row_range, *metrics):
    # 如果需要对 'spectrum radius' 或 'alpha' 选完以后再取平均，例如有多个readout class，可以使用这个函数 
    # 例如： select_rows(df_mean, 'alpha', plot_spectrum_radius, metrics)
    # groupby之后 .mean()的操作会产生浮点数误差，需要round消除掉
    df = df[df[row].isin(row_range)]\
        .groupby(row)[[row, *metrics]]\
        .mean().reset_index(drop=True)
    df = df.round({row: 2})
    return df

def save_fig(fig, fig_dir, fig_name, name_extension = '.png', transparent = False):
    if fig_name is not None:
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        # transparent = False
        fig.savefig(
            fname=os.path.join(fig_dir, fig_name + ('_transparent' if transparent else '') + name_extension), 
            transparent=transparent, 
            bbox_inches='tight', 
            dpi=330,
        )

def plot_memory_subregion_performance(models, model_type_names=('Empirical', 'Rewired', 'Rotated'), number=1000, fig_name = None, fig1_name = None):
    result_dir = 'results'
    task_dir = 'memory_task'
    model_type_tag = 'Network type'
    readout_tag = 'class'

    metrics = 'performance_valid'

    plot_spectrum_radius = {
        0.3: 'Stable', 
        0.5: 'Stable', 
        0.7: 'Stable', 
        0.8: 'Stable', 
        0.9: 'Stable',
        1.0: 'Critical',
        # *** 这里试着把一部分alpha放进来
        1.1: 'Chaotic', 
        1.2: 'Chaotic', 
        1.3: 'Chaotic', 
        1.4: 'Chaotic',
        # ***
        1.5: 'Chaotic', 
        2.0: 'Chaotic', 
        # 2.5: 'Chaotic', 
        # 3.0: 'Chaotic', 
        # 3.5: 'Chaotic',
    }
    spectrum_radius_fields = ('Stable', 'Critical', 'Chaotic')
    spectrum_radius_tag = 'Dynamics'
    df = pd.DataFrame()
    for model_dir, model_type in zip(models, model_type_names): # 对于每种网络
        df_tmp = [load_subregion_performance(i, result_dir, task_dir, model_dir, plot_spectrum_radius, spectrum_radius_tag, readout_tag, metrics) for i in range(number)]
        df_tmp = pd.concat(df_tmp).reset_index(drop=True)
        df_tmp[model_type_tag] = model_type # 加在内循环会稍微慢一些
        df = pd.concat((df, df_tmp))
    df = df.reset_index(drop=True)
    
    tag_name_mappings = {
        'performance_valid': r'Memory capacity',
        'valid accuracy': r'Multitask accuracy',
        'alpha': r'$\alpha$',
        'spectrum radius': r'$\alpha$',
    }
    df.insert(
        loc = df.shape[1], 
        column = tag_name_mappings[metrics], 
        value = (df[metrics] - df[metrics].min()) / (df[metrics].max() - df[metrics].min())
    )
    subregion_labels = ['Frontal', 'Temporal', 'Parietal', 'Occipital', 'Insular', 'Limbic']
    # subregion_short_labels = ['F', 'T', 'P', 'I', 'CG', 'Oc']
    subregion_short_labels = ['Frontal', 'Temporal', 'Parietal', 'Occipital', 'Insular', 'Limbic']
    fig = subregions_performance_fig(
        df, tag_name_mappings, 
        metrics, readout_tag, subregion_labels, 
        spectrum_radius_tag, spectrum_radius_fields, 
        model_type_tag, model_type_names
    )
    fig_1 = critical_subregions_fig(
        df, tag_name_mappings, 
        metrics, readout_tag, subregion_short_labels, 
        spectrum_radius_tag, spectrum_radius_fields, 
        model_type_tag, model_type_names
    )
    save_fig(fig, os.path.join('visualization', 'memory_task'), fig_name)
    save_fig(fig_1, os.path.join('visualization', 'memory_task'), fig1_name)
    return df

def plot_memory_performance(models, model_type_names=('Empirical', 'Rewired', 'Rotated'), number=1000, fig_name = None, fig1_name = None, normalize_01 = False, n_start = 0, spectrum_radii = 'preset'):
    # normalize_01 = False # 是否将 performance_valid 归一化到 [0,1] 区间
    metrics = 'performance_valid'
    model_type_tag = 'Network type'
    result_dir = 'results'
    task_dir = 'memory_task'
    if spectrum_radii == 'preset':
        plot_spectrum_radius = [
            0.3, 0.5, 0.7, 0.8, 0.9, 
            0.95, 
            1.0, 
            1.05, 
            1.1, 1.2, 1.3, 1.4, 1.5, 
            2.0, 2.5, 3.0, 3.5
        ]
        v = 'alpha'
    elif spectrum_radii == 'post':
        plot_spectrum_radius = None
        v = r'$\ln\alpha$'
    df = pd.DataFrame()
    for model_dir, model_type in zip(models, model_type_names):
        df_tmp = [load_performance(i, result_dir, task_dir, model_dir, plot_spectrum_radius, metrics) for i in range(n_start, n_start + number)]
        # df_tmp = [load_performance(i, result_dir, task_dir, models[0], plot_spectrum_radius, metrics) for i in range(1, 3)]
        df_tmp = pd.concat(df_tmp).reset_index(drop=True)
        df_tmp[model_type_tag] = model_type
        df = pd.concat([df, df_tmp])
    df = df.reset_index(drop=True)
    if spectrum_radii == 'post':
        df[v] = np.log(df['alpha'])
    tag_name_mappings = {
        'performance_valid': r'Memory capacity',
        'alpha': r'$\alpha$',
        'spectrum radius': r'$\alpha$',
        r'$\ln\alpha$': r'$\ln\alpha$',
    }
    fig = dynamic_performance_fig(df.copy(), tag_name_mappings, metrics, model_type_tag, normalize_01, variable=v)
    fig_1 = critical_distribution_fig(df, metrics, model_type_tag, model_type_names, normalize_01)
    
    save_fig(fig, os.path.join('visualization', 'memory_task'), fig_name)
    save_fig(fig_1, os.path.join('visualization', 'memory_task'), fig1_name)
    return df

# def plot_multitask_performance(models, model_type_names=('Empirical', 'Rewired', 'Rotated'), number=100, start_n=0, fig_name = None):
#     result_dir = 'results'
#     task_dir = 'multitask14'
#     model_type_tag = 'Network type'
#     normalize = 0.5 # 显示的范围是 0.5, 1.0

#     metrics = 'valid accuracy'

#     plot_spectrum_radius = [
#         0.3, 0.5, 0.7, 0.8, 0.9, 
#         0.95, 
#         1.0, 
#         1.05, 
#         1.1, 1.2, 1.3, 1.4, 1.5, 
#         2.0, 2.5, 3.0, 3.5
#     ]
#     df_mean = pd.DataFrame()
#     df_all = pd.DataFrame()
#     for model_dir, model_type in zip(models, model_type_names):
#         dfs = [load_multitask_df(i, result_dir, task_dir, model_dir) for i in range(start_n, start_n+number)]
#         df_tmp_mean = [d[0] for d in dfs]
#         df_tmp_all = [d[1] for d in dfs]

#         df_tmp_mean = pd.concat(df_tmp_mean).reset_index(drop=True)
#         df_tmp_mean[model_type_tag] = model_type
#         df_mean = pd.concat([df_mean, df_tmp_mean])

#         df_tmp_all = pd.concat(df_tmp_all).reset_index(drop=True)
#         df_tmp_all[model_type_tag] = model_type
#         df_all = pd.concat([df_all, df_tmp_all])
#     df_mean = df_mean.reset_index(drop=True)
#     df_all = df_all.reset_index(drop=True)

#     df_mean = df_mean[df_mean['spectrum radius'].isin(plot_spectrum_radius)]
#     df_all = df_all[df_all['spectrum radius'].isin(plot_spectrum_radius)]
    
#     tag_name_mappings = {
#         'valid accuracy': r'Multitask accuracy',
#         'alpha': r'$\alpha$',
#         'spectrum radius': r'$\alpha$',
#     }
#     fig = dynamic_performance_fig(df_mean.copy(), tag_name_mappings, metrics, model_type_tag, normalize)
#     # fig1 = dynamic_performance_fig(df_all.copy(), metrics, model_type_tag, normalize_01)
    
#     save_fig(fig, os.path.join('visualization', 'multitask14'), fig_name)

def plot_distweight_ws_performance(number=1000, fig_name=None):
    c_id = [9, 44, 46]
    models = [
        f'BNARC1065_{c_id[0]}_{c_id[0]}', 
        f'WS_BN_1065sub_binary_75pct_density_{c_id[1]}_{c_id[1]}', 
        f'WS_BN_1065sub_binary_75pct_density_shuffle_{c_id[2]}_{c_id[2]}',
    ]
    model_type_names=('BNA', 'WS', 'WS shuffle')
    metrics = 'performance_valid'
    model_type_tag = 'Network type'
    
    result_dir = 'results'
    task_dir = 'memory_task'
    plot_spectrum_radius = [
        0.3, 0.5, 0.7, 0.8, 0.9, 
        0.95, 
        1.0, 
        1.05, 
        1.1, 1.2, 1.3, 1.4, 1.5, 
        2.0, 2.5, 3.0, 3.5
    ]
    df = pd.DataFrame()
    for model_dir, model_type in zip(models, model_type_names):
        df_tmp = [load_performance(i, result_dir, task_dir, model_dir, plot_spectrum_radius, metrics) for i in range(number)]
        # df_tmp = [load_performance(i, result_dir, task_dir, models[0], plot_spectrum_radius, metrics) for i in range(1, 3)]
        df_tmp = pd.concat(df_tmp).reset_index(drop=True)
        df_tmp[model_type_tag] = model_type
        df = pd.concat([df, df_tmp])
    df = df.reset_index(drop=True)

    c = Connectome()
    dist = c.wighted_eucidean_distance(c_id[0]), \
        np.mean([c.wighted_eucidean_distance(c_id[1], i) for i in range(number)]), \
        np.mean([c.wighted_eucidean_distance(c_id[2], i) for i in range(number)]),

    dist_mapping = dict(zip(model_type_names, dist))

    tag_name_mappings = {
        'performance_valid': r'$\frac{\mathrm{Memory\ capacity}}{\mathrm{Eucidean\ distance}}$',
        'valid accuracy': r'Multitask accuracy',
        'alpha': r'$\alpha$',
        'spectrum radius': r'$\alpha$',
    }
    df[metrics] = df[metrics] / df[model_type_tag].apply(dist_mapping.get)

    fig = dynamic_performance_fig(df.copy(), tag_name_mappings, metrics, model_type_tag, True)
    save_fig(fig, os.path.join('visualization', 'memory_task'), fig_name)

if __name__ == '__main__':
    1
    # 使用代码 生成fig2
    # models = ['BNARCbinary_1_1', 'BNARCrewired_31_31', 'BNARCrotation_1_1']
    # plot_memory_performance(models, number=1000, fig_name='BNARCbinary_1000trials', normalize_01=False)
    # plot_memory_performance(models, number=1000, fig_name='BNARCbinary_1000trials_normalize', normalize_01=True)

    # models = ['BNARC1065_9_9', 'BNARC1065rewired_39_39', 'BNARC1065rotation_9_9']
    # plot_memory_performance(models, number=1000, fig_name='BNARC1065binary_1000trials', normalize_01=False)
    # plot_memory_performance(models, number=1000, fig_name='BNARC1065binary_1000trials_normalize', normalize_01=True)

    # models = ['BNARCbinary_1_1', 'ER_BNA_binary_density_41_41', 'WS_BNA_binary_density_42_42']
    # type_name = ('BNA', 'ER', 'WS')
    # plot_memory_performance(models, type_name, number=1000, fig_name='BNARCbinary-random_1000trials')

    # models = ['BNARC1065_9_9', 'ER_BN_1065sub_binary_75pct_density_43_43', 'WS_BN_1065sub_binary_75pct_density_44_44']
    # type_name = ('BNA', 'ER', 'WS')
    # plot_memory_performance(models, type_name, number=1000, fig_name='BNARC1065binary-random_1000trials')