from matplotlib import style
import numpy as np
# import torch
from sklearn.linear_model import LinearRegression
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import label_binarize
import seaborn as sns

import os
from data_connectome import Connectome


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
