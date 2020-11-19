import os
import sys
import math
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from sklearn.utils import check_random_state

# Next line to silence pyflakes. This import is needed.
# Axes3D

local_path = 'C:\\KittyDocuments\\Matlab_Processing\\pre_big\\'
# pcaScores_csv = pd.read_csv(os.path.join(local_path, 'pcaScores.csv'), header=None, index_col=None)
siftData_csv = pd.read_csv(os.path.join(local_path, 'new_sift_data.csv'), header=None, index_col=None)
color_csv = pd.read_csv(os.path.join(local_path, 'color_c.csv'), header=None, index_col=None)
label_txt = pd.read_csv(os.path.join(local_path, 'data_labels.txt'), header=None, index_col=None)

n_points = siftData_csv.shape[0]
# X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
n_neighbors = 100
n_components = 10

X = siftData_csv.values
color = color_csv.values.flatten()
label_c = label_txt.values.flatten()

# fig = plt.figure()
# plt.suptitle("Manifold Learning with %i points, %i neighbors" % (n_points, n_neighbors), fontsize=14)

# ax = fig
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral, label=label_c)
# ax.view_init(4, -72)

t0 = time.time()
Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
Y_df = pd.DataFrame(Y)
Y_df.to_csv(path_or_buf=os.path.join(local_path, 'Isomap.csv'), header=None, index=None)
# t1 = time.time()
# print("Isomap: %.2g sec" % (t1 - t0))
# ax = fig.add_subplot(257)
# plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, label=label_c)
# plt.title("Isomap (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# # ax.legend()
# plt.axis('tight')