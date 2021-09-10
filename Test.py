import os
import time
import shutil
# import random
# from collections import Iterable
import numpy as np
import pandas as pd
from PIL import Image
import cv2
# import matplotlib.image as pltimg
# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from Lib_Class import ImageData
# from sklearn.decomposition import PCA
# from sklearn import manifold
# import skimage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def test(filename,chir_reference):
    filename = "CD13_fd.csv"
    chir_reference = 6  # the best CHIR concentration under condition CHIR_hour = 24

    fd_frame = pd.read_csv(filename, index_col=0)

    X, y = [], []

    chir = []

    for i in range(1, 97):
        name = "S" + str(i)

        feat = []
        for i in range(1, 11):
            FD = fd_frame.loc[name]["t%d" % i]
            feat.append(FD)

        label = fd_frame.loc[name]["CHIR"]
        chir.append(label - chir_reference)
        if (label < chir_reference):
            label = -1
        elif (label > chir_reference):
            label = 1
        else:
            label = 0
        X.append(feat)
        y.append(label)

    #### Visualize samples in the feature space, as long as their relative CHIR concentration

    X = np.array(X)
    y = np.array(y)
    chir = np.array(chir)

    X = (X - X.mean(axis=0)) / X.std(axis=0)  # normalize the features.

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r = lda.fit(X, y).transform(X)  # Reduce the dimension to 2

    plt.axes(facecolor="grey")
    plt.scatter(X_r[:, 0], X_r[:, 1], s=8, c=chir, cmap=plt.cm.bwr)
    cbar = plt.colorbar()
    plt.title("Result (linear discriminative analysis)")
    cbar.ax.set_title("CHIR - %d" % chir_reference)

    plt.savefig("%s_result.pdf" % filename)
    plt.show()


if __name__ == '__main__':
    # main program entrance, using for test one function
    print('!Notice! This is NOT the main function running!')
    print('Only TESTING!')

    imgae_path = r'C:\C137\MATLAB_Code\Random_Walker\CD13_img\2018-12-10~III-1_CD13~T43.jpg'
    output_image = r'C:\C137\MATLAB_Code\Random_Walker\CD13_img\2018-12-10~III-1_CD13~T43_mask.jpg'
    this_image = ImageData(imgae_path, 0)
    this_image_mask = this_image.getCellMask()
    cv2.imwrite(output_image, this_image_mask)
