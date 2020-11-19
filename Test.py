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


if __name__ == '__main__':
    # main program entrance, using for test one function
    print('!Notice! This is NOT the main function running!')
    print('Only TESTING!')


    imgae_path=r'C:\C137\MATLAB_Code\Random_Walker\CD13_img\2018-12-10~III-1_CD13~T43.jpg'
    output_image=r'C:\C137\MATLAB_Code\Random_Walker\CD13_img\2018-12-10~III-1_CD13~T43_mask.jpg'
    this_image = ImageData(imgae_path, 0)
    this_image_mask = this_image.getCellMask()
    cv2.imwrite(output_image,this_image_mask)
