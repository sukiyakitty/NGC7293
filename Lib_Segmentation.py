import os
import time
import shutil
import numpy as np
import pandas as pd
# import sklearnaeae
from sklearn.cluster import KMeans
import cv2
from matplotlib import pyplot as plt
from Lib_Function import any_to_image, image_to_gray, python_busy_sleep, get_time, prevent_disk_died
from Lib_Class import ImageData
from Lib_Sort import files_sort_CD09, files_sort_CD11, files_sort_CD13, files_sort_CD26, files_sort_CD27, \
    files_sort_CD39, files_sort_CD41


def call_gabor_analysis(infile, outfile, k, gk, M):
    # call gabor analysis
    os.system("start /min python gabor.py -infile {} -outfile {} -k {} -gk {} -M {}".format(infile, outfile, k, gk, M))
    # os.system("python3 gabor.py -infile {} -outfile {} -k {} -gk {} -M {}".format(infile, outfile, k, gk, M))
    return True


def return_color():
    m = []  # BGR
    m.append([0, 0, 255])
    m.append([0, 255, 0])
    m.append([255, 0, 0])
    m.append([255, 255, 0])
    m.append([255, 0, 255])
    m.append([0, 255, 255])
    m = np.array(m, dtype=np.uint8)
    return m


def image_open_operation(input_file, output_file, gap=23, morph=cv2.MORPH_ELLIPSE):
    # for image open operation
    img = any_to_image(input_file)
    if img is None:
        return None

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap, gap))
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap))
    kernel = cv2.getStructuringElement(morph, (gap, gap))
    output_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    cv2.imwrite(output_file, output_img)

    return True


def folder_image_open_operation(input_path, to_path, files_sort_function, gap=23, morph=cv2.MORPH_ELLIPSE):
    if not os.path.exists(input_path):
        print('!ERROR! The input_path does not existed!')
        return False
    if not os.path.exists(to_path):
        os.makedirs(to_path)

    files = os.listdir(input_path)
    files = files_sort_function(files)
    for f in files:
        infile = os.path.join(input_path, f)
        outfile = os.path.join(to_path, f)
        image_open_operation(infile, outfile, gap=gap, morph=morph)

    return True


def getMean(img_gray):
    temp = img_gray.ravel()
    return [np.mean(temp), np.std(temp)]


def getSIFT(img_gray):
    sift = cv2.xfeatures2d.SIFT_create()
    try:
        noneuse, SIFTdes = sift.detectAndCompute(img_gray, None)
        siftFeatureVector = np.append(np.std(SIFTdes, axis=0, ddof=1), np.mean(SIFTdes, axis=0))
    except BaseException as e:
        # print('!ERROR! ', e)
        siftFeatureVector = np.zeros(256, dtype=np.float64)
    return siftFeatureVector


def getSURF(img_gray):
    surf = cv2.xfeatures2d.SURF_create()
    try:
        noneuse, SURFdes = surf.detectAndCompute(img_gray, None)
        surfFeatureVector = np.append(np.std(SURFdes, axis=0, ddof=1), np.mean(SURFdes, axis=0))
    except BaseException as e:
        # print('!ERROR! ', e)
        surfFeatureVector = np.zeros(128, dtype=np.float64)
    return surfFeatureVector


def getORB(img_gray):
    orb = cv2.ORB_create()
    try:
        noneuse, ORBdes = orb.detectAndCompute(img_gray, None)
        orbFeatureVector = np.append(np.std(ORBdes, axis=0, ddof=1), np.mean(ORBdes, axis=0))
    except BaseException as e:
        # print('!ERROR! ', e)
        orbFeatureVector = np.zeros(64, dtype=np.float64)
    return orbFeatureVector


def get_KMeans(data, n_clusters=3):
    label = KMeans(n_clusters=n_clusters).fit_predict(data)
    return label


def get_Bright_Dark(data, n_clusters=3):
    label = np.ones(data.shape[0]) * 128
    label[data[:, 0] > 192] = 255
    label[data[:, 0] < 128] = 0
    return np.array(label, dtype=np.uint8)


def color_img_segmentation(img, n_clusters=5, show=False):
    # according to the color of img do KMeans

    img = any_to_image(img)
    # img_gray = image_to_gray(img)
    if len(img.shape) == 3:
        pass
    elif len(img.shape) == 2:
        # img_gray = img
        print('!NOTICE! The input image is gray!')
        # pass
        return None

    img_float = img / 255
    img_float = img_float.flatten()
    img_float = img_float.reshape((img.shape[0] * img.shape[1], 3))

    label = KMeans(n_clusters=n_clusters).fit_predict(img_float)
    label = label.reshape(img.shape[:2])
    label = np.array(255 / (label + 1), dtype=np.uint8)

    if show:
        cv2.imshow('color_img_segmentation()', label)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return True


def get_image_features_and_segementation(main_path, sigle_image, kernel=(69, 69), pix=(23, 23), getF=getSIFT,
                                         getC=get_KMeans, n_clusters=3, result_path='pix_features', save=True,
                                         save_name=None, show=False):
    # this program is design for core analysis; one time one well one time point one image
    # for block feature extraction and cluster
    # kernel is feature extraction unit, the window
    # pix is compressed image block, the Super pixel
    # kernel >= pix
    # now it is doing feature extraction
    # input sigle_image has well edge, is 5*5
    # sigle_image is a image path
    # kernel or pix is (w,h)(x,y)

    if save:
        result_path = os.path.join(main_path, result_path)
        if os.path.exists(result_path):
            # shutil.rmtree(output_folder)
            pass
        else:
            os.makedirs(result_path)  # make the output folder

    if type(sigle_image) is str:
        sigle_image = os.path.join(main_path, sigle_image)
        if save_name is None:
            save_name = os.path.split(sigle_image)[1][:-4]

    img = image_to_gray(sigle_image)

    if show:
        print(img.shape)
        print(img[int(img.shape[0] / 2):int(img.shape[0] / 2 + img.shape[0] / 10),
              int(img.shape[1] / 2):int(img.shape[1] / 2 + img.shape[1] / 10)])
        cv2.imshow('image_to_gray()', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if img is None:
        return None
    pix_h = img.shape[0]  # y:height,row
    pix_w = img.shape[1]  # x:width,col
    compress_pix_w = int((pix_w - kernel[0] + pix[0]) / pix[0])  # x:width,col
    compress_pix_h = int((pix_h - kernel[1] + pix[1]) / pix[1])  # y:height,row
    offset_x = int((kernel[0] - pix[0]) / 2)  # x:width,col
    offset_y = int((kernel[1] - pix[1]) / 2)  # y:height,row
    # used_pix_w = compress_pix_w * pix[0] - pix[0] + kernel[0]
    # used_pix_h = compress_pix_h * pix[1] - pix[1] + kernel[1]
    # used_img = img[:used_pix_h, :used_pix_w, :]

    if show:
        print('compress_pix_w = ', compress_pix_w)
        print('compress_pix_h = ', compress_pix_h)

    pix_features_list = []  # is a col of
    for row in range(compress_pix_h):
        for col in range(compress_pix_w):
            stamp = img[row * pix[1]:row * pix[1] + kernel[1], col * pix[0]:col * pix[0] + kernel[0]]
            # if show:
            #     cv2.imshow('steps(' + str(row) + ',' + str(col) + ')', stamp)
            #     cv2.waitKey()
            #     cv2.destroyAllWindows()
            pix_features_list.append(getF(stamp))

    pix_features_list = np.array(pix_features_list)
    pix_features_list = np.nan_to_num(pix_features_list)
    # print(pix_features_list)
    # print(pix_features_list.shape)
    label = getC(pix_features_list, n_clusters=n_clusters)
    # label = pix_features_list[:,0]
    label_map = label.reshape((compress_pix_h, compress_pix_w))
    # label_map = np.array(255 / (label_map + 1), dtype=np.uint8)
    label_map_colored = np.zeros((compress_pix_h, compress_pix_w, 3), dtype=np.uint8)
    color_key = return_color()
    for i in range(compress_pix_h):
        for j in range(compress_pix_w):
            label_map_colored[i, j] = color_key[int(label_map[i, j])]

    # cv2.resize(img_cpy, (compress_pix_w*pix[0], compress_pix_h*pix[1]), interpolation=cv2.INTER_NEAREST)
    label_map_big = cv2.resize(label_map_colored, (0, 0), fx=pix[0], fy=pix[1], interpolation=cv2.INTER_NEAREST)
    print(label_map_big.shape)
    label_result = np.ones((pix_h, pix_w, 3), dtype=np.uint8) * 255
    print(label_result.shape)
    label_result[offset_y:offset_y + compress_pix_h * pix[1],
    offset_x:offset_x + compress_pix_w * pix[0], :] = label_map_big

    if save:
        if save_name is None:
            save_name = 'result.png'
        result_file = os.path.join(result_path, save_name + '.png')
        cv2.imwrite(result_file, label_result)

        outfile_csv = os.path.join(result_path, save_name + '.csv')
        # save_csv(featureVectors, outfile_csv)
        featureVectors = pd.DataFrame(pix_features_list)
        featureVectors.to_csv(path_or_buf=outfile_csv)

    if show:
        cv2.imshow('get_img_pix_features_list()', label_result)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return True


def get_well_features_and_segementation_using_gabor(input_path, to_path, k, gk, M, files_sort_function, max_p, jump=3):
    # using gabor features
    # return and save features list and segementation image Parallel
    # gabor input:::
    # k: Number of clusters
    # gk: Size of the gabor kernel
    # M: Size of the gaussian window

    if not os.path.exists(input_path):
        print('!ERROR! The input_path does not existed!')
        return False
    if not os.path.exists(to_path):
        os.makedirs(to_path)

    files = os.listdir(input_path)
    files = files_sort_function(files)
    this_j = 0
    for f in files:
        this_j = this_j + 1
        if this_j == jump:  # do
            infile = os.path.join(input_path, f)
            outfile = os.path.join(to_path, f)
            call_gabor_analysis(infile, outfile, k, gk, M)
            python_busy_sleep(max_p)
            this_j = 0

    return True


def get_pic_features_and_segementation_using_gabor(input_path, to_path, image_name, k, gk, M, max_p):
    # using gabor features
    # return and save features list and segementation image Parallel
    # input_path: example : r'C:\C137\PROCESSING\CD13\MyPGC_img\SSS_100%'
    # to_path: example : r'C:\C137\PROCESSING\CD13\garbor'
    # gabor input:::
    # k: Number of clusters
    # gk: Size of the gabor kernel
    # M: Size of the gaussian window

    if not os.path.exists(input_path):
        print('!ERROR! The input_path does not existed!')
        return False
    if not os.path.exists(to_path):
        os.makedirs(to_path)

    S_folders = os.listdir(input_path)
    for this_S in S_folders:
        S_folder = os.path.join(input_path, this_S)
        files = os.listdir(S_folder)
        for this_f in files:
            if this_f == image_name:
                f_image = os.path.join(S_folder, this_f)
                out_S = os.path.join(to_path, this_S)
                if not os.path.exists(out_S):
                    os.makedirs(out_S)
                outfile = os.path.join(to_path, this_S, this_f)
                call_gabor_analysis(f_image, outfile, k, gk, M)
                python_busy_sleep(max_p)

    return True


def merge_folder_csv(input_path, to_file, files_sort_function, save=False):
    if not os.path.exists(input_path):
        print('!ERROR! The input_path does not existed!')
        return False
    if save:
        to_path = os.path.split(to_file)[0]
        if not os.path.exists(to_path):
            os.makedirs(to_path)
        if os.path.exists(to_file):
            os.remove(to_file)

    all_mem = None
    files = os.listdir(input_path)
    files = files_sort_function(files)
    for f in files:
        if f.split('.')[1] == 'csv':
            each_csv_file = os.path.join(input_path, f)
            get_time()
            print('!NOTICE! Reading file: ', each_csv_file)
            if all_mem is None:
                all_mem = pd.read_csv(each_csv_file, header=0, index_col=0)
            else:
                each_mem = pd.read_csv(each_csv_file, header=0, index_col=0)
                # all_mem = all_mem.append(each_mem)
                all_mem = pd.concat([all_mem, each_mem], ignore_index=True)
                print('--->>> merging: ' + f)

    if save:
        all_mem.to_csv(path_or_buf=to_file)

    return all_mem


def painting_well_segmentation(labels, in_folder, out_folder, files_sort_function, safe_time=5):
    if not os.path.exists(in_folder):
        print('!ERROR! The in_folder does not existed!')
        return False
    # time.sleep(safe_time)
    if not os.path.exists(out_folder):
        # shutil.rmtree(out_folder)
        time.sleep(safe_time)
        os.makedirs(out_folder)
        time.sleep(safe_time)

    img_type = ['jpg', 'png', 'tif']
    offset = 0
    in_files = os.listdir(in_folder)
    in_files = files_sort_function(in_files)
    for f in in_files:
        if f.split('.')[1] in img_type:
            in_img_f = os.path.join(in_folder, f)
            in_img = image_to_gray(in_img_f)
            pixel = in_img.shape[0] * in_img.shape[1]
            this_labels = labels[offset:offset + pixel]
            offset = offset + pixel
            out_file = os.path.join(out_folder, f)
            plt.imsave(out_file, this_labels.reshape(in_img.shape))

    return True


def done_well_sagementation_using_gabor(main_path, in_folder, out_folder, files_sort_function, k, gk, M, max_p, jump):
    # in_folder: r'C:\C137\PROCESSING\CD09\auto_binary\SSS_100%\S3'
    # out_folder: r'C:\C137\PROCESSING\CD09\Seg_gabor\SSS_100%\S3'
    # gabor input:::
    # k: Number of clusters
    # gk: Size of the gabor kernel
    # M: Size of the gaussian window
    #

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(in_folder):
        print('!ERROR! The in_folder does not existed!')
        return False
    if not os.path.exists(out_folder):
        # shutil.rmtree(out_folder)
        os.makedirs(out_folder)

    featrues_path = os.path.join(out_folder, 'featrues')
    get_time()
    print('!NOTICE! Now doing features extractiong using gabor')
    get_well_features_and_segementation_using_gabor(in_folder, featrues_path, k, gk, M, files_sort_function, max_p,
                                                    jump=jump)
    time.sleep(10)
    python_busy_sleep(2)

    to_file = os.path.join(out_folder, 'ALL_FeatureVectors.csv')
    get_time()
    print('!NOTICE! Now doing .csv merge')
    all_mem = merge_folder_csv(featrues_path, to_file, files_sort_function, save=False)

    get_time()
    print('!NOTICE! Now do_clust')
    labels = get_KMeans(all_mem, n_clusters=k)

    get_time()
    print('!NOTICE! Now clust_painting')
    painting_well_segmentation(labels, featrues_path, os.path.join(out_folder, 'segmentation'), files_sort_function,
                               safe_time=3)

    return True


def return_mask(main_path, sigle_image, kernel=(69, 69), pix=(23, 23), getF=getMean, getC=get_Bright_Dark,
                n_clusters=3, result_path='pix_features', save=True, save_name=None, show=False):
    # this program is design for core analysis; one time one well one time point one image
    # for block feature extraction and cluster
    # kernel is feature extraction unit, the window
    # pix is compressed image block, the Super pixel
    # kernel >= pix
    # now it is doing feature extraction
    # input sigle_image has well edge, is 5*5
    # sigle_image is a image path
    # kernel or pix is (w,h)(x,y)

    if save:
        result_path = os.path.join(main_path, result_path)
        if os.path.exists(result_path):
            # shutil.rmtree(output_folder)
            pass
        else:
            os.makedirs(result_path)  # make the output folder

    if type(sigle_image) is str:
        sigle_image = os.path.join(main_path, sigle_image)
        if save_name is None:
            save_name = os.path.split(sigle_image)[1]

    img = image_to_gray(sigle_image)

    if show:
        print(img.shape)
        print(img[int(img.shape[0] / 2):int(img.shape[0] / 2 + img.shape[0] / 10),
              int(img.shape[1] / 2):int(img.shape[1] / 2 + img.shape[1] / 10)])
        cv2.imshow('image_to_gray()', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if img is None:
        return None
    pix_h = img.shape[0]  # y:height,row
    pix_w = img.shape[1]  # x:width,col
    compress_pix_w = int((pix_w - kernel[0] + pix[0]) / pix[0])  # x:width,col
    compress_pix_h = int((pix_h - kernel[1] + pix[1]) / pix[1])  # y:height,row
    offset_x = int((kernel[0] - pix[0]) / 2)  # x:width,col
    offset_y = int((kernel[1] - pix[1]) / 2)  # y:height,row
    # used_pix_w = compress_pix_w * pix[0] - pix[0] + kernel[0]
    # used_pix_h = compress_pix_h * pix[1] - pix[1] + kernel[1]
    # used_img = img[:used_pix_h, :used_pix_w, :]

    if show:
        print('compress_pix_w = ', compress_pix_w)
        print('compress_pix_h = ', compress_pix_h)

    pix_features_list = []  # is a col of
    for row in range(compress_pix_h):
        for col in range(compress_pix_w):
            stamp = img[row * pix[1]:row * pix[1] + kernel[1], col * pix[0]:col * pix[0] + kernel[0]]
            # if show:
            #     cv2.imshow('steps(' + str(row) + ',' + str(col) + ')', stamp)
            #     cv2.waitKey()
            #     cv2.destroyAllWindows()
            pix_features_list.append(getF(stamp))

    pix_features_list = np.array(pix_features_list)
    pix_features_list = np.nan_to_num(pix_features_list)
    # print(pix_features_list)
    # print(pix_features_list.shape)
    label = getC(pix_features_list, n_clusters=n_clusters)
    # label = pix_features_list[:,0]
    label_map = label.reshape((compress_pix_h, compress_pix_w))
    # label_map = np.array(255 / (label_map + 1), dtype=np.uint8)
    # label_map_colored = np.zeros((compress_pix_h, compress_pix_w, 3), dtype=np.uint8)
    # color_key = return_color()
    # for i in range(compress_pix_h):
    #     for j in range(compress_pix_w):
    #         label_map_colored[i, j] = color_key[int(label_map[i, j])]

    # cv2.resize(img_cpy, (compress_pix_w*pix[0], compress_pix_h*pix[1]), interpolation=cv2.INTER_NEAREST)
    label_map_big = cv2.resize(label_map, (0, 0), fx=pix[0], fy=pix[1], interpolation=cv2.INTER_NEAREST)
    print(label_map_big.shape)
    label_result = np.ones((pix_h, pix_w), dtype=np.uint8) * 128
    print(label_result.shape)
    label_result[offset_y:offset_y + compress_pix_h * pix[1],
    offset_x:offset_x + compress_pix_w * pix[0]] = label_map_big

    if save:
        if save_name is None:
            save_name = 'result.png'
        result_file = os.path.join(result_path, save_name)
        cv2.imwrite(result_file, label_result)

    if show:
        cv2.imshow('get_img_pix_features_list()', label_result)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return False


# def load_data(file_path):
#     f = open(file_path, 'rb')  # 二进制打开
#     data = []
#     img = image.open(f)  # 以列表形式返回图片像素值
#     m, n = img.size  # 活的图片大小
#     for i in range(m):
#         for j in range(n):  # 将每个像素点RGB颜色处理到0-1范围内并存放data
#             x, y, z = img.getpixel((i, j))
#             data.append([x / 256.0, y / 256.0, z / 256.0])
#     f.close()
#     return np.mat(data), m, n  # 以矩阵型式返回data，图片大小
#
#
# img_data, row, col = load_data(r'C:\Users\Kitty\Documents\Desktop\a\2019-11-13~Exp1_Day1~T1.png')
# label = sklearn.cluster.KMeans(n_clusters=3).fit_predict(img_data)  # 聚类中心的个数为3
# label = label.reshape([row, col])  # 聚类获得每个像素所属的类别
# pic_new = image.new("L", (row, col))  # 创建一张新的灰度图保存聚类后的结果
# for i in range(row):  # 根据所属类别向图片中添加灰度值
#     for j in range(col):
#         pic_new.putpixel((i, j), int(256 / (label[i][j] + 1)))
# pic_new.save(r'C:\Users\Kitty\Documents\Desktop\a\1.jpg')


def temp_image_folder_resize(input_f, output_f, pixel=(2160, 2160), box=None):
    # pixel = (w, h)
    # box = (x0,y0,x1,y1)

    if not os.path.exists(input_f):
        print('!ERROR! The input_f does not existed!')
        return False
    if not os.path.exists(output_f):
        os.makedirs(output_f)

    path_list = os.listdir(input_f)
    for img_name in path_list:
        img = os.path.join(input_f, img_name)
        img = image_to_gray(img)

        if box is not None:
            img = img[box[1]:box[3], box[0]:box[2]]

        h, w = img.shape[:2]
        if h != pixel[1] or w != pixel[0]:
            img = cv2.resize(img, pixel, interpolation=cv2.INTER_NEAREST)

        to_file = os.path.join(output_f, img_name)
        cv2.imwrite(to_file, img)

    return True


def get_well_difference(input_f, output_f, sort_function):
    # calculate adjacent pictures' difference of one folder
    # notice: it must giving the sort function

    # if not os.path.exists(main_path):
    #     print('!ERROR! The main_path does not existed!')
    #     return False
    # folder = os.path.join(main_path, folder)
    if not os.path.exists(input_f):
        print('!ERROR! The input_f does not existed!')
        return False
    # result_path = os.path.join(main_path, result_path)
    if not os.path.exists(output_f):
        os.makedirs(output_f)

    path_list = os.listdir(input_f)
    path_list = sort_function(path_list)
    for i in range(len(path_list) - 1):
        img1 = os.path.join(input_f, path_list[i])
        img1 = image_to_gray(img1)
        img2 = os.path.join(input_f, path_list[i + 1])
        img2 = image_to_gray(img2)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if h1 != h2 or w1 != w2:
            img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_NEAREST)

        result = img1 - img2
        result[result < 0] = 0
        # result = abs(result)

        this_range = np.max(result) - np.min(result)
        result = (result - np.min(result)) / this_range
        result = result * 255

        to_file = os.path.join(output_f, path_list[i])
        cv2.imwrite(to_file, result)

    return True


def get_image_image_sub(main_path, img1, img2, result_path='image_sub', name='sub.png', save=True, show=False):
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    result_path = os.path.join(main_path, result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    img1 = image_to_gray(img1)
    img2 = image_to_gray(img2)

    result = img1 - img2
    result[result < 0] = 0
    # result = abs(result)

    if save:
        # plt.savefig(os.path.join(result_path, name))
        to_file = os.path.join(result_path, name)
        cv2.imwrite(to_file, result)
    if show:
        cv2.imshow('image_sub', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return True


def get_folder_negative_film_V1(img1, img2, sub_folder, result_path, sort_function, save=True, show=False):
    # img1 sub every image in sub_folder, and save to result_path

    # if not os.path.exists(main_path):
    #     print('!ERROR! The main_path does not existed!')
    #     return False
    # sub_folder = os.path.join(main_path, sub_folder)
    if not os.path.exists(sub_folder):
        print('!ERROR! The sub_folder does not existed!')
        return False
    # result_path = os.path.join(main_path, result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    img1 = image_to_gray(img1)
    h1, w1 = img1.shape[:2]

    path_list = os.listdir(sub_folder)
    sort_function(path_list)
    for img2_name in path_list:
        img2 = os.path.join(sub_folder, img2_name)
        print('>>> get_folder_negative_film_V1(', img2, ')')
        img2 = image_to_gray(img2)
        h2, w2 = img2.shape[:2]

        if h1 != h2 or w1 != w2:
            img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_NEAREST)

        result = img1 - img2
        result[result < 0] = 0
        # result = abs(result)

        if save:
            # plt.savefig(os.path.join(result_path, name))
            to_file = os.path.join(result_path, img2_name)
            cv2.imwrite(to_file, result)
        if show:
            cv2.imshow('image_sub', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return True


def get_folder_negative_film_V2(img1, img2, sub_folder, result_path, sort_function, save=True, show=False):
    # img is the Brightest value of img1 and the Darkest value of img2
    # img sub every image in sub_folder, and save to result_path

    if not os.path.exists(sub_folder):
        print('!ERROR! The sub_folder does not existed!')
        return False
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    img1 = image_to_gray(img1)
    img2 = image_to_gray(img2)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 != h2 or w1 != w2:
        img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_NEAREST)

    img = img2
    x = (img1 >= 128) & (img2 >= 128) & (img1 >= img2)
    img[x] = img1[x]

    path_list = os.listdir(sub_folder)
    sort_function(path_list)
    for this_img_name in path_list:
        this_img = os.path.join(sub_folder, this_img_name)
        this_img = image_to_gray(this_img)
        this_h, this_w = this_img.shape[:2]

        if h1 != this_h or w1 != this_w:
            this_img = cv2.resize(this_img, (w1, h1), interpolation=cv2.INTER_NEAREST)

        result = img - this_img
        result[result < 0] = 0
        # result = abs(result)

        if save:
            # plt.savefig(os.path.join(result_path, name))
            to_file = os.path.join(result_path, this_img_name)
            cv2.imwrite(to_file, result)
        if show:
            cv2.imshow('image_sub', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return True


def get_folder_negative_film_V3(img1, img2, sub_folder, result_path, sort_function, save=True, show=False):
    # the copy of V2
    # img is the Brightest value of img1 and the Darkest value of img2
    # img sub every image in sub_folder, and save to result_path

    if not os.path.exists(sub_folder):
        print('!ERROR! The sub_folder does not existed!')
        return False
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    img1 = image_to_gray(img1)
    img2 = image_to_gray(img2)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 != h2 or w1 != w2:
        img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_NEAREST)

    img = img2
    x = (img1 >= 128) & (img2 >= 128) & (img1 >= img2)
    img[x] = img1[x]

    path_list = os.listdir(sub_folder)
    sort_function(path_list)
    for this_img_name in path_list:
        this_img = os.path.join(sub_folder, this_img_name)
        this_img = image_to_gray(this_img)
        this_h, this_w = this_img.shape[:2]

        if h1 != this_h or w1 != this_w:
            this_img = cv2.resize(this_img, (w1, h1), interpolation=cv2.INTER_NEAREST)

        result = img - this_img
        result[result < 0] = 0
        # result = abs(result)

        if save:
            # plt.savefig(os.path.join(result_path, name))
            to_file = os.path.join(result_path, this_img_name)
            cv2.imwrite(to_file, result)
        if show:
            cv2.imshow('image_sub', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return True


def get_image_hist(main_path, well_image, result_path='Hist', name='Image_Hist.png', save=True, show=False):
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    result_path = os.path.join(main_path, result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    img = image_to_gray(well_image)

    h, w = img.shape[:2]
    pixelSequence = img.reshape([h * w, ])
    histogram, bins, patch = plt.hist(pixelSequence, 256, facecolor='black', histtype='bar')
    plt.xlabel("gray label")
    plt.ylabel("number of pixels")
    plt.axis([0, 255, 0, np.max(histogram)])
    if save:
        plt.savefig(os.path.join(result_path, name))
    if show:
        plt.show()

    return True


def get_image_auto_binary(main_path, well_image, result_path='auto_binary', inv=False, show=False, blockSize=None,
                          C=16):
    # using cv2.adaptiveThreshold to enhance image(auto intensity threshold)
    # once one imgae
    # NOTICE: blockSize and C
    # blockSize is the MEAN or GAUSSIAN kernal size
    # C is a constant: Distinguish distance
    # if inv is True then do Negative Film

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(well_image):
        print('!ERROR! The well_image does not existed!')
        return False
    result_path = os.path.join(main_path, result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    t_path_list = os.path.split(well_image)  # [r'D:\pro\CD22\SSS_100%\S1', '2018-11-28~IPS_CD13~T1.jpg']
    t1_path_list = os.path.split(t_path_list[0])  # [r'D:\pro\CD22\SSS_100%', 'S1']
    t2_path_list = os.path.split(t1_path_list[0])  # [r'D:\pro\CD22', 'SSS_100%']
    SSS_folder = t2_path_list[1]  # 'SSS_100%'
    S_index = t1_path_list[1]  # 'S1'
    S = int(S_index.split('S')[1])  # 1
    img_name = t_path_list[1]  # '2018-11-28~IPS_CD13~T1.png'
    name_index = img_name[:-4]  # '2018-11-28~IPS_CD13~T1'
    # T = int(name_index.split('~T')[1])  # 1

    # if S <= 2:
    #     return True

    print('>>>', S_index, '---', name_index, ' Image Auto Binary :::')
    to_file = os.path.join(result_path, SSS_folder, S_index, img_name)
    if not os.path.exists(os.path.join(result_path, SSS_folder)):
        os.makedirs(os.path.join(result_path, SSS_folder))
    if not os.path.exists(os.path.join(result_path, SSS_folder, S_index)):
        os.makedirs(os.path.join(result_path, SSS_folder, S_index))

    img = image_to_gray(well_image)
    h, w = img.shape[:2]
    if blockSize is None:
        blockSize = int(np.min([h, w]) / 10)
        if blockSize % 2 != 1:
            blockSize = blockSize - 1

    if C is None:
        C = int(abs(np.mean(img) - np.median(img)))
        if C == 0:
            C = 16

    print('blockSize is: ', blockSize, 'C is: ', C)
    if inv:
        result_th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize, C)
        # result_th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize, C)
    else:
        result_th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, C)
        # result_th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)

    cv2.imwrite(to_file, result_th)

    if show:
        cv2.imshow('adaptiveThreshold_image', result_th)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return True


def research_stitched_image_elastic_bat(main_path, zoom, sort_function, analysis_function, do_SSS=True, do_SSSS=False,
                                        do_parallel=False, process_number=12, from_S=None, to_S=None):
    # core methed!
    # this program is design for go through all the SSS and SSSS images, and do core_features()
    # always after experiment, using the whole SSSS images
    # once one image
    # input main_path is the main path
    # input zoom is the zoom
    # from_S: contain
    # to_S: contain
    # output is True or False

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    zoom_str = "%.0f%%" % (zoom * 100)
    SSS_path = os.path.join(main_path, 'SSS_' + zoom_str)
    SSSS_path = os.path.join(main_path, 'SSSS_' + zoom_str)
    has_SSS = do_SSS
    has_SSSS = do_SSSS

    if not os.path.exists(SSS_path):
        print('!Caution! The', SSS_path, 'does not existed!')
        has_SSS = False
    if not os.path.exists(SSSS_path):
        print('!Caution! The', SSSS_path, 'does not existed!')
        has_SSSS = False

    if has_SSS:
        path_list = os.listdir(SSS_path)
        path_list.sort(key=lambda x: int(x.split('S')[1]))
        for this_S_folder in path_list:  # S1 to S96

            S_number = int(this_S_folder.split('S')[1])
            if from_S is None:
                pass
            elif S_number < int(from_S):
                continue

            if to_S is None:
                pass
            elif S_number > int(to_S):
                continue

            Spath = os.path.join(SSS_path, this_S_folder)
            img_files_list = os.listdir(Spath)
            sort_function(img_files_list)
            for img in img_files_list:  # all time sequence
                input_img = os.path.join(Spath, img)
                # name = this_S_folder + '_' + img
                if do_parallel:
                    # call_do_parallel(analysis_function, main_path, input_img)
                    # multithreading_control(process_number=process_number)
                    print('!Sorry! the funtion is not available now~')
                else:
                    # analysis_function(main_path, input_img, name=name)
                    analysis_function(main_path, input_img)

    if has_SSSS:
        path_list = os.listdir(SSSS_path)
        path_list.sort(key=lambda x: int(x.split('S')[1]))
        for this_S_folder in path_list:  # S1 to S96

            S_number = int(this_S_folder.split('S')[1])
            if from_S is None:
                pass
            elif S_number < int(from_S):
                continue

            if to_S is None:
                pass
            elif S_number > int(to_S):
                continue

            Spath = os.path.join(SSSS_path, this_S_folder)
            img_files_list = os.listdir(Spath)
            sort_function(img_files_list)
            # name = this_S_folder + '_' + img
            for img in img_files_list:  # all time sequence
                input_img = os.path.join(Spath, img)
                if do_parallel:
                    # call_do_parallel(analysis_function, main_path, input_img)
                    # multithreading_control(process_number=process_number)
                    print('!Sorry! the funtion is not available now~')
                else:
                    # analysis_function(main_path, input_img, name=name)
                    analysis_function(main_path, input_img)

    return True


def research_well_bat(main_path, zoom, analysis_function, sort_function, img1, img2, result_path='image_sub',
                      do_SSS=True, do_SSSS=True, do_parallel=False, process_number=12, from_S=None, to_S=None):
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    if not os.path.exists(os.path.join(main_path, result_path)):
        os.makedirs(os.path.join(main_path, result_path))

    zoom_str = "%.0f%%" % (zoom * 100)
    SSS_path = os.path.join(main_path, 'SSS_' + zoom_str)
    SSSS_path = os.path.join(main_path, 'SSSS_' + zoom_str)
    has_SSS = do_SSS
    has_SSSS = do_SSSS

    if not os.path.exists(SSS_path):
        print('!Caution! The', SSS_path, 'does not existed!')
        has_SSS = False
    if not os.path.exists(SSSS_path):
        print('!Caution! The', SSSS_path, 'does not existed!')
        has_SSSS = False

    if has_SSS:
        if not os.path.exists(os.path.join(main_path, result_path, 'SSS_' + zoom_str)):
            os.makedirs(os.path.join(main_path, result_path, 'SSS_' + zoom_str))

        path_list = os.listdir(SSS_path)
        path_list.sort(key=lambda x: int(x.split('S')[1]))
        for this_S_folder in path_list:  # S1 to S96

            S_number = int(this_S_folder.split('S')[1])
            if from_S is None:
                pass
            elif S_number < int(from_S):
                continue

            if to_S is None:
                pass
            elif S_number > int(to_S):
                continue

            from_path = os.path.join(SSS_path, this_S_folder)  # from_path
            destiny_path = os.path.join(main_path, result_path, 'SSS_' + zoom_str, this_S_folder)  # destiny_path

            if not os.path.exists(destiny_path):
                os.makedirs(destiny_path)

            img_1 = os.path.join(from_path, img1)
            img_2 = os.path.join(from_path, img2)

            # analysis_function(img_1, img_2, from_path,destiny_path,sort_function)
            if do_parallel:
                # call_do_parallel(analysis_function, main_path, input_img)
                # multithreading_control(process_number=process_number)
                print('!Sorry! the funtion is not available now~')
            else:
                # analysis_function(main_path, input_img, name=name)
                analysis_function(img_1, img_2, from_path, destiny_path, sort_function)

    if has_SSSS:
        if not os.path.exists(os.path.join(main_path, result_path, 'SSSS_' + zoom_str)):
            os.makedirs(os.path.join(main_path, result_path, 'SSSS_' + zoom_str))

        path_list = os.listdir(SSSS_path)
        path_list.sort(key=lambda x: int(x.split('S')[1]))
        for this_S_folder in path_list:  # S1 to S96

            S_number = int(this_S_folder.split('S')[1])
            if from_S is None:
                pass
            elif S_number < int(from_S):
                continue

            if to_S is None:
                pass
            elif S_number > int(to_S):
                continue

            from_path = os.path.join(SSSS_path, this_S_folder)  # from_path
            destiny_path = os.path.join(main_path, result_path, 'SSSS_' + zoom_str, this_S_folder)  # destiny_path

            if not os.path.exists(destiny_path):
                os.makedirs(destiny_path)

            img_1 = os.path.join(from_path, img1)
            img_2 = os.path.join(from_path, img2)

            # analysis_function(img_1, img_2, from_path,destiny_path,sort_function)
            if do_parallel:
                # call_do_parallel(analysis_function, main_path, input_img)
                # multithreading_control(process_number=process_number)
                print('!Sorry! the funtion is not available now~')
            else:
                # analysis_function(main_path, input_img, name=name)
                analysis_function(img_1, img_2, from_path, destiny_path, sort_function)

    return True


if __name__ == '__main__':
    # main program entrance, using for test one function
    print('!Notice! This is NOT the main function running!')
    print('Only TESTing Lib_Segmentation.py !')

    # main_path = r'C:\C137\PROCESSING\CD09'
    # in_folder = r'C:\C137\PROCESSING\CD09\auto_binary\SSS_100%\S3'
    # out_folder = r'C:\C137\PROCESSING\CD09\TEST_gabor_S3\image'
    # temp_image_folder_resize(in_folder, out_folder, pixel=(2160, 2160), box=(3000, 3000, 17000, 17000))
    # result_folder = r'C:\C137\PROCESSING\CD09\TEST_gabor_S3'
    # for i in range(3, 25):
    #     in_folder = r'C:\C137\PROCESSING\CD13\auto_binary\SSS_100%\S' + str(i)
    #     out_folder = r'S:\CD13\PROCESSING\Gabor\S' + str(i)
    #     temp_image_folder_resize(in_folder, out_folder, pixel=(1000, 1000), box=(1844, 1844, 7376, 7376))
    # main_path = r'S:\CD13\PROCESSING'
    # img_file = r'S:\CD13\PROCESSING\SSS_100%\S10\2018-11-28~IPS_CD13~T1.jpg'
    # prevent_disk_died(main_path, img_file)

    # i = 44
    # in_folder = r'C:\C137\PROCESSING\CD13\auto_binary\SSS_100%\S' + str(i)
    # out_folder = r'S:\CD13\PROCESSING\Gabor\S' + str(i)
    # temp_image_folder_resize(in_folder, out_folder, pixel=(1000, 1000), box=(1844, 1844, 7376, 7376))

    # done_well_sagementation_using_gabor(main_path, in_folder, out_folder, files_sort_function, k, gk, M, max_p, jump)

    input_path = r'C:\C137\PROCESSING\CD13\MyPGC_img\SSS_100%'
    to_path = r'C:\C137\PROCESSING\CD13\garbor'
    get_pic_features_and_segementation_using_gabor(input_path, to_path, '2018-11-30~IPS-3_CD13~T18.jpg', 5, 10, 20, 8)
    # research_stitched_image_elastic_bat(main_path, 1, files_sort_CD13, get_image_auto_binary, do_SSS=True,
    #                                     do_SSSS=False, do_parallel=False, process_number=12, from_S=61, to_S=72)

    # main_path = r'S:\CD13\PROCESSING'
    # in_folder = r'S:\CD13\PROCESSING\Gabor\S44'
    # result_folder = r'S:\CD13\PROCESSING\Gabor\S44_result'
    # done_well_sagementation_using_gabor(main_path, in_folder, result_folder, files_sort_CD13, 5, 66, 66, 8, 10)

    # for i in range(12, 25):
    #     in_folder = r'S:\CD13\PROCESSING\Gabor\S' + str(i)
    #     result_folder = r'S:\CD13\PROCESSING\Gabor\S' + str(i) + '_result'
    #     done_well_sagementation_using_gabor(main_path, in_folder, result_folder, files_sort_CD13, 5, 10, 20, 8, 10)

    # done_well_sagementation_using_gabor(main_path, in_folder, out_folder, files_sort_function, k, gk, M, max_p, jump)
    # gabor input:::
    # k: Number of clusters
    # gk: Size of the gabor kernel
    # M: Size of the gaussian window

    # main_path = r'G:\CD09\PROCESSING'
    # out_folder = r'G:\CD09\PROCESSING\TEST_gabor_S3\image'
    # result_folder = r'G:\CD09\PROCESSING\TEST_gabor_S3'
    # done_well_sagementation_using_gabor(main_path, out_folder, result_folder, files_sort_CD09, 10, 20, 6, 8)

    # sigle_image = r'C:\C137\PROCESSING\CD09\auto_binary\SSS_100%\S3\2018-09-02~IPS_CD09~T1.png'
    # get_img_pix_features_list(main_path, sigle_image, kernel=(69, 69), pix=(23, 23), getF=getSIFT, getC=get_KMeans,
    #                           n_clusters=3, result_path='pix_features', save=True, save_name=None, show=False)

    # main_path = r'C:\C137\PROCESSING\CD09\auto_binary'
    # img1 = r'2018-09-03~I-1_CD09~T13.png'
    # img2 = r'2018-09-06~II-01_CD09~T1.png'
    # research_well_bat(main_path, 1, get_folder_negative_film_V1, files_sort_CD09, img1, img2, result_path='image_sub',
    #                   do_SSS=True, do_SSSS=False, do_parallel=False, process_number=12, from_S=2)

    # main_path = r'C:\C137\PROCESSING\CD13'
    # research_stitched_image_elastic_bat(main_path, 1, files_sort_CD13, get_image_auto_binary, do_SSS=True,
    #                                     do_SSSS=False, do_parallel=False, process_number=12, from_S=44, to_S=44)

    # main_path = r'C:\C137\PROCESSING\CD13\auto_binary'
    # img1 = r'2018-12-01~I-1_CD13~T19.jpg'
    # img2 = r'2018-12-03~I-2_CD13~T19.jpg'
    # research_well_bat(main_path, 1, get_folder_negative_film_V1, files_sort_CD13, img1, img2, result_path='image_sub',
    #                   do_SSS=True, do_SSSS=False, do_parallel=False, process_number=12, from_S=18, to_S=24)

    # main_path = r'C:\C137\PROCESSING\CD09'
    # img1 = r'C:\C137\PROCESSING\CD09\auto_binary\SSS_100%\S3\2018-09-03~I-1_CD09~T13.png'
    # img2 = r'C:\C137\PROCESSING\CD09\auto_binary\SSS_100%\S3\2018-09-06~II-01_CD09~T1.png'
    # sub_folder = r'C:\C137\PROCESSING\CD09\auto_binary\SSS_100%\S3'
    # result_path = r'C:\C137\PROCESSING\CD09\image_sub_S3_V2'
    # get_folder_negative_film_V2(img1, img2, sub_folder, result_path, save=True, show=False)
    # sigle_image = r'C:\C137\PROCESSING\CD09\auto_binary\SSS_100%\S3\2018-09-06~II-01_CD09~T1.png'
    # get_img_pix_features_list(main_path, sigle_image, kernel=(69 * 2, 69 * 2), pix=(23 * 2, 23 * 2), getF=getMean,
    #                           getC=get_KMeans,
    #                           n_clusters=2, result_path='pix_features', save=True, save_name=None, show=True)
    # return_mask(main_path, sigle_image, kernel=(69, 69), pix=(23, 23), getF=getMean, getC=get_Bright_Dark,
    #             n_clusters=3, result_path='pix_features', save=True, save_name=None, show=False)
    # img = r'2018-09-04~I-2_CD09~T3.png'
    # img = r'2018-12-06~II-2_CD13~T18.jpg'
    # color_img_segmentation(img, n_clusters=3, show=True)

    # research_stitched_image_elastic_bat(main_path, 1, files_sort_CD09, get_image_auto_binary, do_SSS=True,
    #                                     do_SSSS=False, do_parallel=False, process_number=12)
    # img1 = r'C:\C137\PROCESSING\CD09\auto_binary\SSS_100%\S4\2018-09-03~I-1_CD09~T13.png'
    # img2 = r'C:\C137\PROCESSING\CD09\auto_binary\SSS_100%\S3\2018-09-13~F_CD09~T1.png'
    # sub_folder = r'C:\C137\PROCESSING\CD09\auto_binary\SSS_100%\S4'

    # get_image_sub(main_path, img1, img2, result_path='image_sub', name='sub_inv.png', save=True, show=False)
    # get_folder_negative_film_V1(main_path, img1, sub_folder, result_path='image_sub_S4', save=True, show=False)

    # img1 = r'C:\C137\PROCESSING\CD09\auto_binary\SSS_100%\S9\2018-09-03~I-1_CD09~T13.png'
    # sub_folder = r'C:\C137\PROCESSING\CD09\auto_binary\SSS_100%\S9'
    # img1 = r'C:\C137\PROCESSING\CD09\pix_features\2018-09-03~I-1_CD09~T13.png'
    # sub_folder = r'C:\C137\PROCESSING\CD09\auto_binary\SSS_100%\S3'
    # result_path = r'C:\C137\PROCESSING\CD09\image_sub_S3_V3'
    # get_folder_negative_film_V1(img1, sub_folder, result_path=result_path)

    # folder = r'C:\C137\PROCESSING\CD09\image_sub_S10'
    # temp_get_well_difference(main_path, folder, result_path='well_difference')

    # input_f = r'C:\C137\PROCESSING\CD09\image_sub_S10'
    # output_f = r'C:\Users\Kitty\Desktop\S10_2160'
    # temp_image_folder_resize(input_f, output_f, pdixel=(2160, 2160))

    # get_img_pix_features_list(main_path, img, kernel=(69, 69), pix=(3, 3), getF=getSIFT, getC=get_KMeans,
    #                           result_path='pix_features', save=True, show=False)

    # x = np.ones((600, 800, 3), dtype=np.uint8)
    # cv2.imshow('1', x)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # x = x * 255
    # cv2.imshow('255', x)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # np.array()

    # print(r)
