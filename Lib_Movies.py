import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import moviepy.editor as mpe
from Lib_Function import any_to_image, image_to_gray
from Lib_Sort import files_sort_CD09, files_sort_CD13, files_sort_CD26


def lambda_esi(x):
    x['temp'] -= x['temp'].min()
    return x


def extract_sp_image(main_path, repetition=False):
    # This program is design for extract the specific time point cell image
    # Notice:
    # the main_path must Contain: 'Experiment_Plan.csv' and 'DateTime.csv'
    # Experiment_Plan.csv (DF):S_index,medium,density,truth_density,chir,chir_hour,rest_hour
    #                               S1	B27	40	0.533267119	4	24  48
    #                               S96	B27	40	0.51046606	10	42  30
    #                                   ... ...
    # DateTime.csv (DF):chir_hour_index,date,experiment,t
    #                       0	2018/12/1	I-1_CD13	1
    #                       24	2018/12/1	I-1_CD13	19
    #                           ... ...
    # repetition=0: means the experiment has NO biological repetition
    # repetition=1: means the experiment has biological repetition
    # biological repetition: means the experiment has many chir_hour_index(maybe not the same Convergence degree)

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(os.path.join(main_path, 'Experiment_Plan.csv')):
        print('!ERROR! The Experiment_Plan.csv does not existed!')
        return False
    if not os.path.exists(os.path.join(main_path, 'Experiment_Date_Time.csv')):
        print('!ERROR! The Experiment_Date_Time.csv does not existed!')
        return False

    main_folder = 'Interesting_Time_Point'
    if os.path.exists(os.path.join(main_path, main_folder)):
        shutil.rmtree(os.path.join(main_path, main_folder))
    os.makedirs(os.path.join(main_path, main_folder))
    path_0_h = os.path.join(main_path, main_folder, '0H')
    path_CHIR_End = os.path.join(main_path, main_folder, 'CHIR-End')
    path_I_End = os.path.join(main_path, main_folder, 'I-End')
    path_144H = os.path.join(main_path, main_folder, '144H')
    path_Result = os.path.join(main_path, main_folder, 'Result')
    os.makedirs(path_0_h)
    os.makedirs(path_CHIR_End)
    os.makedirs(path_I_End)
    os.makedirs(path_144H)
    os.makedirs(path_Result)

    # Experiment_Plan.csv (DF):S_index,medium,density,truth_density,chir,chir_hour,rest_hour
    conditions_csv_mem = pd.read_csv(os.path.join(main_path, 'Experiment_Plan.csv'), header=0, index_col=0)
    # DateTime.csv (DF):chir_hour_index,date,experiment,t
    datetime_csv_mem = pd.read_csv(os.path.join(main_path, 'Experiment_Date_Time.csv'), header=0, index_col=0)

    if repetition:
        truly_density_median = conditions_csv_mem['truth_density'].median()
        conditions_csv_mem['temp'] = abs(conditions_csv_mem['truth_density'] - truly_density_median)
        conditions_csv_mem_2 = conditions_csv_mem.groupby(['chir', 'chir_hour', 'rest_hour']).apply(lambda_esi)

    this_files = os.listdir(main_path)
    for this_folder in this_files:
        if this_folder[:3] == 'SSS':  # this_folder is SSS_20% SSSS_100% or SSSS_25% etc.

            this_S_folders = os.listdir(os.path.join(main_path, this_folder))  # this_S_folders is S1,S2,S3... etc.
            for this_S_folder in this_S_folders:  # this_S_folder is S1 to S96
                if (repetition and conditions_csv_mem_2.loc[this_S_folder, 'temp'] == 0) or (not repetition):

                    this_S_files = os.listdir(os.path.join(main_path, this_folder, this_S_folder))  # all images in S1
                    hour_condition = conditions_csv_mem.loc[this_S_folder, 'chir_hour']
                    hour_rest = conditions_csv_mem.loc[this_S_folder, 'rest_hour']
                    i_end = hour_condition + hour_rest
                    i = 0

                    for hour_index in [0, hour_condition, i_end, 144]:

                        i += 1
                        if hour_index in datetime_csv_mem.index:
                            if i == 1:
                                folder_str = os.path.join(path_0_h, this_folder)
                            elif i == 2:
                                folder_str = os.path.join(path_CHIR_End, this_folder)
                            elif i == 3:
                                folder_str = os.path.join(path_I_End, this_folder)
                            elif i == 4:
                                folder_str = os.path.join(path_144H, this_folder)
                            else:
                                pass
                            if not os.path.exists(folder_str):
                                os.makedirs(folder_str)
                            date_str = datetime_csv_mem.loc[hour_index, 'date']
                            experiment_str = datetime_csv_mem.loc[hour_index, 'experiment']
                            time_point = datetime_csv_mem.loc[hour_index, 't']
                            image_name_str = date_str + '~' + experiment_str + '~'

                            if time_point == -1:
                                temp_count = 0
                                for temp_i in this_S_files:
                                    if temp_i.find(image_name_str) == 0:
                                        temp_count += 1
                                time_point_str = str(temp_count)
                            else:
                                time_point_str = str(time_point)
                            image_name_str += 'T' + time_point_str + '.jpg'  # 2018-12-01~I-1_CD13~T1.jpg

                            for this_image in this_S_files:  # this_image is one image
                                if this_image == image_name_str:
                                    new_image_name = image_name_str[:-4] + '~' + this_S_folder + '.jpg'
                                    shutil.copy(os.path.join(main_path, this_folder, this_S_folder, this_image),
                                                os.path.join(folder_str, new_image_name))
                        elif hour_index < 0:
                            print('!Warning! :The', this_S_folder, 'is not in Experiment Plan!')
                        else:
                            print('!Warning! :The', hour_index, 'hour does not existed!')
    return True


def make_whole_96well_IF(result_path, source_zoom, width=320, height=320, C=3):
    # using SSS_30% stitching result, stitching the whole 96 well(8row*12col)
    # input :
    # result_path: the folder where the input IF result image in exp:r'C:\CD13\result_CD13_20%'
    # source_zoom: the input zoom exp:0.3
    # mov_width & mov_height: is the one well size; the output is mov_height*12,mov_height*8
    # C: the IF channel num
    # output : only C images in the folder below main folder 'Whole_SSS_20%'

    t_xep_namp = os.path.split(result_path)
    main_path = t_xep_namp[0]
    well_w = 12
    well_h = 8
    S = well_h * well_w
    zoom_str = "%.0f%%" % (source_zoom * 100)
    output_str = 'Whole_SSS_' + str(320)
    output_folder = os.path.join(main_path, output_str)  # is the output folder
    if os.path.exists(main_path):
        if os.path.exists(output_folder):
            # shutil.rmtree(output_folder)
            pass
        else:
            os.makedirs(output_folder)  # make the output folder
        img_list = os.listdir(result_path)  # img_list is the image list
        if len(img_list) == S * C:
            pass
        else:
            print('!Warning! : Missing result IF image!')
            return False
        img_name = img_list[0].split('_' + zoom_str + '_')
        pre_name = img_name[0] + '_' + zoom_str + '_'
        end_name = '.' + img_name[1].split('.')[1]
        for c in range(1, C + 1):
            c_str = 'c' + str(c)
            whole_image = Image.new('RGB', (width * well_w, height * well_h))
            for s in range(1, S + 1):
                if s <= 9:
                    s_str = 's0' + str(s)
                else:
                    s_str = 's' + str(s)
                name = pre_name + s_str + c_str + end_name
                # print(name)
                full_img_path = os.path.join(result_path, name)
                if os.path.exists(full_img_path):
                    this_mosaic = Image.open(full_img_path)
                    rsz_img = this_mosaic.resize((width, height), Image.ANTIALIAS)
                    row = (s - 1) // well_w  # row is y
                    order = row % 2
                    if order == 0:
                        col = (s - 1) % well_w  # col is x
                    else:
                        col = well_w - 1 - (s - 1) % well_w
                    whole_image.paste(rsz_img, (col * width, row * height))
                else:
                    print('!Warning! : Missing the image :', full_img_path)
            whole_image.save(os.path.join(output_folder, c_str + end_name))
        return True
    else:
        return False


def make_whole_96well_movie(main_path, source_zoom, mov_width=320, mov_height=320):
    # using SSS_30% stitching result, stitching the whole 96 well(8row*12col) and make time series movie
    # input :
    # main_path: the folder where the input IF result image in exp:r'C:\CD13'
    # source_zoom: the input zoom exp:0.3
    # mov_width & mov_height: is the one well size; the output is mov_height*12,mov_height*8
    # output : only the SSS time series images in the folder below main folder 'Whole_SSS_30%'
    # output : only the SSS movie file exp:r'C:\CD13\Video\SSS.avi'
    well_w = 12
    well_h = 8
    zoom_str = "%.0f%%" % (source_zoom * 100)
    output_str = 'Whole_SSS_' + str(320)
    output_str_2 = 'Video'
    SSS_str = 'SSS_' + zoom_str
    SSSS_str = 'SSSS_' + zoom_str
    SSS_folder = os.path.join(main_path, SSS_str)
    SSSS_folder = os.path.join(main_path, SSSS_str)
    output_folder = os.path.join(main_path, output_str)  # is the output folder
    output_folder_2 = os.path.join(main_path, output_str_2)
    if os.path.exists(main_path):
        if os.path.exists(output_folder):
            # shutil.rmtree(output_folder)
            pass
        else:
            os.makedirs(output_folder)  # make the output folder
        if os.path.exists(output_folder_2):
            # shutil.rmtree(output_folder)
            pass
        else:
            os.makedirs(output_folder_2)  # make the output folder
        S96s_list = os.listdir(SSS_folder)  # S96s_list is the folder S01 to S96
        for S01 in S96s_list:
            if S01 == 'S1':
                pic_list = os.listdir(os.path.join(SSS_folder, S01))  # pic_list is all the image name
        pic_list.sort(key=lambda x: [int(x.split('~')[0].split('-')[0]), int(x.split('~')[0].split('-')[1]),
                                     int(x.split('~')[0].split('-')[2]), int(x.split('~T')[1].split('.')[0])])
        for T1 in pic_list:  # T1 is the whole 96 well at one time
            whole_image = Image.new('RGB', (mov_width * well_w, mov_height * well_h))  # one time one whole image
            for i in range(1, 97):
                i_str = 'S' + str(i)
                img_path = os.path.join(SSS_folder, i_str, T1)
                if os.path.exists(img_path):
                    this_mosaic = Image.open(img_path)
                    rsz_img = this_mosaic.resize((mov_width, mov_height), Image.ANTIALIAS)
                    row = (i - 1) // well_w  # row is y
                    order = row % 2
                    if order == 0:
                        col = (i - 1) % well_w  # col is x
                    else:
                        col = well_w - 1 - (i - 1) % well_w
                    whole_image.paste(rsz_img, (col * mov_width, row * mov_height))
                else:
                    print('!Warning! : Missing the image :', img_path)
            whole_image.save(os.path.join(output_folder, T1))
        output_file = os.path.join(output_folder_2, 'SSS.avi')
        make_movies(output_folder, output_file, mov_width=mov_width * well_w, mov_height=mov_height * well_h)
        return True
    else:
        return False


def make_movies(pic_path, output_file, mov_width=None, mov_height=None, zoom=None, fps=3, color=False,
                sort_function=None):
    # using SSS_30% and SSSS_30% stitching result, stitching one well(96*2 times) and make time series movie
    # using ::: cv2.VideoWriter
    # input :
    # pic_path: give the image files path(only Contained image file) exp:r'J:\PROCESSING\CD13\SSS_30%\S1'
    # output_file: the full out put movie name exp:r'J:\PROCESSING\CD13\Video\SSS_S1.avi'(only .avi!)
    # mov_width & mov_height: the output movie size(one well)
    # fps: default 3
    # color: default True
    # ! zoom can by pass mov_width & mov_height

    if not os.path.exists(pic_path):
        print('!ERROR! The pic_path does not existed!')
        return False

    output_path = os.path.split(output_file)
    output_path_0 = output_path[0]
    if not os.path.exists(output_path_0):
        print('!ERROR! The output_path does not existed!')
        return False

    # print(output_path_0)
    if output_file[-4:] == '.avi':
        print('Now making movie :', output_file)
    else:
        print('Only support .avi')
        return False

    image_file_type = ('.jpg', '.png', '.tif')

    files = os.listdir(pic_path)
    if sort_function is not None:
        sort_function(files)

    for file in files:
        if file[-4:] in image_file_type:
            img_path = os.path.join(pic_path, file)
            first_img = cv2.imread(img_path, -1)  # is the first imge
            img_width = first_img.shape[1]
            img_height = first_img.shape[0]
            break

    if mov_width is None:
        mov_width = img_width
    if mov_height is None:
        mov_height = img_height

    if zoom is not None:
        mov_width = int(zoom * img_width)
        mov_height = int(zoom * img_height)

    # if the folder Contained not pic files, then return False
    # for file in files:
    #     if file[-4:] in image_file_type:
    #         pass
    #     else:
    #         flag += 1
    # if flag > 0:
    #     print('pic_path has ', flag, 'files is not a picture!!')
    #     return False

    # chose one encode method
    # fourcc = cv2.VideoWriter_fourcc(*'XVID') #
    # fourcc = cv2.VideoWriter_fourcc(*'X264')  #
    # fourcc = cv2.VideoWriter_fourcc(*'I420') #
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # motion-jpeg codec
    # fourcc = cv2.VideoWriter_fourcc(*'PIM1') # MPEG-1 codec
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX') # MPEG-4 codec
    # fourcc = cv2.VideoWriter_fourcc(*'MP42') # MPEG-4.2 codec
    # fourcc = cv2.VideoWriter_fourcc(*'DIV3') # MPEG-4.3 codec
    # fourcc = cv2.VideoWriter_fourcc(*'U263') # H263 codec
    # fourcc = cv2.VideoWriter_fourcc(*'I263') # H263I codec
    fourcc = cv2.VideoWriter_fourcc(*'FLV1')  # FLV1 codec
    out_video = cv2.VideoWriter(output_file, fourcc, fps, (mov_width, mov_height), color)
    # sort_function(files)
    for file in files:
        if file[-4:] in image_file_type:
            img_path = os.path.join(pic_path, file)
            # img = Image.open(img_path)
            if color:
                img = cv2.imread(img_path, 1)
            else:
                img = cv2.imread(img_path, 0)
            if mov_width > img.shape[1]:
                print('!Warning! : Output setting width > img.width', mov_width, '>', img.shape[1])
                print('The img file is : ', img_path)
            if mov_height > img.shape[0]:
                print('!Warning! : Output setting height > img.height', mov_height, '>', img.shape[0])
                print('The img file is : ', img_path)
            # rsz_img = img.resize((mov_width, mov_height), Image.ANTIALIAS)
            # cv_img = np.asarray(rsz_img)
            rsz_img = img
            if mov_width != img.shape[1] or mov_height != img.shape[0]:
                rsz_img = cv2.resize(img, (mov_width, mov_height), interpolation=cv2.INTER_NEAREST)
            if color and len(rsz_img.shape) == 2:
                rsz_img = cv2.cvtColor(rsz_img, cv2.COLOR_GRAY2BGR)
                # img_GRAY = cv2.cvtColor(rsz_img, cv2.COLOR_RGB2GRAY)
            print('>>> make_movie() add: ', img_path)
            rsz_img = np.uint8(rsz_img)
            out_video.write(rsz_img)

    out_video.release()

    return True


def make_movie_imgs(imgs_path, output_file, fps=3, sort_function=None):
    # new core methed!
    # using ::: moviepy.editor
    # make pictures to one movie (with sound, using moviepy)
    # input a images files path: imgs_path

    if not os.path.exists(imgs_path):
        print('!ERROR! The imgs_path does not existed!')
        return False

    movie_file_type = ('.mp4')
    image_file_type = ('.jpg', '.png', '.tif')

    if output_file[-4:] in movie_file_type:
        print('Now making movie :', output_file)
    else:
        print('Only support .mp4 !')
        return False

    files = os.listdir(imgs_path)
    if sort_function is not None:
        sort_function(files)
    files_names = [os.path.join(imgs_path, f) for f in files if f[-4:] in image_file_type]
    out_clip = mpe.ImageSequenceClip(files_names, fps=fps)
    out_clip.write_videofile(output_file, fps=fps)

    return True


def movie_exp_bat(path, output_path, mov_width=None, mov_height=None, zoom=None, fps=3, color=False,
                  sort_function=None):
    # make time series movie in one father folder
    # input :
    # path: [full path] the father folder folder. Exp:r'E:\CD09\PROCESSING\MyPGC_img\SSS_50%'
    # output_path: [full path] the output_path. Exp:r'E:\CD09\PROCESSING\MyPGC_img\SSS_50_Video'
    # mov_width & mov_height: the output movie size(one well)

    if not os.path.exists(path):
        print('!ERROR! The path does not existed!')
        return False
    # if not os.path.exists(output_path):
    #     print('!ERROR! The output_path does not existed!')
    #     return False
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    folders = os.listdir(path)
    for folder in folders:
        video_name = folder + '.avi'
        pic_path = os.path.join(path, folder)
        output_file = os.path.join(output_path, video_name)
        make_movies(pic_path, output_file, mov_width=mov_width, mov_height=mov_height, zoom=zoom, fps=fps, color=color,
                    sort_function=sort_function)

    return True


def movie_exp_bat_old(main_path, source_zoom, output_str='Video', mov_width=None, mov_height=None, fps=3, color=False,
                      sort_function=None):
    # using SSS_30% and SSSS_30% stitching result, stitching one well(96*2 times) and make time series movie
    # input :
    # main_path: the folder where the input IF result image in exp:r'C:\CD13'
    # source_zoom: the input zoom exp:0.3
    # mov_width & mov_height: the output movie size(one well)
    # output : SSS and SSSS movie file exp:r'C:\CD13\Video\SSS_S1.avi'

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    zoom_str = "%.0f%%" % (source_zoom * 100)
    SSS_str = 'SSS_' + zoom_str
    SSSS_str = 'SSSS_' + zoom_str
    SSS_folder = os.path.join(main_path, SSS_str)
    SSSS_folder = os.path.join(main_path, SSSS_str)

    output_folder = os.path.join(main_path, output_str)  # is the output folder
    if os.path.exists(output_folder):
        # shutil.rmtree(output_folder)
        pass
    else:
        os.makedirs(output_folder)  # make the output folder

    if os.path.exists(SSS_folder):
        S96s = os.listdir(SSS_folder)  # S96s is the folder S01 to S96
        S96s.sort(key=lambda x: int(x.split('S')[1]))
        for S01 in S96s:  # S01 is the folder name
            pic_path = os.path.join(SSS_folder, S01)  # pic_path is the S01 full folder
            # if mov_width is None or mov_height is None:
            #     img_list = os.listdir(pic_path)
            #     first_img = os.path.join(pic_path, img_list[0])  # is the first imge
            #     img = Image.open(first_img)
            #     mov_width = img.width
            #     mov_height = img.height
            output_file = os.path.join(output_folder, 'SSS_' + S01 + '.avi')
            # # print(pic_path)
            # make_movies(pic_path, output_file, mov_width=mov_width, mov_height=mov_height)
            make_movies(pic_path, output_file, mov_width=mov_width, mov_height=mov_height, fps=fps, color=color,
                        sort_function=sort_function)

    if os.path.exists(SSSS_folder):
        S96s = os.listdir(SSSS_folder)  # S96s is the folder S01 to S96
        S96s.sort(key=lambda x: int(x.split('S')[1]))
        for S01 in S96s:  # S01 is the folder name
            pic_path = os.path.join(SSSS_folder, S01)  # pic_path is the S01 full folder
            # if mov_width == 0:
            #     img_list = os.listdir(pic_path)
            #     first_img = os.path.join(pic_path, img_list[0])  # is the first imge
            #     img = Image.open(first_img)
            #     mov_width = img.width
            #     mov_height = img.height
            output_file = os.path.join(output_folder, 'SSSS_' + S01 + '.avi')
            # make_movies(pic_path, output_file, mov_width=mov_width, mov_height=mov_height)
            make_movies(pic_path, output_file, mov_width=mov_width, mov_height=mov_height, fps=fps, color=color,
                        sort_function=sort_function)

    return True


def make_2compare_img(main_path, s_1, s_2, using_zoom=1, output_path='compare', xywh=None, label=None,
                      sort_function=None, cmp_number=None, img_width=None, img_height=None, zoom=None):
    # make 2 well images into one compare image using S number as input(it must have SSS_100% structure)
    # s_1, s_2: well number
    # zoom: image zoom: 0.3 or 0.5 or 1 or ...
    # output_path='compare'
    # xywh: the x,y,w,h of cut box: example: xywh = [3688, 3688, 1844, 1844]
    # label is Experiment Plan: example: label=r'C:\Users\Kitty\Desktop\test\Experiment_Plan.csv'
    # sort_function is a function of file names sort

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    output_path = os.path.join(main_path, output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    image_file_type = ('.jpg', '.png', '.tif')

    zoom_str = "%.0f%%" % (using_zoom * 100)
    from_path = 'SSS_' + zoom_str
    from_path = os.path.join(main_path, from_path)
    if not os.path.exists(from_path):
        print('!ERROR! The from_SSS path does not existed!')
        return False

    from_path_1 = os.path.join(from_path, 'S' + str(s_1))
    from_path_2 = os.path.join(from_path, 'S' + str(s_2))

    if not os.path.exists(from_path_1) or not os.path.exists(from_path_1):
        print('!ERROR! The S path does not existed!')
        return False

    to_path = 'S' + str(s_1) + '~' + 'S' + str(s_2)
    to_path = os.path.join(output_path, to_path)
    if not os.path.exists(to_path):
        os.makedirs(to_path)

    files_1 = os.listdir(from_path_1)
    files_2 = os.listdir(from_path_2)

    if len(files_1) >= len(files_2):  # the path must only conten image files
        files = files_1
    else:
        files = files_2
    # files Contain the most files

    if sort_function is not None:
        sort_function(files)

    if label is None:
        text_color_1 = (255, 0, 255)
        text_color_2 = (255, 0, 255)
        text_s_1 = 'S' + str(s_1)
        text_s_2 = 'S' + str(s_2)
    elif type(label) is str:
        exp_file = os.path.join(main_path, label)
        if not os.path.exists(exp_file):
            print('!WORNING! The exp_file does not existed!')
            text_color_1 = (255, 0, 255)
            text_color_2 = (255, 0, 255)
            text_s_1 = 'S' + str(s_1)
            text_s_2 = 'S' + str(s_2)
        else:
            exp_DF = pd.read_csv(exp_file, header=0, index_col=0)
            IF_result_list = []
            for i in range(1, exp_DF.shape[0] + 1):
                i_S = 'S' + str(i)
                IF_result_list.append(exp_DF.loc[i_S, 'IF_human'])
            chir = []
            for i in range(1, exp_DF.shape[0] + 1):
                i_S = 'S' + str(i)
                chir.append(exp_DF.loc[i_S, 'chir'])
            chir_hour = []
            for i in range(1, exp_DF.shape[0] + 1):
                i_S = 'S' + str(i)
                chir_hour.append(exp_DF.loc[i_S, 'chir_hour'])
            text_s_1 = 'S' + str(s_1) + '_CHIR' + str(chir[s_1 - 1]) + '_' + str(
                chir_hour[s_1 - 1]) + 'H : ' + '%.0f%%' % (IF_result_list[s_1 - 1] * 100)
            text_s_2 = 'S' + str(s_2) + '_CHIR' + str(chir[s_2 - 1]) + '_' + str(
                chir_hour[s_2 - 1]) + 'H : ' + '%.0f%%' % (IF_result_list[s_2 - 1] * 100)
            if IF_result_list[s_1 - 1] <= 0.1:
                text_color_1 = (0, 0, 255)
            elif IF_result_list[s_1 - 1] < 0.4:
                text_color_1 = (0, 255, 255)
            elif IF_result_list[s_1 - 1] <= 0.6:
                text_color_1 = (255, 255, 0)
            else:
                text_color_1 = (0, 255, 0)
            if IF_result_list[s_2 - 1] <= 0.1:
                text_color_2 = (0, 0, 255)
            elif IF_result_list[s_2 - 1] < 0.4:
                text_color_2 = (0, 255, 255)
            elif IF_result_list[s_2 - 1] <= 0.6:
                text_color_2 = (255, 255, 0)
            else:
                text_color_2 = (0, 255, 0)

    else:
        print('!ERROR! input lable is not identified!')
        return False

    if cmp_number is not None:
        print('!NOTICE! Using first ', cmp_number, ' images!')
        cmp_number_count = cmp_number
        if cmp_number_count < 1:
            print('!NOTICE! 0 images!')
            return False

    is_first = True
    for file in files:  # files Contain the most files
        # two difficult problems：
        # 1. from_path_1 or from_path_2 may do NOT contain file!
        # 2. the imgae in from_path_1 or from_path_2 may do NOT have the same size!

        if file[-4:] in image_file_type:

            if cmp_number is not None:
                cmp_number_count = cmp_number_count - 1
                if cmp_number_count < 1:
                    break

            img_1_path = os.path.join(from_path_1, file)
            img_2_path = os.path.join(from_path_2, file)
            img_to_path = os.path.join(to_path, file)
            file_name = file.split('.')[0]
            file_name = file_name.split('~')[1] + '~' + file_name.split('~')[2]

            if is_first:  # it must contain the first image
                if not os.path.exists(img_1_path) or not os.path.exists(img_2_path):
                    print('!ERROR! Can NOT find the first image! It must contain the first image!')
                    return False
                img_1 = cv2.imread(img_1_path, 1)
                img_2 = cv2.imread(img_2_path, 1)
                img_1_width = img_1.shape[1]
                img_1_height = img_1.shape[0]
                img_2_width = img_2.shape[1]
                img_2_height = img_2.shape[0]
                if img_width is None:
                    img_width = max(img_1_width, img_2_width)
                if img_height is None:
                    img_height = max(img_1_height, img_2_height)
                if zoom is not None:
                    img_width = int(max(img_1_width, img_2_width) * zoom)
                    img_height = int(max(img_1_height, img_2_height) * zoom)
                data_type = img_1.dtype
                is_first = False

            if os.path.exists(img_1_path):
                img_1 = cv2.imread(img_1_path, 1)  # img_1 = cv2.imread(img_1_path, -1)
            else:
                img_1 = np.zeros((img_height, img_width, 3), dtype=data_type)

            if os.path.exists(img_2_path):
                img_2 = cv2.imread(img_2_path, 1)  # img_2 = cv2.imread(img_2_path, -1)
            else:
                img_2 = np.zeros((img_height, img_width, 3), dtype=data_type)

            if img_1.shape[1] != img_width or img_1.shape[0] != img_height:
                img_1 = cv2.resize(img_1, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            if img_2.shape[1] != img_width or img_2.shape[0] != img_height:
                img_2 = cv2.resize(img_2, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

            if xywh is not None:
                x = xywh[0]
                y = xywh[1]
                w = xywh[2]
                h = xywh[3]
            else:
                x = 0
                y = 0
                w = img_width
                h = img_height

            if len(img_1.shape) == 2:
                img_shape = (h, w * 2)  # .shap(h,w)
            elif len(img_1.shape) == 3:
                img_shape = (h, w * 2, 3)  # .shap(h,w,c)
            to_image = np.zeros(img_shape, dtype=data_type)

            if len(img_1.shape) == 2:
                to_image[:, :w] = img_1[y:y + h, x:x + w]
                to_image[:, w:] = img_2[y:y + h, x:x + w]
            elif len(img_1.shape) == 3:
                to_image[:, :w, :] = img_1[y:y + h, x:x + w, :]
                to_image[:, w:, :] = img_2[y:y + h, x:x + w, :]

            to_image = cv2.putText(to_image, text_s_1, (int(w / 2 - w / 7), int(h / 12)), cv2.FONT_HERSHEY_SIMPLEX,
                                   int(w / 900), text_color_1, int(w / 300))
            to_image = cv2.putText(to_image, text_s_2, (int(3 * w / 2 - w / 7), int(h / 12)), cv2.FONT_HERSHEY_SIMPLEX,
                                   int(w / 900), text_color_2, int(w / 300))
            to_image = cv2.putText(to_image, file_name, (int(w - w / 7), int(h - h / 12)),
                                   cv2.FONT_HERSHEY_SIMPLEX, int(w / 900), (255, 0, 255), int(w / 300))

            cv2.imwrite(img_to_path, to_image)

    return True


def make_2folder_compare_img(main_path, folder_1, folder_2, output_path='compare', xywh=None, label=None,
                             sort_function=None, cmp_number=None, img_width=None, img_height=None):
    # make 2 well images into one compare image using image folder as input
    # s_1, s_2: well number
    # zoom: image zoom: 0.3 or 0.5 or 1 or ...
    # output_path='compare'
    # xywh: the x,y,w,h of cut box: example: xywh = [3688, 3688, 1844, 1844]
    # label is Experiment Plan: example: label=r'C:\Users\Kitty\Desktop\test\Experiment_Plan.csv'
    # sort_function is a function of file names sort

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    output_path = os.path.join(main_path, output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    image_file_type = ('.jpg', '.png', '.tif')

    # zoom_str = "%.0f%%" % (zoom * 100)
    # from_path = 'SSS_' + zoom_str
    # from_path = os.path.join(main_path, from_path)
    # if not os.path.exists(from_path):
    #     print('!ERROR! The from_SSS path does not existed!')
    #     return False

    from_path_1 = os.path.join(main_path, folder_1)
    from_path_2 = os.path.join(main_path, folder_2)

    if not os.path.exists(from_path_1) or not os.path.exists(from_path_1):
        print('!ERROR! The input folder path does not existed!')
        return False

    to_path = folder_1 + '~' + folder_2
    to_path = os.path.join(output_path, to_path)
    if not os.path.exists(to_path):
        os.makedirs(to_path)

    files_1 = os.listdir(from_path_1)
    files_2 = os.listdir(from_path_2)

    if len(files_1) >= len(files_2):  # the path must only conten image files
        files = files_1
    else:
        files = files_2

    sort_function(files)

    if label is None:
        text_color_1 = (255, 0, 255)
        text_color_2 = (255, 0, 255)
        text_s_1 = folder_1
        text_s_2 = folder_2
    elif type(label) is list:
        text_color_1 = (255, 0, 255)
        text_color_2 = (255, 0, 255)
        text_s_1 = label[0]
        text_s_2 = label[1]
    else:
        print('!NOTICE! Wrong label format!')
        text_color_1 = (255, 0, 255)
        text_color_2 = (255, 0, 255)
        text_s_1 = folder_1
        text_s_2 = folder_2

    if cmp_number is not None:
        print('!NOTICE! Using first ', cmp_number, ' images!')
        cmp_number_count = cmp_number
        if cmp_number_count < 1:
            print('!NOTICE! 0 images!')
            return False

    for file in files:

        if file[-4:] in image_file_type:

            if cmp_number is not None:
                cmp_number_count = cmp_number_count - 1
                if cmp_number_count < 1:
                    break

            img_1_path = os.path.join(from_path_1, file)
            img_2_path = os.path.join(from_path_2, file)
            img_to_path = os.path.join(to_path, file)
            file_name = file.split('.')[0]
            file_name = file_name.split('~')[1] + '~' + file_name.split('~')[2]

            dog_img_1 = False
            if os.path.exists(img_1_path):
                img_1 = cv2.imread(img_1_path, 1)  # img_1 = cv2.imread(img_1_path, -1)
                if img_width is None:
                    img_width = img_1.shape[1]
                if img_height is None:
                    img_height = img_1.shape[0]
                dog_img_1 = True
            if os.path.exists(img_2_path):
                img_2 = cv2.imread(img_2_path, 1)  # img_2 = cv2.imread(img_2_path, -1)
                if not dog_img_1:
                    if img_width is None:
                        img_width = img_2.shape[1]
                    if img_height is None:
                        img_height = img_2.shape[0]
                    img_1 = np.zeros(img_2.shape, dtype=img_2.dtype)
            else:
                img_2 = np.zeros(img_1.shape, dtype=img_1.dtype)

            if img_1.shape[1] != img_width or img_1.shape[0] != img_height:
                img_1 = cv2.resize(img_1, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            if img_2.shape[1] != img_width or img_2.shape[0] != img_height:
                img_2 = cv2.resize(img_2, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

            if xywh is not None:
                x = xywh[0]
                y = xywh[1]
                w = xywh[2]
                h = xywh[3]
            else:
                x = 0
                y = 0
                w = img_width
                h = img_height

            if len(img_1.shape) == 2:
                img_shape = (h, w * 2)  # .shap(h,w)
            elif len(img_1.shape) == 3:
                img_shape = (h, w * 2, 3)  # .shap(h,w,c)
            to_image = np.zeros(img_shape, dtype=img_1.dtype)

            if len(img_1.shape) == 2:
                to_image[:, :w] = img_1[y:y + h, x:x + w]
                to_image[:, w:] = img_2[y:y + h, x:x + w]
            elif len(img_1.shape) == 3:
                to_image[:, :w, :] = img_1[y:y + h, x:x + w, :]
                to_image[:, w:, :] = img_2[y:y + h, x:x + w, :]

            to_image = cv2.putText(to_image, text_s_1, (int(w / 2 - w / 7), int(h / 12)), cv2.FONT_HERSHEY_SIMPLEX,
                                   int(w / 900), text_color_1, int(w / 300))
            to_image = cv2.putText(to_image, text_s_2, (int(3 * w / 2 - w / 7), int(h / 12)), cv2.FONT_HERSHEY_SIMPLEX,
                                   int(w / 900), text_color_2, int(w / 300))
            to_image = cv2.putText(to_image, file_name, (int(w - w / 7), int(h - h / 12)),
                                   cv2.FONT_HERSHEY_SIMPLEX, int(w / 900), (255, 0, 255), int(w / 300))

            cv2.imwrite(img_to_path, to_image)

    return True


def make_compare_and_save_movie(main_path, s_1, s_2, using_zoom=1, output_path='compare', xywh=None, label=None,
                                sort_function=None, cmp_number=None, img_width=None, img_height=None):
    if not make_2compare_img(main_path, s_1, s_2, using_zoom=using_zoom, output_path=output_path, xywh=xywh,
                             label=label,
                             sort_function=sort_function, cmp_number=cmp_number, img_width=img_width,
                             img_height=img_height):
        return False

    to_path = 'S' + str(s_1) + '~' + 'S' + str(s_2)
    to_path = os.path.join(main_path, output_path, to_path)
    output_file = 'S' + str(s_1) + '~' + 'S' + str(s_2) + '.mp4'
    output_file = os.path.join(main_path, output_path, output_file)

    # if not make_movies(to_path, output_file, mov_width=mov_width, mov_height=mov_height, fps=1, color=True,sort_function=sort_function):
    #     return False
    if not make_movie_imgs(to_path, output_file, fps=1, sort_function=sort_function):
        return False

    return True


def make_folder_compare_and_save_movie(main_path, folder_1, folder_2, output_path='compare', xywh=None, label=None,
                                       sort_function=None, cmp_number=None, img_width=None, img_height=None):
    if not make_2folder_compare_img(main_path, folder_1, folder_2, output_path=output_path, xywh=xywh, label=label,
                                    sort_function=sort_function, cmp_number=cmp_number, img_width=img_width,
                                    img_height=img_height):
        return False

    # to_path = 'S' + str(s_1) + '~' + 'S' + str(s_2)
    # to_path = os.path.join(main_path, output_path, to_path)
    # output_file = 'S' + str(s_1) + '~' + 'S' + str(s_2) + '.mp4'
    # output_file = os.path.join(main_path, output_path, output_file)

    to_path = folder_1 + '~' + folder_2
    to_path = os.path.join(main_path, output_path, to_path)
    output_file = folder_1 + '~' + folder_2 + '.mp4'
    output_file = os.path.join(main_path, output_path, output_file)

    # if not make_movies(to_path, output_file, mov_width=mov_width, mov_height=mov_height, fps=1, color=True,sort_function=sort_function):
    #     return False
    if not make_movie_imgs(to_path, output_file, fps=1, sort_function=sort_function):
        return False

    return True


def make_texted_img(img, text, color):
    # print text on image

    img = any_to_image(img)
    if img is None:
        return False
    if type(text) is not str:
        print('!ERROR! type(text) is not str')
        return False

    if color <= 0.1:
        text_color = (0, 0, 255)
    elif color < 0.4:
        text_color = (0, 255, 255)
    elif color <= 0.6:
        text_color = (255, 255, 0)
    else:
        text_color = (0, 255, 0)

    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    w = img.shape[1]
    h = img.shape[0]
    img = cv2.putText(img, text, (int(w / 3), int(h / 12)), cv2.FONT_HERSHEY_SIMPLEX, int(w / 900), text_color,
                      int(w / 300))

    return img


if __name__ == '__main__':
    # main program entrance, using for test one function
    print('!Notice! This is NOT the main function running!')
    print('Only TESTing Lib_Movies.py !')

    main_path = r'C:\C137\PROCESSING\CD13\auto_binary'
    label = r'C:\C137\PROCESSING\CD13\Experiment_Plan.csv'
    make_compare_and_save_movie(main_path, 12, 13, using_zoom=1, output_path='compare', xywh=None, label=label,
                                sort_function=files_sort_CD13, img_width=2160, img_height=2160)


    # main_path = r'C:\C137\PROCESSING\CD09'
    # path = r'C:\C137\PROCESSING\CD09\auto_binary\SSS_100%'
    # output_path = r'C:\C137\PROCESSING\CD09\auto_binary_movie_10%'
    # movie_exp_bat(path, output_path, mov_width=None, mov_height=None, zoom=0.1, fps=3, color=False,
    #               sort_function=files_sort_CD09)

    # main_path = r'C:\C137\PROCESSING\CD09\auto_binary'
    # # output_path = r'C:\C137\PROCESSING\CD09\auto_binary'
    # label = r'C:\C137\PROCESSING\CD09\auto_binary\Experiment_Plan.csv'
    # make_compare_and_save_movie(main_path, 10, 19, using_zoom=1, output_path='compare', xywh=None, label=label,
    #                             sort_function=files_sort_CD09, img_width=2160, img_height=2160)

    # folder_1 = r'C:\C137\PROCESSING\CD09\auto_binary\SSS_100%\S3'
    # folder_2 = r'C:\C137\PROCESSING\CD09\auto_binary\SSS_100%\S10'
    # make_folder_compare_and_save_movie(main_path, folder_1, folder_2, output_path='compare', xywh=None, label=label,
    #                                    sort_function=files_sort_CD09, cmp_number=None, img_width=2160, img_height=2160)

    # xywh = [0, 0, 1000, 1000]
    # make_folder_compare_and_save_movie(main_path, 'S44_selected', 'S44_myPGC_selected', output_path='compare',
    #                                    xywh=None, mov_width=None, mov_height=None, label=None,
    #                                    sort_function=files_sort_CD13, cmp_number=None, img_width=1000, img_height=1000)
    # make_folder_compare_and_save_movie(main_path, 'S44_selected', 'S44_reduce_Gabor_5_10_3_evolution', output_path='compare',
    #                                    xywh=None, mov_width=None, mov_height=None, label=None,
    #                                    sort_function=files_sort_CD13, cmp_number=None, img_width=1000, img_height=1000)
    # make_folder_compare_and_save_movie(main_path, 'S44_myPGC_selected', 'S44_reduce_Gabor_5_10_3_evolution', output_path='compare',
    #                                    xywh=None, mov_width=None, mov_height=None, label=None,
    #                                    sort_function=files_sort_CD13, cmp_number=None, img_width=1000, img_height=1000)
    # main_path = r'E:\CD26\Processing\MyPGC_img'

    # path = r'E:\CD26\Processing\MyPGC_img\compare'
    # output_path = r'E:\CD26\Processing\MyPGC_img\compare'
    # label = r'Experiment_Plan.csv'
    # cmp_number = 13
    # make_compare_and_save_movie(main_path, 48, 37, zoom=1, output_path='compare', xywh=None, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26, cmp_number=cmp_number)
    # make_compare_and_save_movie(main_path, 25, 26, zoom=1, output_path='compare', xywh=None, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26, cmp_number=cmp_number)
    # make_compare_and_save_movie(main_path, 24, 22, zoom=1, output_path='compare', xywh=None, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26, cmp_number=cmp_number)
    # make_compare_and_save_movie(main_path, 49, 52, zoom=1, output_path='compare', xywh=None, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26, cmp_number=cmp_number)
    # make_compare_and_save_movie(main_path, 72, 68, zoom=1, output_path='compare', xywh=None, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26, cmp_number=cmp_number)
    # make_compare_and_save_movie(main_path, 57, 58, zoom=1, output_path='compare', xywh=None, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26, cmp_number=cmp_number)
    # movie_exp_bat(path, output_path, mov_width=None, mov_height=None, zoom=0.5, fps=1, color=True,
    #               sort_function=files_sort_CD26)
    # output_path = r'E:\CD26\Processing\MyPGC_img\SSS_50%_Video'

    # extract_sp_image(main_path, repetition=False)
    # make_movies(pic_path, output_file, mov_width=None, mov_height=None, zoom=0.3, fps=2, color=False,
    #             sort_function=files_sort_CD09)
    # movie_exp_bat(path, output_path, mov_width=None, mov_height=None, zoom=0.5, fps=2, color=False,
    #               sort_function=files_sort_CD26)

    # pic_path = r'C:\C137\PROCESSING\CD09'
    # movie_exp_bat(main_path, 0.12, output_str='Video', mov_width=None, mov_height=None, fps=3, color=False,
    #               sort_function=files_sort_CD09)
    # pic_path = r'E:\CD26\Processing\MyPGC_img\All_wells_320'
    # output_file = r'E:\CD26\Processing\MyPGC_img\All_wells_320.avi'
    # make_movies(pic_path, output_file, mov_width=None, mov_height=None, zoom=None, fps=2, color=False,
    #             sort_function=files_sort_CD26)

    # main_path = r'E:\CD26\Processing'
    # output_path = 'compare_details'
    # label = r'Experiment_Plan.csv'
    # xywh = [3688, 3688, 1844, 1844]
    # xywh = [1844, 1844, 922, 922]

    # make_compare_and_save_movie(main_path, 3, 9, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 22, 16, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 27, 33, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 46, 40, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 51, 57, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 70, 64, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 75, 81, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 94, 88, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    #
    # make_compare_and_save_movie(main_path, 4, 10, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 21, 15, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 28, 34, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 45, 39, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 52, 58, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 69, 63, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 76, 82, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 93, 87, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    #
    # make_compare_and_save_movie(main_path, 4, 76, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 3, 75, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 22, 76, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 21, 75, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 94, 76, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 93, 75, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 9, 81, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 10, 82, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 16, 81, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 15, 82, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 88, 81, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    # make_compare_and_save_movie(main_path, 87, 82, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                             mov_height=None, label=label, sort_function=files_sort_CD26)
    #
    # for i in [52, 53, 68, 69, 34, 39, 58, 63, 2, 23, 26, 47, 50, 71, 74, 20, 29, 44, 8, 17, 32, 41, 56, 65, 80, 11, 14,
    #           35, 38, 59, 62, 83]:
    #     for j in [k for k in range(1, 97)]:
    #         make_compare_and_save_movie(main_path, i, j, zoom=1, output_path=output_path, xywh=xywh, mov_width=None,
    #                                     mov_height=None, label=label, sort_function=files_sort_CD26)

    # make_movies(r'E:\PROCESSING\CD13\compare_details\S13~S15', r'E:\PROCESSING\CD13\compare_details\S13~S15.avi',
    #             mov_width=1920, mov_height=960, fps=1, color=True)

    # xywh = [3688, 3688, 1844, 1844]
    # good-good
    # make_compare_and_save_movie(main_path, 13, 36, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 36, 37, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 37, 42, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 42, 43, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 43, 45, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 45, 13, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # good-bad
    # make_compare_and_save_movie(main_path, 13, 1, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 36, 2, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 37, 10, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 42, 11, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 43, 15, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 45, 34, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 13, 49, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 36, 60, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 37, 62, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 42, 65, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 43, 68, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 45, 71, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 45, 72, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # bad_bad
    # make_compare_and_save_movie(main_path, 10, 15, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 48, 34, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 62, 84, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 83, 89, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 90, 94, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 72, 1, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 74, 60, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # controlled variable
    # make_compare_and_save_movie(main_path, 28, 4, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 29, 5, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 30, 6, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 43, 19, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 44, 20, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 45, 21, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 28, 76, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 29, 77, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 30, 78, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 43, 91, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 44, 92, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 45, 93, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    #
    # make_compare_and_save_movie(main_path, 10, 11, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 11, 12, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 12, 15, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 15, 14, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 14, 13, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 13, 10, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 25, 1, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 26, 2, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 27, 3, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 31, 7, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 32, 8, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 33, 9, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 34, 10, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 35, 11, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 36, 12, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 37, 13, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 38, 14, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 39, 15, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 40, 16, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 41, 17, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 42, 18, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 46, 22, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 47, 23, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 48, 24, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # rand
    # make_compare_and_save_movie(main_path, 48, 25, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 47, 26, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 46, 27, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 45, 28, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 44, 29, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 43, 30, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 42, 31, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 41, 32, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 40, 33, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 39, 34, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 38, 35, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 37, 36, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # rand
    # make_compare_and_save_movie(main_path, 48, 1, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 48, 96, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 47, 2, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 47, 95, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 46, 3, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 46, 94, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 45, 4, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 45, 93, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 44, 5, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 44, 92, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 43, 6, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 43, 91, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 42, 7, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 42, 90, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 41, 8, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 41, 89, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 40, 9, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 40, 88, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 39, 10, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 39, 87, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 38, 11, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 38, 86, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 37, 12, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 37, 85, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)

    # output_path = 'compare_well'
    # xywh = None

    # make_compare_and_save_movie(main_path, 13, 36, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 36, 37, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 37, 42, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 42, 43, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 43, 45, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 45, 13, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # good-bad
    # make_compare_and_save_movie(main_path, 13, 1, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 36, 2, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 37, 10, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 42, 11, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 43, 15, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 45, 34, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 13, 49, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 36, 60, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 37, 62, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 42, 65, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 43, 68, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 45, 71, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 45, 72, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # bad_bad
    # make_compare_and_save_movie(main_path, 10, 15, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 48, 34, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 62, 84, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 83, 89, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 90, 94, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 72, 1, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 74, 60, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # controlled variable
    # make_compare_and_save_movie(main_path, 28, 4, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 29, 5, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 30, 6, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 43, 19, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 44, 20, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 45, 21, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 28, 76, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 29, 77, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 30, 78, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 43, 91, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 44, 92, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 45, 93, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    #
    # make_compare_and_save_movie(main_path, 10, 11, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 11, 12, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 12, 15, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 15, 14, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 14, 13, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 13, 10, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 25, 1, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 26, 2, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 27, 3, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 31, 7, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 32, 8, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 33, 9, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 34, 10, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 35, 11, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 36, 12, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 37, 13, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 38, 14, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 39, 15, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 40, 16, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 41, 17, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 42, 18, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 46, 22, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 47, 23, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 48, 24, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # rand
    # make_compare_and_save_movie(main_path, 48, 25, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 47, 26, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 46, 27, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 45, 28, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 44, 29, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 43, 30, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 42, 31, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 41, 32, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 40, 33, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 39, 34, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 38, 35, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 37, 36, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # # rand
    # make_compare_and_save_movie(main_path, 48, 1, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 48, 96, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 47, 2, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 47, 95, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 46, 3, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 46, 94, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 45, 4, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 45, 93, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 44, 5, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 44, 92, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 43, 6, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 43, 91, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 42, 7, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 42, 90, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 41, 8, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 41, 89, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 40, 9, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 40, 88, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 39, 10, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 39, 87, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 38, 11, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 38, 86, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 37, 12, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)
    # make_compare_and_save_movie(main_path, 37, 85, zoom=1, output_path=output_path, mov_width=1920, mov_height=960,
    #                             xywh=xywh, label=label)

    # s_1 = 1
    # s_2 = 3

    # make_compare_img(main_path, s_1, s_2, zoom=1, output_path='compare')
    # make_compare_and_save_movie(main_path, s_1, s_2, zoom=1, output_path='compare',mov_width=1920, mov_height=960)

    # output_path = 'compare'
    #
    # to_path = 'S' + str(s_1) + '~' + 'S' + str(s_2)
    # to_path = os.path.join(main_path, output_path, to_path)
    # output_file = 'S' + str(s_1) + '~' + 'S' + str(s_2) + '.avi'
    # output_file = os.path.join(main_path, output_path, output_file)
    #
    # make_movies(to_path, output_file, mov_width=1920, mov_height=960, fps=3, color=False)
