import os, shutil
import numpy as np
import cv2
from Lib_Function import any_to_image, image_to_gray, get_CZI_image, \
    trans_blur, trans_CLAHE, trans_Unsharp_Masking


def copy_AEimages_to_paired_HR_LR_GT(input_path, output_path, prefix, B=1, T=1, all_S=96, Z=1, C=1, all_M=25,
                                     used_M=[7, 8, 9, 12, 13, 14, 17, 18, 19], HR_path=r'A', LR_path=r'B',
                                     GT_path=r'C'):
    # get_CZI_image(path, B, T, S, Z, C, M) return [img_path, 'S1', '2018-09-03', 'I-1_CD09', 'T1', 'Z1', 'C1', 'M1']
    # used_M=[6,7,8,11,12,13,16,17,18]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    HR_path = os.path.join(output_path, HR_path)
    if not os.path.exists(HR_path):
        os.makedirs(HR_path)
    LR_path = os.path.join(output_path, LR_path)
    if not os.path.exists(LR_path):
        os.makedirs(LR_path)
    GT_path = os.path.join(output_path, GT_path)
    if not os.path.exists(GT_path):
        os.makedirs(GT_path)

    for S in range(1, all_S + 1):
        for M in used_M:
            GT_source_img = get_CZI_image(input_path, B, T, S, Z, C, M)
            img_name = prefix + '~' + GT_source_img[1] + '~' + GT_source_img[7] + '.png'  # 'prefix~S1~M1'
            img_original = image_to_gray(GT_source_img[0])
            img_enhanced = trans_CLAHE(trans_Unsharp_Masking(img_original))
            img_blur = trans_blur(img_original)
            cv2.imwrite(os.path.join(HR_path, img_name), img_enhanced)
            cv2.imwrite(os.path.join(LR_path, img_name), img_blur)
            cv2.imwrite(os.path.join(GT_path, img_name), img_original)

    return


def make_CD61_copy_to_seg(main_path, well_image):
    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False
    if not os.path.exists(well_image):
        print('!ERROR! The well_image does not existed!')
        return False

    folder_bright = 'hand_labeling_Day5'
    folder_End = 'hand_labeling_End'
    folder_IF = 'hand_labeling_IF'

    do_bright = ['2021-05-24~CD59_d6_cpc~Z1', '2021-05-24~CD59_d6_cpc~Z2', '2021-05-24~CD59_d6_cpc~Z3',
                 '2021-05-24~CD59_d6_cpc~Z4', '2021-05-24~CD59_d6_cpc~Z5']

    do_End = ['2021-05-31~CD61_d12_liveCM~T1']

    do_IF = ['2021-06-12~CD61_IF~T1~C1', '2021-06-12~CD61_IF~T1~C2', '2021-06-12~CD61_IF~T1~C3']

    if os.path.exists(well_image):
        t_path_list = os.path.split(well_image)  # [r'D:\pro\CD22\SSS_100%\S1', '2018-11-28~IPS_CD13~T1.jpg']
        t1_path_list = os.path.split(t_path_list[0])  # [r'D:\pro\CD22\SSS_100%', 'S1']
        t2_path_list = os.path.split(t1_path_list[0])  # [r'D:\pro\CD22', 'SSS_100%']
        SSS_folder = t2_path_list[1]  # 'SSS_100%'
        S_index = t1_path_list[1]  # 'S1'
        img_name = t_path_list[1]  # '2018-11-28~IPS_CD13~T1.png'
        name_index = img_name[:-4]  # '2018-11-28~IPS_CD13~T1'

        if name_index in do_bright:
            print('>>>', S_index, '---', name_index, 'make_CD61_copy_to_seg() :::')
            this_end = img_name.split('~')[-1]
            to_file = os.path.join(main_path, folder_bright, S_index + '~' + this_end)
            if not os.path.exists(os.path.join(main_path, folder_bright)):
                os.makedirs(os.path.join(main_path, folder_bright))
            shutil.copy(well_image, to_file)

        if name_index in do_End:
            print('>>>', S_index, '---', name_index, 'make_CD61_copy_to_seg() :::')
            this_end = img_name.split('~')[-1]
            to_file = os.path.join(main_path, folder_End, S_index + '~End.png')
            if not os.path.exists(os.path.join(main_path, folder_End)):
                os.makedirs(os.path.join(main_path, folder_End))
            shutil.copy(well_image, to_file)

        if name_index in do_IF:
            print('>>>', S_index, '---', name_index, 'make_CD61_copy_to_seg() :::')
            this_end = img_name.split('~')[-1]
            to_file = os.path.join(main_path, folder_IF, S_index + '~' + this_end)
            if not os.path.exists(os.path.join(main_path, folder_IF)):
                os.makedirs(os.path.join(main_path, folder_IF))
            shutil.copy(well_image, to_file)

    return True


def folder_image_resize(image_path, size=(2580, 2580)):
    if not os.path.exists(image_path):
        print('!ERROR! The image_path does not existed!')
        return False

    path_list = os.listdir(image_path)

    for i in path_list:  # r'Label_1.png'
        img_dirfile = os.path.join(image_path, i)
        if os.path.isfile(img_dirfile):
            o_img = any_to_image(img_dirfile)
            d_img = cv2.resize(o_img, size, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(img_dirfile, d_img)
        else:
            folder_image_resize(img_dirfile, size=size)

    return True


def folder_image_resize_0(image_path, size=(2580, 2580)):
    if not os.path.exists(image_path):
        print('!ERROR! The image_path does not existed!')
        return False

    path_list = os.listdir(image_path)

    for i in path_list:  # r'Label_1.png'
        img_file = os.path.join(image_path, i)
        o_img = any_to_image(img_file)
        d_img = cv2.resize(o_img, size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(img_file, d_img)

    return True


def folder_image_cut_n2_blocks(image_path, output_path, n=3):
    if not os.path.exists(image_path):
        print('!ERROR! The image_path does not existed!')
        return False
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    path_list = os.listdir(image_path)
    for this_img in path_list:  # r'Label_1.png'
        img_file = os.path.join(image_path, this_img)
        o_img = any_to_image(img_file)
        height = o_img.shape[0]
        width = o_img.shape[1]
        item_width = int(width / n)
        item_height = int(height / n)

        box_list = []  # (col0, row0, col1, row1)
        for i in range(0, n):  # row
            for j in range(0, n):  # col
                box = (i * item_width, j * item_height, (i + 1) * item_width, (j + 1) * item_height)
                box_list.append(box)
        image_list = [o_img[box[1]:box[3], box[0]:box[2]] for box in box_list]

        index = 0
        for image in image_list:
            o_file = os.path.join(output_path,
                                  this_img.split('.')[0] + '_' + str(index) + '.' + this_img.split('.')[-1])
            cv2.imwrite(o_file, image)
            index += 1

    return True


def image2_write(path_A, path_B, path_AB):
    im_A = cv2.imread(path_A, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv2.imread(path_B, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_AB = np.concatenate([im_A, im_B], 1)
    cv2.imwrite(path_AB, im_AB)


def A_B_AB(main_path):
    image_file_type = ('.jpg', '.png', '.tif')

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    A_path = os.path.join(main_path, r'A')
    if not os.path.exists(A_path):
        print('!ERROR! The A_path does not existed!')
        return False

    B_path = os.path.join(main_path, r'B')
    if not os.path.exists(A_path):
        print('!ERROR! The B_path does not existed!')
        return False

    AB_path = os.path.join(main_path, r'AB')
    if os.path.exists(AB_path):
        # shutil.rmtree(output_folder)
        pass
    else:
        os.makedirs(AB_path)  # make the output folder

    folder_list = os.listdir(A_path)
    for this_folder in folder_list:
        A_sub_path = os.path.join(A_path, this_folder)
        B_sub_path = os.path.join(B_path, this_folder)
        if not os.path.exists(B_sub_path):
            print('!ERROR! The B_sub_path does not existed!')
            return False
        AB_sub_path = os.path.join(AB_path, this_folder)
        if not os.path.exists(AB_sub_path):
            os.makedirs(AB_sub_path)

        img_list = os.listdir(A_sub_path)
        for this_img in img_list:
            if this_img[-4:] in image_file_type:
                this_imgA_path = os.path.join(A_sub_path, this_img)
                this_imgB_path = os.path.join(B_sub_path, this_img)
                this_imgAB_path = os.path.join(AB_sub_path, this_img)
                if not os.path.exists(this_imgB_path):
                    print('!ERROR! The this_imgB_path does not existed!')
                    continue
                image2_write(this_imgA_path, this_imgB_path, this_imgAB_path)


def remove_suffix_2(image_path):
    if not os.path.exists(image_path):
        print('!ERROR! The image_path does not existed!')
        return False

    path_list = os.listdir(image_path)

    for i in path_list:  # r'Label_1.png'
        old_img_file = os.path.join(image_path, i)
        new_name = i.split('~')[0] + '~' + i.split('~')[1] + '.' + i.split('.')[-1]
        new_img_file = os.path.join(image_path, new_name)
        os.rename(old_img_file, new_img_file)

    return True


def add_prefix(image_path, prefix):
    if not os.path.exists(image_path):
        print('!ERROR! The image_path does not existed!')
        return False

    path_list = os.listdir(image_path)

    for i in path_list:  # r'Label_1.png'
        old_img_file = os.path.join(image_path, i)
        new_name = prefix + '~' + i
        new_img_file = os.path.join(image_path, new_name)
        os.rename(old_img_file, new_img_file)

    return True


if __name__ == '__main__':
    # input_path = r'E:\Image_Processing\CD13\2018-12-17\Result3_CD13'
    # output_path = r'C:\Users\Kitty\Desktop\CD13_Bright_2048'
    # copy_AEimages_to_paired_HR_LR_GT(input_path, output_path, 'CD13')

    # folder_image_cut_n2_blocks(r'C:\Users\Kitty\Desktop\CD13_Bright_2048\A', r'C:\Users\Kitty\Desktop\CD13_Bright_128\A', n=16)
    # folder_image_cut_n2_blocks(r'C:\Users\Kitty\Desktop\CD13_Bright_2048\B', r'C:\Users\Kitty\Desktop\CD13_Bright_128\B', n=16)
    # folder_image_cut_n2_blocks(r'C:\Users\Kitty\Desktop\CD13_Bright_2048\C', r'C:\Users\Kitty\Desktop\CD13_Bright_128\C', n=16)
    A_B_AB(r'C:\Users\Kitty\Desktop\CD13_Bright_128')

    # image_path = r'L:\cTnT_2580'
    # folder_image_resize(image_path, size=(2580, 2580))
    # folder_image_cut_n2_blocks(r'L:\cTnT_2580\A\train', r'L:\cTnT_860\A\train', n=3)
    # folder_image_cut_n2_blocks(r'L:\cTnT_2580\B\train', r'L:\cTnT_860\B\train', n=3)
    # A_B_AB(r'L:\cTnT_860')

    # image_path = r'C:\DATA\cTnT_original_2580'
    # folder_image_resize(image_path, size=(2580, 2580))
    # remove_suffix_2(r'C:\DATA\cTnT_original_2580\A\train')
    # remove_suffix_2(r'C:\DATA\cTnT_original_2580\B\train')
    # folder_image_cut_n_blocks(r'C:\DATA\cTnT_original_2580\A\train',  r'C:\DATA\cTnT_original_860\A\train', n=3)
    # folder_image_cut_n_blocks(r'C:\DATA\cTnT_original_2580\B\train',  r'C:\DATA\cTnT_original_860\B\train', n=3)
    # A_B_AB(r'C:\DATA\cTnT_original_860')
