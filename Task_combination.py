import os, shutil
import numpy as np
import cv2
from Lib_Function import any_to_image, image_to_gray, get_CZI_image, \
    trans_blur, trans_CLAHE, trans_myPGC, trans_Unsharp_Masking, get_ImageVar, cut_dir_center


def dir_enhanced(input_path, output_path):
    img_list = os.listdir(input_path)
    for img in img_list:
        image = image_to_gray(os.path.join(input_path, img))
        img_enhanced = trans_CLAHE(trans_Unsharp_Masking(image))
        cv2.imwrite(os.path.join(output_path, img), img_enhanced)


def dir_enhanced2(input_path, output_path):
    img_list = os.listdir(input_path)
    for img in img_list:
        image = os.path.join(input_path, img)
        img_enhanced = trans_CLAHE(trans_myPGC(image))
        cv2.imwrite(os.path.join(output_path, img), img_enhanced)


def choose_test_ABC(input_path, output_path, name_list):
    if not os.path.exists(input_path):
        return False

    A = r'A'
    B = r'B'
    C = r'C'
    in_folder = r'train'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    A_path = os.path.join(input_path, A, in_folder)
    B_path = os.path.join(input_path, B, in_folder)
    C_path = os.path.join(input_path, C, in_folder)

    img_list = os.listdir(A_path)
    for img in img_list:
        if img.split('.')[0] in name_list:
            this_img_A = os.path.join(A_path, img)
            this_img_B = os.path.join(B_path, img)
            this_img_C = os.path.join(C_path, img)
            to_img_A = os.path.join(output_path, img.split('.')[0] + '~A.' + img.split('.')[-1])
            to_img_B = os.path.join(output_path, img.split('.')[0] + '~B.' + img.split('.')[-1])
            to_img_C = os.path.join(output_path, img.split('.')[0] + '~C.' + img.split('.')[-1])
            shutil.copy(this_img_A, to_img_A)
            shutil.copy(this_img_B, to_img_B)
            shutil.copy(this_img_C, to_img_C)


def choose_AEimages_allZ_bat(input_path, output_path, prefix, used_S, used_M=[7, 8, 9, 12, 13, 14, 17, 18, 19]):
    for s in used_S:
        for m in used_M:
            choose_AEimages_for_allZ(input_path, output_path, prefix, s, m)


def choose_AEimages_for_allZ(input_path, output_path, prefix, used_S, used_M, B=1, T=1, all_S=96, allZ=11, C=1,
                             all_M=25):
    # get_CZI_image(path, B, T, S, Z, C, M) return [img_path, 'S1', '2018-09-03', 'I-1_CD09', 'T1', 'Z1', 'C1', 'M1']
    # get_CZI_image(path, B, T, S, Z, C, M) return [       0,    1,            2,          3,    4,    5,    6,    7]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    z = 1
    z_img = get_CZI_image(input_path, B, T, used_S, z, C, used_M)
    while z_img is not None:
        img_name = prefix + '~' + z_img[1] + '~' + z_img[7] + '~' + z_img[5] + '.png'
        to_file = os.path.join(output_path, img_name)
        shutil.copy(z_img[0], to_file)
        z += 1
        z_img = get_CZI_image(input_path, B, T, used_S, z, C, used_M)


def choose_AEimages_to_paired_HR_LR_GT(input_path, output_path, prefix, B=1, T=1, all_S=96, allZ=11, C=1, all_M=25,
                                       used_M=[7, 8, 9, 12, 13, 14, 17, 18, 19], HR_path=r'A', LR_path=r'B',
                                       GT_path=r'C'):
    # get_CZI_image(path, B, T, S, Z, C, M) return [img_path, 'S1', '2018-09-03', 'I-1_CD09', 'T1', 'Z1', 'C1', 'M1']
    # get_CZI_image(path, B, T, S, Z, C, M) return [       0,    1,            2,          3,    4,    5,    6,    7]
    # used_M=[6,7,8,11,12,13,16,17,18]
    in_folder = r'train'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    HR_path = os.path.join(output_path, HR_path, in_folder)
    if not os.path.exists(HR_path):
        os.makedirs(HR_path)
    LR_path = os.path.join(output_path, LR_path, in_folder)
    if not os.path.exists(LR_path):
        os.makedirs(LR_path)
    GT_path = os.path.join(output_path, GT_path, in_folder)
    if not os.path.exists(GT_path):
        os.makedirs(GT_path)

    for S in range(1, all_S + 1):
        for M in used_M:

            z = 1
            z_img = get_CZI_image(input_path, B, T, S, z, C, M)
            z_imgVar_list = []
            while z_img is not None:
                z_imgVar_list.append(get_ImageVar(z_img[0]))
                z += 1
                z_img = get_CZI_image(input_path, B, T, S, z, C, M)
            if z_imgVar_list:  # this M is existed
                Z_good = z_imgVar_list.index(max(z_imgVar_list)) + 1
                Z_bad = z_imgVar_list.index(min(z_imgVar_list)) + 1
            else:  # this M is not existed!
                print('!ERROR! Missing', input_path, 'B=', B, 'T=', T, 'S=', S, 'z=', z, 'C=', C, 'M=', M)

            GT_source_img_path = get_CZI_image(input_path, B, T, S, Z_good, C, M)
            img_name = prefix + '~' + GT_source_img_path[1] + '~' + GT_source_img_path[7] + '.png'  # 'prefix~S1~M1'
            img_original = image_to_gray(GT_source_img_path[0])
            img_enhanced = trans_CLAHE(trans_Unsharp_Masking(img_original))
            img_blur = image_to_gray(get_CZI_image(input_path, B, T, S, Z_bad, C, M)[0])
            cv2.imwrite(os.path.join(HR_path, img_name), img_enhanced)
            cv2.imwrite(os.path.join(LR_path, img_name), img_blur)
            cv2.imwrite(os.path.join(GT_path, img_name), img_original)
            print('Choose: Good_focus Bad_focus GT from ', img_name)


def generate_folder_to_paired_GT_LR(input_path, output_path, GT_path=r'A', LR_path=r'B', size=(256, 256)):
    # generate training data set (paired LR GT) for CIEGAN
    # input_path: images folder
    # output_path: contains A B folder; and inside: train test val

    if not os.path.exists(input_path):
        print('!ERROR! The input_path does not existed!')
        return False

    in_folder = r'train'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    GT_path = os.path.join(output_path, GT_path, in_folder)
    if not os.path.exists(GT_path):
        os.makedirs(GT_path)
    LR_path = os.path.join(output_path, LR_path, in_folder)
    if not os.path.exists(LR_path):
        os.makedirs(LR_path)

    input_list = os.listdir(input_path)

    for i in input_list:  # r'Label_1.png'
        img_input = os.path.join(input_path, i)
        img_original = image_to_gray(img_input)
        img_resized = cv2.resize(img_original, size, interpolation=cv2.INTER_NEAREST)
        # img_enhanced = trans_CLAHE(trans_Unsharp_Masking(img_original))
        img_blur = trans_blur(img_resized)
        cv2.imwrite(os.path.join(GT_path, i), img_resized)
        cv2.imwrite(os.path.join(LR_path, i), img_blur)
        print('Generated from: ', i)

    A_B_AB(output_path, A=r'A', B=r'B')

    return


def generate_AEimages_to_paired_HR_LR_GT(input_path, output_path, prefix, B=1, T=1, all_S=96, Z=1, C=1, all_M=25,
                                         used_M=[7, 8, 9, 12, 13, 14, 17, 18, 19], HR_path=r'A', LR_path=r'B',
                                         GT_path=r'C'):
    # get_CZI_image(path, B, T, S, Z, C, M) return [img_path, 'S1', '2018-09-03', 'I-1_CD09', 'T1', 'Z1', 'C1', 'M1']
    # used_M=[6,7,8,11,12,13,16,17,18]
    in_folder = r'train'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    HR_path = os.path.join(output_path, HR_path, in_folder)
    if not os.path.exists(HR_path):
        os.makedirs(HR_path)
    LR_path = os.path.join(output_path, LR_path, in_folder)
    if not os.path.exists(LR_path):
        os.makedirs(LR_path)
    GT_path = os.path.join(output_path, GT_path, in_folder)
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
            print('Generated: HR LR GT from ', img_name)

    return


def generate_AEimages_to_paired_HR_LR_GT_forCD09(input_path, output_path, prefix='CD09',
                                                 T=['2018-09-13~F_CD09~T1', '2018-09-17~Result_CD09~T1~C1',
                                                    '2018-09-17~Result_CD09~T1~C3', '2018-09-17~Result_CD09~T1~C4'],
                                                 S=[2, 3, 6, 10, 11, 14, 15, 23],
                                                 HR_path=r'A', LR_path=r'B', GT_path=r'C', in_pix=128):
    # image_cut_n2_blocks(img_original, output_path=None, n=11, image_name=None, del_index=None, save=True, to_gray=True)
    n = 11
    in_folder = r'train'
    del_index = [0, 1, 9, 10, 11, 21, 99, 109, 110, 111, 119, 120]
    subpic_pix = 1844
    # in_pix = 256
    cut_n = int(subpic_pix / in_pix)
    cut_resize = (in_pix * cut_n, in_pix * cut_n)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    HR_path = os.path.join(output_path, HR_path, in_folder)
    if not os.path.exists(HR_path):
        os.makedirs(HR_path)
    LR_path = os.path.join(output_path, LR_path, in_folder)
    if not os.path.exists(LR_path):
        os.makedirs(LR_path)
    GT_path = os.path.join(output_path, GT_path, in_folder)
    if not os.path.exists(GT_path):
        os.makedirs(GT_path)

    for s in S:
        this_path = os.path.join(input_path, 'S' + str(s))
        path_list = os.listdir(this_path)
        for img in path_list:
            if img.split('.')[0] in T:
                img_path = os.path.join(this_path, img)

                img_name = prefix + '~' + 'S' + str(s) + '.png'
                img_original = image_to_gray(img_path)
                img_enhanced = trans_CLAHE(trans_Unsharp_Masking(img_original))
                img_blur = trans_blur(img_original)

                img_original = image_cut_n2_blocks(img_original, n=11, del_index=del_index, save=False)
                img_enhanced = image_cut_n2_blocks(img_enhanced, n=11, del_index=del_index, save=False)
                img_blur = image_cut_n2_blocks(img_blur, n=11, del_index=del_index, save=False)

                for index, image in enumerate(img_original):
                    image_name = img_name.split('.')[0] + '_' + str(index) + '.' + img_name.split('.')[-1]
                    image = cv2.resize(image, cut_resize, interpolation=cv2.INTER_NEAREST)
                    image_cut_n2_blocks(image, output_path=GT_path, n=cut_n, image_name=image_name)
                for index, image in enumerate(img_enhanced):
                    image_name = img_name.split('.')[0] + '_' + str(index) + '.' + img_name.split('.')[-1]
                    image = cv2.resize(image, cut_resize, interpolation=cv2.INTER_NEAREST)
                    image_cut_n2_blocks(image, output_path=HR_path, n=cut_n, image_name=image_name)
                for index, image in enumerate(img_blur):
                    image_name = img_name.split('.')[0] + '_' + str(index) + '.' + img_name.split('.')[-1]
                    image = cv2.resize(image, cut_resize, interpolation=cv2.INTER_NEAREST)
                    image_cut_n2_blocks(image, output_path=LR_path, n=cut_n, image_name=image_name)
                print('Generated: HR LR GT from ', img_name)

    return


def image_resize(in_, out_, size=(256, 256)):
    # resize all images in in_ and output in out_
    # in_ or out_ can be a image file path or dir path
    # if out_ is a dir, it must existed

    if os.path.isfile(in_):
        name_img = os.path.split(in_)[-1]
        o_img = any_to_image(in_)
        d_img = cv2.resize(o_img, size, interpolation=cv2.INTER_NEAREST)
        print(out_)
        if os.path.isdir(out_):
            cv2.imwrite(os.path.join(out_, name_img), d_img)
        else:
            cv2.imwrite(out_, d_img)
    elif os.path.isdir(in_):
        path_list = os.listdir(in_)
        for i in path_list:  # r'Label_1.png'
            img_dirfile = os.path.join(in_, i)
            image_resize(img_dirfile, out_, size=size)
    else:
        print('!ERROR! The input path or image does not existed!')
        return False

    return True


def folder_image_resize(image_path, size=(2580, 2580)):
    # resize all images in sub-folder and overwrite itself

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
    # resize images in folder image_path and overwrite itself

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


def folder_image_cut_n2_blocks(image_path, output_path, n=3, to_gray=False):
    if not os.path.exists(image_path):
        print('!ERROR! The image_path does not existed!')
        return False
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    path_list = os.listdir(image_path)
    for this_img in path_list:  # r'Label_1.png'
        img_file = os.path.join(image_path, this_img)
        if to_gray:
            o_img = image_to_gray(img_file)
        else:
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
            # print(o_file)
            cv2.imwrite(o_file, image)
            index += 1

        print('Cut ', this_img, ' into ', str(n), ' blocks!')

    return True


def image_cut_n2_blocks(image_in, output_path=None, n=11, image_name=None, del_index=None, save=True, to_gray=True):
    if type(image_in) is str:
        image_name = os.path.split(image_in)[-1]
    elif save and image_name is None:
        print('!ERROR! If save and input is ndarray, must given a image name(exp:"1.png")!')
        return False

    if to_gray:
        o_img = image_to_gray(image_in)
    else:
        o_img = any_to_image(image_in)

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

    if del_index is not None:
        # for i in del_index:
        #     del(image_list[i])
        # image_list_out = [x for x in image_list if image_list.index(x) not in del_index]
        keep_index = [i for i in range(n * n) if i not in del_index]
        results = [image_list[i] for i in keep_index]
    else:
        results = image_list

    if save:
        if output_path is None:
            print('!Warning! output_path was not given!')
            return results
        elif not os.path.exists(output_path):
            os.makedirs(output_path)

        for index, image in enumerate(results):
            o_file = os.path.join(output_path,
                                  image_name.split('.')[0] + '_' + str(index) + '.' + image_name.split('.')[-1])
            # if del_index is not None and index in del_index:
            #     continue
            cv2.imwrite(o_file, image)
        print('Cut ', image_name, ' into ', str(n), ' blocks!')

    return results


def folder_image_selected_cut_n2_blocks(image_path, output_path, n=16, used_CD=None, used_S=None, used_M=None,
                                        to_gray=False):
    if not os.path.exists(image_path):
        print('!ERROR! The image_path does not existed!')
        return False
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    path_list = os.listdir(image_path)
    for this_img in path_list:  # r'CD11~S1~M7.png'
        M = int(this_img.split('~M')[-1].split('.')[0])
        S = int(this_img.split('~S')[-1].split('~')[0])
        CD = int(this_img.split('~')[0].split('CD')[-1])

        if CD in used_CD and S in used_S and M in used_M:
            img_file = os.path.join(image_path, this_img)
            if to_gray:
                o_img = image_to_gray(img_file)
            else:
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
                # print(o_file)
                cv2.imwrite(o_file, image)
                index += 1

            print('Cut ', this_img, ' into ', str(n), ' blocks!')

    return True


def image2_write(path_A, path_B, path_AB):
    im_A = cv2.imread(path_A, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv2.imread(path_B, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_AB = np.concatenate([im_A, im_B], 1)
    cv2.imwrite(path_AB, im_AB)


def A_B_AB(main_path, A=r'A', B=r'B'):
    image_file_type = ('.jpg', '.png', '.tif')

    if not os.path.exists(main_path):
        print('!ERROR! The main_path does not existed!')
        return False

    A_path = os.path.join(main_path, A)
    if not os.path.exists(A_path):
        print('!ERROR! The ' + A + '_path does not existed!')
        return False

    B_path = os.path.join(main_path, B)
    if not os.path.exists(A_path):
        print('!ERROR! The ' + B + '_path does not existed!')
        return False

    AB_path = os.path.join(main_path, A + B)
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
            print('!ERROR! The ' + B + '_sub_path does not existed!')
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
                    print('!ERROR! The this_img' + B + '_path does not existed!')
                    continue
                image2_write(this_imgA_path, this_imgB_path, this_imgAB_path)
                print('Merge ', this_img, ' into ' + A + B + ' style!')


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


def test(input_path, output_path):
    if not os.path.exists(input_path):
        print('!ERROR! The input_path does not existed!')
        return False
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    path_list = os.listdir(input_path)

    for i in path_list:  # r'Label_1.png'
        in_img = os.path.join(input_path, i)
        new_img = trans_CLAHE(in_img)
        cv2.imwrite(os.path.join(output_path, i), new_img)
    return True


if __name__ == '__main__':
    input_path = r'C:\DATA\CIEGAN_eval\CIEGAN_eval_ad_1\GT'
    output_path = r'C:\DATA\CIEGAN_eval\CIEGAN_eval_ad_1\test'
    test(input_path, output_path)

    # input_path = r'C:\DATA\KAGGEL_HPAIC\train'
    # output_path = r'C:\DATA\KAGGEL_HPAIC\forAB'
    # generate_folder_to_paired_GT_LR(input_path, output_path, GT_path=r'A', LR_path=r'B', size=(256, 256))

    # A_B_AB(r'C:\DATA\KAGGEL_HPAIC')

    # cut_dir_center(r'D:\Green\Sub_Projects\ML_assists_hiPSC-CM\selected_wells\i',r'D:\Green\Sub_Projects\ML_assists_hiPSC-CM\selected_wells\o',gray=True)

    # dir_enhanced2(r'E:\Coral\Selected\SSS_PS', r'E:\Coral\Selected\SSS_final')

    # dir_enhanced(r'D:\Green\Sub_Projects\ML_assists_hiPSC-CM\selected_wells\Selected',
    #              r'D:\Green\Sub_Projects\ML_assists_hiPSC-CM\selected_wells\TE')

    # folder_image_cut_n2_blocks(r'E:\Data\Living_2048\A\train', r'E:\Data\Living_256\A\train', n=8, to_gray=True)
    # folder_image_cut_n2_blocks(r'E:\Data\Living_2048\B\train', r'E:\Data\Living_256\B\train', n=8, to_gray=True)
    # folder_image_cut_n2_blocks(r'E:\Data\Living_2048\C\train', r'E:\Data\Living_256\C\train', n=8, to_gray=True)
    # A_B_AB(r'E:\Data\Living_256')

    # folder_image_cut_n2_blocks(r'E:\Data\cTnT_2048\A\train', r'E:\Data\cTnT_256\A\train', n=8, to_gray=True)
    # folder_image_cut_n2_blocks(r'E:\Data\cTnT_2048\B\train', r'E:\Data\cTnT_256\B\train', n=8, to_gray=True)
    # folder_image_cut_n2_blocks(r'E:\Data\cTnT_2048\C\train', r'E:\Data\cTnT_256\C\train', n=8, to_gray=True)
    # folder_image_cut_n2_blocks(r'E:\Data\DAPI_2048\A\train', r'E:\Data\DAPI_256\A\train', n=8, to_gray=True)
    # folder_image_cut_n2_blocks(r'E:\Data\DAPI_2048\B\train', r'E:\Data\DAPI_256\B\train', n=8, to_gray=True)
    # folder_image_cut_n2_blocks(r'E:\Data\DAPI_2048\C\train', r'E:\Data\DAPI_256\C\train', n=8, to_gray=True)
    # A_B_AB(r'E:\Data\cTnT_256')
    # A_B_AB(r'E:\Data\DAPI_256')

    # input_path = r'E:\Data\CD09_for_CIEGAN'
    # output_path = r'E:\Data\CD09_DAPI_256'
    # generate_AEimages_to_paired_HR_LR_GT_forCD09(input_path, output_path, prefix='CD09',
    #                                              T=['2018-09-17~Result_CD09~T1~C3'], in_pix=256)
    # A_B_AB(r'E:\Data\CD09_DAPI_256')
    # A_B_AB(r'E:\Data\CD09_DAPI_256', A='C', B=r'B')
    #
    # output_path = r'E:\Data\CD09_cTnT_256'
    # generate_AEimages_to_paired_HR_LR_GT_forCD09(input_path, output_path, prefix='CD09',
    #                                              T=['2018-09-17~Result_CD09~T1~C4'], in_pix=256)
    # A_B_AB(r'E:\Data\CD09_cTnT_256')
    # A_B_AB(r'E:\Data\CD09_cTnT_256', A='C', B=r'B')

    # input_path = r'E:\Data\CD09_Bright_256'
    # output_path = r'E:\Data\Selected_CD09_Bright_256'
    # name_list = ['CD09~S3_33_2', 'CD09~S3_52_35', 'CD09~S3_62_11', 'CD09~S10_19_41', 'CD09~S10_23_42', 'CD09~S10_23_46',
    #              'CD09~S10_28_44', 'CD09~S10_29_21', 'CD09~S10_29_46', 'CD09~S10_30_0', 'CD09~S10_30_35',
    #              'CD09~S10_30_36', 'CD09~S10_30_37', 'CD09~S10_33_23', 'CD09vS10_53_29', 'CD09~S10_77_22',
    #              'CD09~S10_77_23']
    # choose_test_ABC(input_path, output_path, name_list=name_list)

    # input_path = r'D:\Z\CD58A_Result'
    # output_path = r'C:\DATA\Z\CD58A'
    # used_S = [3, 4, 6, 15, 18, 30, 31, 37, 38, 46, 47, 48]
    # choose_AEimages_allZ_bat(input_path, output_path, r'CD58A', used_S)
    #
    # input_path = r'D:\Z\CD58B_Result'
    # output_path = r'C:\DATA\Z\CD58B'
    # used_S = [2, 3, 4, 6, 9, 10, 15, 19, 22, 26, 30, 34, 35, 39, 44, 47, 48]
    # choose_AEimages_allZ_bat(input_path, output_path, r'CD58B', used_S)
    #
    # input_path = r'D:\Z\CD61_d12_liveCM'
    # output_path = r'C:\DATA\Z\CD61'
    # used_S = [i for i in range(4, 11)] + [13, 19, 20, 21] + [i for i in range(24, 32)] + [i for i in range(43, 55)] + [
    #     58, 59] + [i for i in range(62, 83)]
    # choose_AEimages_allZ_bat(input_path, output_path, r'CD61', used_S)
    #
    # input_path = r'D:\Z\CD63_d14_liveCM'
    # output_path = r'C:\DATA\Z\CD63'
    # used_S = [3, 4, 5, 7, 8, 9, 10] + [i for i in range(13, 21)] + [i for i in range(19, 45)] + \
    #          [i for i in range(49, 97)]
    # choose_AEimages_allZ_bat(input_path, output_path, r'CD63', used_S)
    #
    # input_path = r'D:\Z\CD64A_d14_liveCM'
    # output_path = r'C:\DATA\Z\CD64'
    # used_S = [3, 4, 5, 7, 8] + [i for i in range(14, 22)] + [i for i in range(27, 35)] + [i for i in range(38, 47)] + \
    #          [i for i in range(49, 97)]
    # choose_AEimages_allZ_bat(input_path, output_path, r'CD64', used_S)

    # for s in use_S:
    #     for m in [7, 8, 9, 12, 13, 14, 17, 18, 19]:
    #         choose_AEimages_for_allZ(input_path, output_path, prefix, s, m)

    # input_path = r'E:\Coral\CD11\2018-11-20\2018-11-20_1100_F_CD11'
    # output_path = r'C:\DATA\Z\CD11'
    # prefix = r'CD11'
    # for s in [2, 3, 4, 5, 7, 8, 17, 20, 22, 28, 32, 51, 70, 74, 75, 76, 77, 78, 79, 80, 94]:
    #     for m in [7, 8, 9, 12, 13, 14, 17, 18, 19]:
    #         choose_AEimages_for_allZ(input_path, output_path, prefix, s, m)

    # A_B_AB(r'E:\Data\CD09_Bright', A='C', B=r'B')
    # A_B_AB(r'E:\Data\CD09_Bright_256', A='C', B=r'B')

    # input_path = r'E:\Data\CD09_for_CIEGAN'
    # output_path = r'E:\Data\CD09_Bright_512'
    # generate_AEimages_to_paired_HR_LR_GT_forCD09(input_path, output_path, prefix='CD09', T=['2018-09-13~F_CD09~T1'],
    #                                              in_pix=512)
    # A_B_AB(r'E:\Data\CD09_Bright_512')

    # input_path = r'E:\Data\CD09_for_CIEGAN'
    # output_path = r'E:\Data\CD09_Bright_256'
    # generate_AEimages_to_paired_HR_LR_GT_forCD09(input_path, output_path, prefix='CD09', T=['2018-09-13~F_CD09~T1'], in_pix=256)
    # A_B_AB(r'E:\Data\CD09_Bright_256')

    # A_B_AB(r'E:\Data\CD09_Bright')
    # A_B_AB(r'E:\Data\CD09_Bright_d')
    # A_B_AB(r'E:\Data\CD09_DAPI')
    # A_B_AB(r'E:\Data\CD09_cTnT')

    # input_path = r'E:\Data\CD09_for_CIEGAN'
    # output_path = r'E:\Data\CD09_Bright'
    # generate_AEimages_to_paired_HR_LR_GT_forCD09(input_path, output_path, prefix='CD09',
    #                                              T=['2018-09-13~F_CD09~T1'])
    # input_path = r'E:\Data\CD09_for_CIEGAN'
    # output_path = r'E:\Data\CD09_Bright_d'
    # generate_AEimages_to_paired_HR_LR_GT_forCD09(input_path, output_path, prefix='CD09',
    #                                              T=['2018-09-17~Result_CD09~T1~C1'])
    #
    # input_path = r'E:\Data\CD09_for_CIEGAN'
    # output_path = r'E:\Data\CD09_DAPI'
    # generate_AEimages_to_paired_HR_LR_GT_forCD09(input_path, output_path, prefix='CD09',
    #                                              T=['2018-09-17~Result_CD09~T1~C3'])
    #
    # input_path = r'E:\Data\CD09_for_CIEGAN'
    # output_path = r'E:\Data\CD09_cTnT'
    # generate_AEimages_to_paired_HR_LR_GT_forCD09(input_path, output_path, prefix='CD09',
    #                                              T=['2018-09-17~Result_CD09~T1~C4'])

    # input_path = r'E:\Coral\CD09\SSS_100%'
    # output_path = r'C:\DATA\CD09_Bright'
    # generate_AEimages_to_paired_HR_LR_GT_forCD09(input_path, output_path, prefix='CD09',
    #                                              T=['2018-09-13~F_CD09~T1'])
    # input_path = r'E:\Coral\CD09\SSS_100%'
    # output_path = r'C:\DATA\CD09_Bright_d'
    # generate_AEimages_to_paired_HR_LR_GT_forCD09(input_path, output_path, prefix='CD09',
    #                                              T=['2018-09-17~Result_CD09~T1~C1'])
    #
    # input_path = r'E:\Coral\CD09\SSS_100%'
    # output_path = r'C:\DATA\CD09_DAPI'
    # generate_AEimages_to_paired_HR_LR_GT_forCD09(input_path, output_path, prefix='CD09',
    #                                              T=['2018-09-17~Result_CD09~T1~C3'])
    #
    # input_path = r'E:\Coral\CD09\SSS_100%'
    # output_path = r'C:\DATA\CD09_cTnT'
    # generate_AEimages_to_paired_HR_LR_GT_forCD09(input_path, output_path, prefix='CD09',
    #                                              T=['2018-09-17~Result_CD09~T1~C4'])

    # image_file = r'C:\DATA\CIEGAN_eval_5\2018-09-13~F_CD09~T1.png'
    # output_path = r'C:\DATA\CIEGAN_eval_5\GT'
    # image_cut_n2_blocks(image_file, output_path, n=11, del_index=[0, 1, 9, 10, 11, 21, 99, 109, 110, 111, 119, 120])

    # input_path = r'C:\DATA\CD11\2018-11-20_1100_F_CD11'
    # output_path = r'C:\DATA\Living_2048'
    # prefix = r'CD11F'
    # choose_AEimages_to_paired_HR_LR_GT(input_path, output_path, prefix, B=1, T=1, all_S=96, allZ=11, C=1, all_M=25,
    #                                    used_M=[7, 8, 9, 12, 13, 14, 17, 18, 19], HR_path=r'A', LR_path=r'B',
    #                                    GT_path=r'C')
    # folder_image_cut_n2_blocks(r'C:\Data\Living_2048\A\train', r'C:\Data\Living_128\A\train', n=16)
    # folder_image_cut_n2_blocks(r'C:\Data\Living_2048\B\train', r'C:\Data\Living_128\B\train', n=16)
    # folder_image_cut_n2_blocks(r'C:\Data\Living_2048\C\train', r'C:\Data\Living_128\C\train', n=16)
    # A_B_AB(r'C:\Data\Living_128')

    # input_path = r'C:\DATA\CD13\2018-12-17\Result3_CD13'
    # output_path = r'C:\DATA\DAPI_2048'
    # generate_AEimages_to_paired_HR_LR_GT(input_path, output_path, 'CD13', C=3)
    # input_path = r'C:\DATA\CD11\2018-11-25_Result_CD11'
    # generate_AEimages_to_paired_HR_LR_GT(input_path, output_path, 'CD11', C=1)
    #
    # used_CD = [13]
    # used_S = [3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 2, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    #           35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 76,
    #           77, 78, 79, 81, 82, 85]
    # used_M = [8, 12, 13, 14, 18]
    #
    # folder_image_selected_cut_n2_blocks(r'C:\DATA\DAPI_2048\A\train', r'C:\DATA\DAPI_128\A\train', n=16,
    #                                     used_CD=used_CD, used_S=used_S, used_M=used_M)
    # folder_image_selected_cut_n2_blocks(r'C:\DATA\DAPI_2048\B\train', r'C:\DATA\DAPI_128\B\train', n=16,
    #                                     used_CD=used_CD, used_S=used_S, used_M=used_M)
    # folder_image_selected_cut_n2_blocks(r'C:\DATA\DAPI_2048\C\train', r'C:\DATA\DAPI_128\C\train', n=16,
    #                                     used_CD=used_CD, used_S=used_S, used_M=used_M)
    #
    # used_CD = [11]
    # used_S = [2, 3, 4, 5, 8, 9, 17, 19, 20, 21, 22, 23, 26, 27, 28, 31, 32, 42, 43, 45, 46, 47, 50, 51, 55, 66, 70, 71,
    #           74, 75, 76, 77, 78, 79, 80, 90, 94, 95]
    # used_M = [8, 12, 13, 14, 18]
    #
    # folder_image_selected_cut_n2_blocks(r'C:\DATA\DAPI_2048\A\train', r'C:\DATA\DAPI_128\A\train', n=16,
    #                                     used_CD=used_CD, used_S=used_S, used_M=used_M)
    # folder_image_selected_cut_n2_blocks(r'C:\DATA\DAPI_2048\B\train', r'C:\DATA\DAPI_128\B\train', n=16,
    #                                     used_CD=used_CD, used_S=used_S, used_M=used_M)
    # folder_image_selected_cut_n2_blocks(r'C:\DATA\DAPI_2048\C\train', r'C:\DATA\DAPI_128\C\train', n=16,
    #                                     used_CD=used_CD, used_S=used_S, used_M=used_M)
    #
    # A_B_AB(r'C:\Data\DAPI_128')

    # input_path = r'C:\DATA\CD13\2018-12-17\Result3_CD13'
    # output_path = r'C:\DATA\cTnT_2048'
    # generate_AEimages_to_paired_HR_LR_GT(input_path, output_path, 'CD13', C=2)
    # input_path = r'C:\DATA\CD11\2018-11-25_Result_CD11'
    # generate_AEimages_to_paired_HR_LR_GT(input_path, output_path, 'CD11', C=2)
    #
    # used_CD = [13]
    # used_S = [3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 2, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    #           35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 76,
    #           77, 78, 79, 81, 82, 85]
    # used_M = [8, 12, 13, 14, 18]
    #
    # folder_image_selected_cut_n2_blocks(r'C:\DATA\cTnT_2048\A\train', r'C:\DATA\cTnT_128\A\train', n=16,
    #                                     used_CD=used_CD, used_S=used_S, used_M=used_M)
    # folder_image_selected_cut_n2_blocks(r'C:\DATA\cTnT_2048\B\train', r'C:\DATA\cTnT_128\B\train', n=16,
    #                                     used_CD=used_CD, used_S=used_S, used_M=used_M)
    # folder_image_selected_cut_n2_blocks(r'C:\DATA\cTnT_2048\C\train', r'C:\DATA\cTnT_128\C\train', n=16,
    #                                     used_CD=used_CD, used_S=used_S, used_M=used_M)
    #
    # used_CD = [11]
    # used_S = [2, 3, 4, 5, 8, 9, 17, 19, 20, 21, 22, 23, 26, 27, 28, 31, 32, 42, 43, 45, 46, 47, 50, 51, 55, 66, 70, 71,
    #           74, 75, 76, 77, 78, 79, 80, 90, 94, 95]
    # used_M = [8, 12, 13, 14, 18]
    #
    # folder_image_selected_cut_n2_blocks(r'C:\DATA\cTnT_2048\A\train', r'C:\DATA\cTnT_128\A\train', n=16,
    #                                     used_CD=used_CD, used_S=used_S, used_M=used_M)
    # folder_image_selected_cut_n2_blocks(r'C:\DATA\cTnT_2048\B\train', r'C:\DATA\cTnT_128\B\train', n=16,
    #                                     used_CD=used_CD, used_S=used_S, used_M=used_M)
    # folder_image_selected_cut_n2_blocks(r'C:\DATA\cTnT_2048\C\train', r'C:\DATA\cTnT_128\C\train', n=16,
    #                                     used_CD=used_CD, used_S=used_S, used_M=used_M)
    #
    # A_B_AB(r'C:\Data\cTnT_128')

    # input_path = r'C:\DATA\CD13\2018-12-17\Result3_CD13'
    # output_path = r'C:\DATA\Bright_2048'
    # generate_AEimages_to_paired_HR_LR_GT(input_path, output_path, 'CD13', C=1)
    # input_path = r'C:\DATA\CD11\2018-11-25_Result_CD11'
    # generate_AEimages_to_paired_HR_LR_GT(input_path, output_path, 'CD11', C=3)
    #
    # used_CD = [13]
    # used_S = [3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 2, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    #           35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 76,
    #           77, 78, 79, 81, 82, 85]
    # used_M = [8, 12, 13, 14, 18]
    #
    # folder_image_selected_cut_n2_blocks(r'C:\DATA\Bright_2048\A\train', r'C:\DATA\Bright_128\A\train', n=16,
    #                                     used_CD=used_CD, used_S=used_S, used_M=used_M)
    # folder_image_selected_cut_n2_blocks(r'C:\DATA\Bright_2048\B\train', r'C:\DATA\Bright_128\B\train', n=16,
    #                                     used_CD=used_CD, used_S=used_S, used_M=used_M)
    # folder_image_selected_cut_n2_blocks(r'C:\DATA\Bright_2048\C\train', r'C:\DATA\Bright_128\C\train', n=16,
    #                                     used_CD=used_CD, used_S=used_S, used_M=used_M)
    #
    # used_CD = [11]
    # used_S = [2, 3, 4, 5, 8, 9, 17, 19, 20, 21, 22, 23, 26, 27, 28, 31, 32, 42, 43, 45, 46, 47, 50, 51, 55, 66, 70, 71,
    #           74, 75, 76, 77, 78, 79, 80, 90, 94, 95]
    # used_M = [8, 12, 13, 14, 18]
    #
    # folder_image_selected_cut_n2_blocks(r'C:\DATA\Bright_2048\A\train', r'C:\DATA\Bright_128\A\train', n=16,
    #                                     used_CD=used_CD, used_S=used_S, used_M=used_M)
    # folder_image_selected_cut_n2_blocks(r'C:\DATA\Bright_2048\B\train', r'C:\DATA\Bright_128\B\train', n=16,
    #                                     used_CD=used_CD, used_S=used_S, used_M=used_M)
    # folder_image_selected_cut_n2_blocks(r'C:\DATA\Bright_2048\C\train', r'C:\DATA\Bright_128\C\train', n=16,
    #                                     used_CD=used_CD, used_S=used_S, used_M=used_M)

    # folder_image_cut_n2_blocks(r'C:\Data\Bright_2048\A\train', r'C:\Data\Bright_128\A\train', n=16)
    # folder_image_cut_n2_blocks(r'C:\Data\Bright_2048\B\train', r'C:\Data\Bright_128\B\train', n=16)
    # folder_image_cut_n2_blocks(r'C:\Data\Bright_2048\C\train', r'C:\Data\Bright_128\C\train', n=16)

    # A_B_AB(r'C:\Data\Bright_128')

    # input_path = r'E:\Image_Processing\CD13\2018-12-17\Result3_CD13'
    # output_path = r'C:\Users\Kitty\Desktop\CD13_Bright_2048'
    # copy_AEimages_to_paired_HR_LR_GT(input_path, output_path, 'CD13')

    # folder_image_cut_n2_blocks(r'C:\Users\Kitty\Desktop\CD13_Bright_2048\A', r'C:\Users\Kitty\Desktop\CD13_Bright_128\A', n=16)
    # folder_image_cut_n2_blocks(r'C:\Users\Kitty\Desktop\CD13_Bright_2048\B', r'C:\Users\Kitty\Desktop\CD13_Bright_128\B', n=16)
    # folder_image_cut_n2_blocks(r'C:\Users\Kitty\Desktop\CD13_Bright_2048\C', r'C:\Users\Kitty\Desktop\CD13_Bright_128\C', n=16)
    # A_B_AB(r'C:\Users\Kitty\Desktop\CD13_Bright_128')

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
