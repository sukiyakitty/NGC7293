import os
import numpy as np
import cv2


def any_to_image(img):
    # core methods
    # image pre processing, can processing any image form to np.ndarray format
    # input img can be path str
    # output img is cv2 np.ndarray (colored or original)!

    if type(img) is str:
        if not os.path.exists(img):
            print('!ERROR! The image path does not existed!')
            return None
        img = cv2.imread(img, 1)  # BGR .shape=(h,w,3)
        img = np.uint8(img)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif type(img) is np.ndarray:
        img = np.uint8(img)
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = img[:, :, 0:3]
            # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:
            # img_gray = img
            print('!NOTICE! The input image is gray!')
            # pass
        else:
            print('!ERROR! The image shape error!')
            return None
    else:
        print('!ERROR! Please input correct CV2 image file or file path!')
        return None

    return img


def folder_image_resize(image_path, size=(2480, 2480)):
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
            folder_image_resize(img_dirfile)

    return True


def folder_image_resize_0(image_path, size=(2480, 2480)):
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


def folder_image_cut_n_blocks(image_path, output_path, n=3):
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


def image_write(path_A, path_B, path_AB):
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
                image_write(this_imgA_path, this_imgB_path, this_imgAB_path)


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
    image_path = r'C:\DATA\cTnT_original_2580'
    folder_image_resize(image_path, size=(2580, 2580))
    remove_suffix_2(r'C:\DATA\cTnT_original_2580\A\train')
    remove_suffix_2(r'C:\DATA\cTnT_original_2580\B\train')
    folder_image_cut_n_blocks(r'C:\DATA\cTnT_original_2580\A\train',  r'C:\DATA\cTnT_original_860\A\train', n=3)
    folder_image_cut_n_blocks(r'C:\DATA\cTnT_original_2580\B\train',  r'C:\DATA\cTnT_original_860\B\train', n=3)
    A_B_AB(r'C:\DATA\cTnT_original_860')
