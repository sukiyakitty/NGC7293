import sys
import os
import shutil
import argparse


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False


def my_rename(file_path):
    all_files = os.listdir(file_path)
    for this_file in all_files:
        # print(this_file)
        if this_file[:3] == 'IPS':
            os.rename(os.path.join(file_path, this_file), os.path.join(file_path, 'I-0' + this_file[3:]))


def my_mkdir(file_path):
    well = 72
    z_direction = 3
    for i in range(1, well + 1):
        if i < 10:
            S = 's0' + str(i)
        else:
            S = 's' + str(i)
        for j in range(1, z_direction + 1):
            Z = 'z' + str(j)
            this_name = os.path.join(file_path, S, Z)
            mkdir(this_name)


def run72(file_path, well, z_direction):
    all_files = os.listdir(file_path)
    for this_file in all_files:
        for i in range(1, well + 1):
            if i < 10:
                S = 's0' + str(i)
            else:
                S = 's' + str(i)
            if this_file[-12:-9] == S:
                for j in range(1, z_direction + 1):
                    Z = 'z' + str(j)
                    if this_file[-6:-4] == Z:
                        shutil.move(os.path.join(file_path, this_file),
                                    os.path.join(file_path, S, Z, this_file))
                        # shutil.copyfile(os.path.join(file_path,this_file),os.path.join(file_path,S,Z,this_file))
                        print(S, '', Z, 'has done!')


def run96(file_path, well, z_location):
    all_files = os.listdir(file_path)
    for this_file in all_files:
        for i in range(1, well + 1):
            if i < 10:
                S = 's0' + str(i)
            else:
                S = 's' + str(i)
            if this_file[-9:-6] == S:
                shutil.copyfile(os.path.join(file_path, this_file), os.path.join(file_path, S, z_location, this_file))
                print(S, '', z_location, 'has done!')


def main(args):
    if os.path.exists(args.file_path):
        print('The processing file path: ', args.file_path)
    else:
        exit('INPUT Folder Path does not existing!')
    # my_rename(args.file_path)
    # my_mkdir(args.file_path)
    # run72(args.file_path,72,3)
    run96(args.file_path, 72, 'z2')


def parseArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, nargs='?', default='C:\\Users\\Kitty\\Desktop\\test',
                        help='The ALL the ZEN exported image File path.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parseArguments(sys.argv[1:]))
