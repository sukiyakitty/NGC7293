import os
import math
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine


class CurrentFolderStructs:

    def re_init_vars(self):
        self.path_has_block = False
        self.path_has_time = False
        self.path_has_scene = False
        self.path_has_z_direction = False
        self.path_has_channel = False
        self.path_has_mosaic_tiles = False
        self.current_B = 0  # the B number, start from 1
        self.current_T = 0  # the T number, start from 1
        self.current_S = 0  # the S number, start from 1
        self.current_Z = 0  # the Z number, start from 1
        self.current_C = 0  # the C number, start from 1
        self.current_M = 0  # the M number, start from 1
        self.current_image_file = ''
        self.current_valid_image_file = ''
        self.is_available = False

    def __init__(self, fp):
        self.folder_path = fp
        self.image_file_type = ('.jpg', '.png', '.tif')
        self.path_has_block = False
        self.path_has_time = False
        self.path_has_scene = False
        self.path_has_z_direction = False
        self.path_has_channel = False
        self.path_has_mosaic_tiles = False
        self.current_B = 0  # the B number, start from 1
        self.current_T = 0  # the T number, start from 1
        self.current_S = 0  # the S number, start from 1
        self.current_Z = 0  # the Z number, start from 1
        self.current_C = 0  # the C number, start from 1
        self.current_M = 0  # the M number, start from 1
        self.current_image_file = ''
        self.current_valid_image_file = ''
        self.is_available = False
        if os.path.exists(self.folder_path):
            self.re_init_structs()
            # self.is_available = True

    def re_init_structs(self):
        # the path on disk IS from 0(but in ZEN is from 1)
        self.re_init_vars()
        this_files = os.listdir(self.folder_path)  # finding B folder
        for this_file in this_files:
            if this_file[:1] == 'B':
                self.path_has_block = True
                self.current_B += 1

        if self.path_has_block:  # found B or B=    [x:\CD00\processing\2019-01-01\XEP_name\B]
            this_path = os.path.join(self.folder_path, 'B=' + str(self.current_B - 1))
            if (not os.path.exists(this_path)) and self.current_B == 1:
                this_path = os.path.join(self.folder_path, 'B')
        else:  # not found B or B=    [x:\CD00\processing\2019-01-01\XEP_name]
            this_path = self.folder_path
            self.current_B = 1

        dog = True  # finding T folder
        while dog:
            if os.path.exists(this_path):
                this_files = os.listdir(this_path)
                for this_file in this_files:
                    if this_file[:2] == 'T=':
                        self.path_has_time = True
                        self.current_T += 1
                dog = False
            else:
                time.sleep(0.1)

        if self.path_has_time:  # found T=     [x:\CD00\processing\2019-01-01\XEP_name\B\T=5]
            this_path = os.path.join(this_path, 'T=' + str(self.current_T - 1))
        else:  # not found T=     [x:\CD00\processing\2019-01-01\XEP_name\B]
            self.current_T = 1

        dog = True  # finding S folder
        while dog:
            if os.path.exists(this_path):
                this_files = os.listdir(this_path)
                for this_file in this_files:
                    if this_file[:2] == 'S=':
                        self.is_available = True
                        self.path_has_scene = True
                        self.current_S += 1
                dog = False
            else:
                time.sleep(0.1)

        if self.path_has_scene:  # found S=     [x:\CD00\processing\2019-01-01\XEP_name\B\T=5\S=95]
            this_path = os.path.join(this_path, 'S=' + str(self.current_S - 1))
        else:  # not found S=     [x:\CD00\processing\2019-01-01\XEP_name\B\T=5]
            self.current_S = 1

        dog = True  # finding Z folder
        while dog:
            if os.path.exists(this_path):
                this_files = os.listdir(this_path)
                for this_file in this_files:
                    if this_file[:2] == 'Z=':
                        self.path_has_z_direction = True
                        self.current_Z += 1
                dog = False
            else:
                time.sleep(0.1)

        if self.path_has_z_direction:  # found Z=     [x:\CD00\processing\2019-01-01\XEP_name\B\T=5\S=95\Z=2]
            this_path = os.path.join(this_path, 'Z=' + str(self.current_Z - 1))
        else:  # not found Z=     [x:\CD00\processing\2019-01-01\XEP_name\B\T=5\S=96]
            self.current_Z = 1

        dog = True  # finding C folder
        while dog:
            if os.path.exists(this_path):
                this_files = os.listdir(this_path)
                for this_file in this_files:
                    if this_file[:2] == 'C=':
                        self.path_has_channel = True
                        self.current_C += 1
                dog = False
            else:
                time.sleep(0.1)

        if self.path_has_channel:  # found C=     [x:\CD00\processing\2019-01-01\XEP_name\B\T=5\S=95\Z=2\C=1]
            this_path = os.path.join(this_path, 'C=' + str(self.current_C - 1))
        else:  # found C=     [x:\CD00\processing\2019-01-01\XEP_name\B\T=5\S=95\Z=2]
            self.current_C = 1

        dog = True  # finding M folder
        while dog:
            if os.path.exists(this_path):
                this_files = os.listdir(this_path)
                for this_file in this_files:
                    if this_file[-4:] in self.image_file_type:
                        self.path_has_mosaic_tiles = True
                        self.current_M += 1
                        dog = False
            else:
                time.sleep(0.1)

        if self.current_B > 0:
            self.is_available = True

    def re_init_structs_old(self):
        # the path on disk IS from 0(but in ZEN is from 1)
        self.re_init_vars()
        this_files = os.listdir(self.folder_path)
        for this_file in this_files:
            if this_file[:1] == 'B':
                self.path_has_block = True
                self.current_B += 1
        # find T= :
        if self.path_has_block:
            this_path = os.path.join(self.folder_path, 'B=' + str(self.current_B - 1))
            if (not os.path.exists(this_path)) and self.current_B == 1:
                this_path = os.path.join(self.folder_path, 'B')
            dog = True
            while dog:
                if os.path.exists(this_path):
                    this_files = os.listdir(this_path)
                    for this_file in this_files:
                        if this_file[:2] == 'T=':
                            self.path_has_time = True
                            self.current_T += 1
                    dog = False
                else:
                    time.sleep(0.1)
        # find S= :
        if self.path_has_time:
            this_path = os.path.join(this_path, 'T=' + str(self.current_T - 1))
            dog = True
            while dog:
                if os.path.exists(this_path):
                    this_files = os.listdir(this_path)
                    for this_file in this_files:
                        if this_file[:2] == 'S=':
                            self.is_available = True
                            self.path_has_scene = True
                            self.current_S += 1
                    dog = False
                else:
                    time.sleep(0.1)
        else:  # no T folder
            pass
        if self.path_has_scene:
            this_path = os.path.join(this_path, 'S=' + str(self.current_S - 1))
            dog = True
            while dog:
                if os.path.exists(this_path):
                    this_files = os.listdir(this_path)
                    for this_file in this_files:
                        if this_file[:2] == 'Z=':
                            self.path_has_z_direction = True
                            self.current_Z += 1
                    dog = False
                else:
                    time.sleep(0.1)
        if self.path_has_z_direction:
            this_path = os.path.join(this_path,
                                     'Z=' + str(self.current_Z - 1))  # the str(1) shouold be str(self.current_Z - 1)
            dog = True
            while dog:
                if os.path.exists(this_path):
                    this_files = os.listdir(this_path)
                    for this_file in this_files:
                        if this_file[:2] == 'C=':
                            self.path_has_channel = True
                            self.current_C += 1
                    dog = False
                else:
                    time.sleep(0.1)
        if self.path_has_channel:
            this_path = os.path.join(this_path, 'C=' + str(self.current_C - 1))
            dog = True
            while dog:
                if os.path.exists(this_path):
                    this_files = os.listdir(this_path)
                    for this_file in this_files:
                        if this_file[-4:] in self.image_file_type:
                            self.path_has_mosaic_tiles = True
                            self.current_M += 1
                            dog = False
                else:
                    time.sleep(0.1)
        if self.current_B > 0:
            self.is_available = True

    #      ! notic that: the last this_file is not always the latest comming image, it depends on the OS file order!
    #                    So the code need to be improved!
    #         self.current_image_file = os.path.join(this_path, this_file)

    def getRecentImage(self):
        self.re_init_structs()
        self.current_image_file = self.getSpecificImageName(self.current_B, self.current_T, self.current_S,
                                                            self.current_Z, self.current_C, self.current_M)
        return self.current_image_file

    def getRecentValidImage(self):
        dog = True
        while dog:
            self.re_init_structs()
            if self.current_Z >= 2 - 1 and self.current_C >= 1 - 1:
                self.current_valid_image_file = self.getSpecificImageName(self.current_B - 1, self.current_T - 1,
                                                                          self.current_S - 1, 2 - 1, 1 - 1,
                                                                          self.current_M - 1)
                dog = False
            else:
                time.sleep(0.1)
        return self.current_valid_image_file

    def getSpecificImageName(self, B, T, S, Z, C, M):  # BTSZCM start from 1
        this_path = os.path.join(self.folder_path, 'B=' + str(B - 1), 'T=' + str(T - 1), 'S=' + str(S - 1),
                                 'Z=' + str(Z - 1), 'C=' + str(C - 1))
        if (not os.path.exists(this_path)) and (B == 1):
            this_path = os.path.join(self.folder_path, 'B', 'T=' + str(T - 1), 'S=' + str(S - 1),
                                     'Z=' + str(Z - 1), 'C=' + str(C - 1))
        this_files = os.listdir(this_path)
        for this_file in this_files:
            if int(this_file.split('_M')[-1].split('_')[0]) == M - 1:
                return os.path.join(this_path, this_file)


class ImageData:

    def check_img(self):
        if self.img is None:
            return False
        return True

    def imshow(self, image, title):
        #        plt.close('all')
        if not self.check_img():
            print('!ERROR! no image was input!! [from ImageData]')
            return False
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()
        plt.close('all')

    def getSIFT(self):
        if not self.check_img():
            print('!ERROR! no image was input!! [from ImageData]')
            return None
        sift = cv2.xfeatures2d.SIFT_create()
        try:
            noneuse, self.SIFTdes = sift.detectAndCompute(self.img_gray, None)
            self.siftFeatureVector = np.append(np.std(self.SIFTdes, axis=0, ddof=1), np.mean(self.SIFTdes, axis=0))
        except BaseException as e:
            print('!ERROR! ', e)
            self.siftFeatureVector = np.zeros(256, dtype=np.float64)
        return self.siftFeatureVector

    def getSURF(self):
        if not self.check_img():
            print('!ERROR! no image was input!! [from ImageData]')
            return None
        surf = cv2.xfeatures2d.SURF_create()
        try:
            noneuse, self.SURFdes = surf.detectAndCompute(self.img_gray, None)
            self.surfFeatureVector = np.append(np.std(self.SURFdes, axis=0, ddof=1), np.mean(self.SURFdes, axis=0))
        except BaseException as e:
            print('!ERROR! ', e)
            self.surfFeatureVector = np.zeros(128, dtype=np.float64)
        return self.surfFeatureVector

    def getORB(self):
        if not self.check_img():
            print('!ERROR! no image was input!! [from ImageData]')
            return None
        orb = cv2.ORB_create()
        try:
            noneuse, self.ORBdes = orb.detectAndCompute(self.img_gray, None)
            self.orbFeatureVector = np.append(np.std(self.ORBdes, axis=0, ddof=1), np.mean(self.ORBdes, axis=0))
        except BaseException as e:
            print('!ERROR! ', e)
            self.orbFeatureVector = np.zeros(64, dtype=np.float64)
        return self.orbFeatureVector

    def getDensity(self, resolution=200, c1=1, c2=30, threshold=0.25, show=False):
        # used for mask density using a mean kernel
        # input box: mask box size
        # c1, c2: is the Canny parameters
        # threshold: if the mask box AVG intensity >= threshold, then label it 1
        # output: density is a % number

        if not self.check_img():
            print('!ERROR! no image was input!! [from ImageData]')
            return None

        cut_box = math.floor(self.img_w / resolution)  # width/box
        # self.image_canny = cv2.Canny(np.uint8(self.img_gray), c1, c2)
        try:
            self.image_canny = cv2.Canny(self.img_gray, c1, c2)
        except BaseException as e:
            print('!ERROR! ', e)
            # self.image_canny = np.zeros((self.img_h, self.img_w), dtype=self.image.dtype)
            self.density = -1
            return self.density
        else:
            pass
        finally:
            pass
        self.image_mask = np.zeros([math.floor(self.img_h / cut_box), math.floor(self.img_w / cut_box)], dtype=np.uint8)
        for row in range(math.floor(self.img_h / cut_box)):  # height/box
            for col in range(math.floor(self.img_w / cut_box)):
                each_block = self.image_canny[(row + 1) * cut_box - cut_box + 1 - 1:(row + 1) * cut_box,
                             (col + 1) * cut_box - cut_box + 1 - 1:(col + 1) * cut_box]
                if np.mean(each_block) >= threshold * 255:
                    self.image_mask[row, col] = 1
                else:
                    self.image_mask[row, col] = 0
        self.density = np.mean(self.image_mask)

        if show:
            self.imshow(self.image_canny, 'image_canny')
            self.imshow(self.image_mask, 'image_mask')

        return self.density

    def getCellMask(self, resolution=200, c1=1, c2=30, threshold=0.25):
        # used for get cell clone mask
        # input box: mask box size
        # c1, c2: is the Canny parameters
        # threshold: if the mask box AVG intensity >= threshold, then label it 1
        # output: Mask is a % 0/1 image

        if not self.check_img():
            print('!ERROR! no image was input!! [from ImageData]')
            return None

        cut_box = math.floor(self.img_w / resolution)  # width/box

        try:
            self.image_canny = cv2.Canny(self.img_gray, c1, c2)
        except BaseException as e:
            print('!ERROR! ', e)
            return None
        else:
            pass
        finally:
            pass

        self.image_mask = np.zeros([math.floor(self.img_h / cut_box), math.floor(self.img_w / cut_box)], dtype=np.uint8)
        for row in range(math.floor(self.img_h / cut_box)):  # height/box
            for col in range(math.floor(self.img_w / cut_box)):
                each_block = self.image_canny[(row + 1) * cut_box - cut_box + 1 - 1:(row + 1) * cut_box,
                             (col + 1) * cut_box - cut_box + 1 - 1:(col + 1) * cut_box]
                if np.mean(each_block) >= threshold * 255:
                    self.image_mask[row, col] = 255
                else:
                    self.image_mask[row, col] = 0

        return self.image_mask

    def getPerimeter(self, show=False):
        if not self.check_img():
            print('!ERROR! no image was input!! [from ImageData]')
            return None

        # int(self.img_w / 16)
        self.getDensity(resolution=400, c1=1, c2=30, threshold=0.25, show=False)
        self.image_mask = self.image_mask * 255
        self.image_mask = np.array(self.image_mask, dtype=np.uint8)

        tolerance = 10
        ret, thresh = cv2.threshold(self.image_mask, tolerance, 64, cv2.THRESH_BINARY)
        image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.perimeter = 0
        for this_c in contours:
            c_p_num = len(this_c)
            self.perimeter = self.perimeter + c_p_num
        if show:
            self.imshow(thresh, 'thresh')
            # drawContours_img = cv2.drawContours(np.zeros(self.image_mask.shape, dtype=np.int8), contours, -1,
            #                                     (255, 255, 255), 1)
            # self.imshow(drawContours_img, 'drawContours')
        return self.perimeter

    def getFractal(self):
        if not self.check_img():
            print('!ERROR! no image was input!! [from ImageData]')
            return None

        blur = cv2.GaussianBlur(self.img_gray, (111, 111), 0)
        self.myPGC = np.abs(self.img_gray - blur)
        cv2.imwrite(r'./temp/tmep_myPGC_.png', self.myPGC)

        matlab_engine = matlab.engine.start_matlab()
        self.fractal = matlab_engine.Task_Fractal_S(r'./temp/tmep_myPGC_.png')
        matlab_engine.quit()

        return self.fractal

    def getEntropy(self):
        if not self.check_img():
            print('!ERROR! no image was input!! [from ImageData]')
            return None

        # hist_256list = [0]*256
        # for i in range(self.img_gray.shape[0]): # rows
        #     for j in range(self.img_gray.shape[1]): # cols
        #         hist_256list[ int(self.img_gray[i,j])] += 1

        hist_256list = cv2.calcHist([self.img_gray], [0], None, [256], [0, 255])
        P_hist = hist_256list / (self.img_h * self.img_w)
        self.entropy = -np.nansum(P_hist * np.log2(P_hist))

        return self.entropy

    def getContours(self, show=False):
        if not self.check_img():
            print('!ERROR! no image was input!! [from ImageData]')
            return None

        resolution = int(self.img_w / 4)
        self.getDensity(resolution=resolution, c1=1, c2=30, threshold=0.25, show=False)
        self.image_mask = self.image_mask * 255
        self.image_mask = np.array(self.image_mask, dtype=np.uint8)

        tolerance = 10
        ret, thresh = cv2.threshold(self.image_mask, tolerance, 64, cv2.THRESH_BINARY)
        image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.contours = cv2.drawContours(np.zeros(self.image_mask.shape, dtype=np.uint8), contours, -1,
                                         (255, 255, 255), 1)
        # self.contours = cv2.cvtColor(self.contours, cv2.COLOR_BGR2GRAY)

        if show:
            self.imshow(thresh, 'thresh')
            self.imshow(self.contours, 'drawContours')

        return self.contours

    def getOtherParameters(self):
        pass

    def __init__(self, input_img, show=False):
        self.img = None
        self.img_gray = None
        self.myPGC = None
        self.img_h = None
        self.img_w = None
        self.density = None
        self.perimeter = None
        self.contours = None
        self.fractal = None
        self.entropy = None

        if type(input_img) is str:
            if os.path.exists(input_img):
                self.image_path = input_img
                # self.image = pltimg.imread(self.image_path)
                self.img = cv2.imread(self.image_path)
        elif type(input_img) is np.ndarray:
            self.img = input_img

        if self.img is not None:
            if len(self.img.shape) == 3:
                self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            elif len(self.img.shape) == 2:
                self.img_gray = self.img
            self.img_h = self.img_gray.shape[0]  # y:height
            self.img_w = self.img_gray.shape[1]  # x:width
            if show:
                # self.getDensity()
                self.imshow(self.img, 'image')
                # self.imshow(self.image_canny, 'image_canny')
                # self.imshow(self.image_mask, 'image_mask')


class ImageName:

    def __init__(self, img_path):

        self.existence = False
        self.main_path = None
        self.img_name = None
        self.batch = None
        self.ifPGC = False
        self.SSS = False
        self.SSSS = False
        self.zoom = None
        self.year = None
        self.month = None
        self.day = None
        self.IPSstage = None
        self.stage = None
        self.hour = None
        self.B = None
        self.T = None
        self.S = None
        self.Z = None
        self.C = None
        self.M = None
        self.format = None

        if type(img_path) is str:
            if os.path.exists(img_path):
                self.existence = True

        t_path_list = os.path.split(
            img_path)  # [r'G:\CD46\PROCESSING\MyPGC_img\SSSS_100%\S1', '2020-08-03~CD46_Stage-1_24H~T1.png']
        t1_path_list = os.path.split(t_path_list[0])  # [r'G:\CD46\PROCESSING\MyPGC_img\SSSS_100%', 'S1']
        t2_path_list = os.path.split(t1_path_list[0])  # [r'G:\CD46\PROCESSING\MyPGC_img', 'SSSS_100%']
        t3_path_list = os.path.split(t2_path_list[0])  # [r'G:\CD46\PROCESSING', 'MyPGC_img'] [r'G:\CD46', 'PROCESSING']

        if t3_path_list[1] == 'MyPGC_img':
            self.main_path = t3_path_list[0]
            self.ifPGC = True
        else:
            self.main_path = t2_path_list[0]

        SSSS_folder = t2_path_list[1]  # 'SSSS_100%'

        if SSSS_folder.split('_')[0] == 'SSS':
            self.SSS = True
        elif SSSS_folder.split('_')[0] == 'SSSS':
            self.SSSS = True

        self.zoom = int(SSSS_folder.split('_')[1].split('%')[0]) / 100

        S_index = t1_path_list[1]  # 'S1'
        self.S = int(S_index.split('S')[1])  # 1

        self.img_name = t_path_list[1]  # '2020-08-03~CD46_Stage-1_24H~T1.png'
        name_index = self.img_name.split('.')[0]  # '2020-08-03~CD46_Stage-1_24H~T1'
        self.format = self.img_name.split('.')[1]  # 'png'

        name_list = name_index.split('~')  # '2020-08-03' 'CD46_Stage-1_24H' 'T1'

        self.year = int(name_list[0].split('-')[0])  # '2020'
        self.month = int(name_list[0].split('-')[1])  # '08'
        self.day = int(name_list[0].split('-')[2])  # '03'

        middle_name_list = name_list[1].split('_')  # 'CD46' 'Stage-1' '24H' |  'CD46' 'IPS-2'

        self.batch = middle_name_list[0]  # 'CD46'
        if middle_name_list[-1].find('H') >= 0:
            self.hour = int(middle_name_list[-1].split('H')[0])  # '24H'
        elif middle_name_list[-1].find('h') >= 0:
            self.hour = int(middle_name_list[-1].split('h')[0])  # '24h'

        if len(middle_name_list) > 1:
            name_stage = middle_name_list[1]  # 'Stage-1'
            if name_stage.find('IPS') >= 0 or name_stage.find('Ips') >= 0 or name_stage.find(
                    'ips') >= 0 or name_stage.find('iPS') >= 0:
                self.stage = 0
                if name_stage.find('-') >= 0:
                    self.IPSstage = int(name_stage.split('-')[1])  # '1'
            else:
                if name_stage.find('-') >= 0:
                    self.stage = int(name_stage.split('-')[1])  # '1'

        if len(name_list) == 3:
            if name_index.find('~T') >= 0:
                self.T = int(name_list[2].split('T')[1])
        elif len(name_list) == 4:
            if name_index.find('~T') >= 0:
                self.T = int(name_list[2].split('T')[1])
            if name_index.find('~C') >= 0:
                self.C = int(name_list[3].split('C')[1])
