import os
import os.path
import cv2
import numpy as np
import torch.utils.data as data
from torchvision.transforms import ToTensor
from Degradation_Model.utils_image import pad_img_for_swin_train


class TestMotherData(data.Dataset):
    def __init__(self, directory, DataSetName):
        super(TestMotherData, self).__init__()

        self.directory = directory
        self.data_list = []
        self.DataSetName = DataSetName
        file_dir = self.directory

        for f in os.listdir(file_dir):
            if '_IR' in f:
                self.data_list.append(os.path.join(file_dir, f.split('_')[0]))

        if self.DataSetName == 'LLVIP_RGB':
            self.end_IR = '.jpg'
            self.end_VI = '.jpg'

        elif self.DataSetName == 'M3FD_RGB':
            self.end_IR = '.png'
            self.end_VI = '.png'

        elif self.DataSetName == 'Road_RGB':
            self.end_IR = '.jpg'
            self.end_VI = '.jpg'

        elif self.DataSetName == 'TNO':
            self.end_IR = '.png'
            self.end_VI = '.png'

    def __getitem__(self, index):
        IR = cv2.imread(self.data_list[index] + '_IR' + self.end_IR, cv2.IMREAD_GRAYSCALE)
        IR = np.float32(IR / 255.)

        if self.DataSetName == 'TNO':
            VI = cv2.imread(self.data_list[index] + '_VI' + self.end_VI, cv2.IMREAD_GRAYSCALE)
            VI = cv2.cvtColor(VI, cv2.COLOR_GRAY2RGB)
            VI = np.float32(VI / 255.)

        else:
            VI = cv2.imread(self.data_list[index] + '_VI' + self.end_VI, cv2.IMREAD_UNCHANGED)
            VI = cv2.cvtColor(VI, cv2.COLOR_BGR2RGB)
            VI = np.float32(VI / 255.)

        h_old, w_old, _ = VI.shape
        IR = pad_img_for_swin_train(IR, 8)
        VI = pad_img_for_swin_train(VI, 8)

        TEST_DATA = {
            'IR': ToTensor()(IR),
            'VI': ToTensor()(VI),
            'Image_name': os.path.split(self.data_list[index])[1],
            'H': h_old,
            'W': w_old
        }

        return TEST_DATA

    def __len__(self):
        return len(self.data_list)
