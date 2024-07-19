import os
import os.path
import cv2
import torch.utils.data as data
from torchvision.transforms import ToTensor
import numpy as np
from Degradation_Model.degradation_model import DegradationModelNP


class TeacherData(data.Dataset):
    def __init__(self, directory, FLAGS):
        super(TeacherData, self).__init__()
        self.FLAGS = FLAGS
        self.directory = directory
        self.data_list = []
        file_dir = self.directory

        for f in os.listdir(file_dir):
            if '_IR' in f:
                self.data_list.append(
                    file_dir + f.split('_')[0] + '_')

    def __getitem__(self, index):
        IR = cv2.imread(str(self.data_list[index]) + 'IR.png', cv2.IMREAD_GRAYSCALE)
        IR = np.float32(IR / 255.)

        VI = cv2.imread(str(self.data_list[index]) + 'VI.png', cv2.IMREAD_COLOR)
        VI_GRAY = cv2.cvtColor(VI, cv2.COLOR_BGR2GRAY)
        VI_GRAY = np.float32(VI_GRAY / 255.)

        TRAIN_DATA = {
            'IR': ToTensor()(IR),
            'VI_GRAY': ToTensor()(VI_GRAY)
        }

        return TRAIN_DATA

    def __len__(self):
        return len(self.data_list)


class StudentData(data.Dataset):
    def __init__(self, directory, FLAGS):
        super(StudentData, self).__init__()
        self.FLAGS = FLAGS
        self.DE_Model = DegradationModelNP(self.FLAGS)
        self.directory = directory
        self.data_list = []
        file_dir = self.directory

        for f in os.listdir(file_dir):
            if '_IR' in f:
                self.data_list.append(
                    file_dir + f.split('_')[0] + '_')

    def __getitem__(self, index):

        image_HQ_IR = cv2.imread(str(self.data_list[index]) + 'IR.png', cv2.IMREAD_GRAYSCALE)
        image_HQ_IR = cv2.cvtColor(image_HQ_IR, cv2.COLOR_BGR2RGB)
        image_HQ_IR = np.float32(image_HQ_IR / 255.)

        image_HQ_VI = cv2.imread(str(self.data_list[index]) + 'VI.png', cv2.IMREAD_COLOR)
        image_HQ_VI = cv2.cvtColor(image_HQ_VI, cv2.COLOR_BGR2RGB)
        image_HQ_VI = np.float32(image_HQ_VI / 255.)

        image_LQ_IR = self.DE_Model.de_ir(image_HQ_IR)
        image_LQ_VI = self.DE_Model.de_vi(image_HQ_VI)
        image_LQ_IR = cv2.cvtColor(image_LQ_IR, cv2.COLOR_RGB2GRAY)

        image_HQ_IR = self.DE_Model.usm_sharper(image_HQ_IR)
        image_HQ_VI = self.DE_Model.usm_sharper(image_HQ_VI)
        image_HQ_IR = cv2.cvtColor(image_HQ_IR, cv2.COLOR_RGB2GRAY)

        TRAIN_DATA = {
            'image_HQ_IR': ToTensor()(image_HQ_IR),
            'image_HQ_VI': ToTensor()(image_HQ_VI),
            'image_LQ_IR': ToTensor()(image_LQ_IR),
            'image_LQ_VI': ToTensor()(image_LQ_VI),
        }

        return TRAIN_DATA

    def __len__(self):
        return len(self.data_list)
