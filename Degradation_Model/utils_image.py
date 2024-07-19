import time

import numpy as np
import PIL
from PIL import Image
import cv2
import torch
from torchvision.transforms import ToPILImage

'''
# --------------------------------------------
# image format conversion
# --------------------------------------------
# numpy(single) <--->  numpy(unit)
# --------------------------------------------
'''


def uint2single(img):
    return np.float32(img / 255.)


def single2uint(img):
    return np.uint8((img.clip(0, 1) * 255.).round())


def uint162single(img):
    return np.float32(img / 65535.)


def single2uint16(img):
    return np.uint16((img.clip(0, 1) * 65535.).round())


'''
# --------------------------------------------
# image type conversion
# --------------------------------------------
# numpy(single) <--->  PIL(single)
# --------------------------------------------
'''


def single2pil(img):
    img = single2uint(img)

    return Image.fromarray(img)


def pil2single(img):
    img = np.array(img)

    return uint2single(img)


'''
# --------------------------------------------
# Augmentation, flipe and/or rotate
# --------------------------------------------
# The following are enough.
# augmet_img: numpy image of WxHxC or WxH
# --------------------------------------------
'''


def augment_img(img_a, img_b, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img_a, img_b
    elif mode == 1:
        return np.flipud(np.rot90(img_a)), np.flipud(np.rot90(img_b))
    elif mode == 2:
        return np.flipud(img_a), np.flipud(img_b)
    elif mode == 3:
        return np.rot90(img_a, k=3), np.rot90(img_b, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img_a, k=2)), np.flipud(np.rot90(img_b, k=2))
    elif mode == 5:
        return np.rot90(img_a), np.rot90(img_b)
    elif mode == 6:
        return np.rot90(img_a, k=2), np.rot90(img_b, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img_a, k=3)), np.flipud(np.rot90(img_b, k=3))


'''
# --------------------------------------------
# Read Image
# --------------------------------------------
'''


def read_img(path):
    """
    # --------------------------------------------
    # Read image
    # --------------------------------------------

    """

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # BGR or G
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB

    img = np.float32(img / 255.)

    return img


'''
# --------------------------------------------
# PAD Image use CV2
# --------------------------------------------
'''


def pad_img_for_swin_train(img_lq, window_size):

    if img_lq.ndim == 2:

        h_old, w_old = img_lq.shape

    else:

        h_old, w_old, _ = img_lq.shape

    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old

    img_hq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REPLICATE)

    return img_hq

    # window_size = 7
    #
    # img_lq = read_img(r'D:\2D\Data\TrainData\NIR/00001_IR.png')
    # print('GT shape:', img_lq.shape)
    #
    # cv2.imshow('origin',img_lq),cv2.waitKey(0),cv2.destroyAllWindows()
    #
    # h_old, w_old, c = img_lq.shape
    #
    # h_pad = (h_old // window_size + 1) * window_size - h_old
    # w_pad = (w_old // window_size + 1) * window_size - w_old
    #
    # img_hq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad,cv2.BORDER_REPLICATE)
    #
    # print('PAD shape:', img_hq.shape)
    # cv2.imshow('PAD',img_hq),cv2.waitKey(0),cv2.destroyAllWindows()


class ReRestructurePics(object):
    """
    本方案均在图像处理的最后一步来处理, 统一以Pytorch的Tensor来处理，以同时满足GPU和CPU计算。
        可以嵌入到DataLoader和For循环中。

    """

    def __init__(self, args):
        super(ReRestructurePics, self).__init__()
        self.FLAGS = args

    @staticmethod
    def MaskRule(img_index, mask_a, mask_b, rule):
        """
        构建以坐标为基础的Mask规则
        """

        if rule == 'ABA':
            for i, j in img_index:
                if i % 2 != j % 2:
                    mask_a[i][j] = 0
                else:
                    mask_b[i][j] = 0

        elif rule == 'AAA-BBB':
            for i, j in img_index:
                if i % 2 == 0:
                    mask_a[i][j] = 0
                else:
                    mask_b[i][j] = 0

        elif rule == 'AAA|BBB':
            for i, j in img_index:
                if j % 2 == 0:
                    mask_a[i][j] = 0
                else:
                    mask_b[i][j] = 0

        return mask_a, mask_b

    def MakeMask(self, img_a, image_type, rule):
        """
        ‘HW-NP’：cv2读取的HW灰度图。
        ‘HWC-NP’：cv2读取的HWC彩色图像。
        ‘BCHW-Tensor’：Tensor图像，同时兼容灰度图和彩色图。
        """
        mask_a = None
        mask_b = None

        if image_type == 'HW-NP':

            mask_a = torch.ones_like(img_a)
            mask_b = mask_a.clone().detach()
            img_index = torch.nonzero(mask_a)

            mask_a, mask_b = self.MaskRule(img_index, mask_a, mask_b, rule=rule)

        elif image_type == 'HWC-NP':

            mask_a = torch.ones_like(img_a[:, :, 0])
            mask_b = mask_a.clone().detach()

            img_index = torch.nonzero(mask_a)

            mask_a, mask_b = self.MaskRule(img_index, mask_a, mask_b, rule=rule)

            mask_a = mask_a.unsqueeze(2).repeat(1, 1, 3)
            mask_b = mask_b.unsqueeze(2).repeat(1, 1, 3)

        elif image_type == 'BCHW-Tensor':

            mask_a = torch.ones_like(img_a[0, 0, :, :])
            mask_b = mask_a.clone().detach()

            img_index = torch.nonzero(mask_a)

            mask_a, mask_b = self.MaskRule(img_index, mask_a, mask_b, rule=rule)

        return mask_a, mask_b

    @staticmethod
    def ReMaskMultiModalImages(img_a, img_b, mask_a, mask_b, order):
        """
        顺序为图像Mask的顺序，A2B为，A图先Mask，B图后Mask。

        """

        if order == 'A2B':
            re_mask_img = img_a * mask_a + img_b * mask_b

        else:
            re_mask_img = img_a * mask_b + img_b * mask_a

        return re_mask_img

    def Test(self, path_a, path_b, out_path, image_type, rule, order, del_file):
        """
        自带测试。
        image_type为‘HW-NP‘，‘HWC-NP’，‘BCHW-Tensor’三种中的一种。
        rule为’ABA‘，’AAA-BBB‘，’AAA|BBB‘中的一种
        del_file为是否删除output路径下的文件。

        """

        if del_file == 'True':
            from Utils.Utils import del_files
            del_files(out_path)

        img_a = cv2.imread(path_a, cv2.IMREAD_UNCHANGED)
        img_b = cv2.imread(path_b, cv2.IMREAD_UNCHANGED)

        if image_type == 'HW-NP':
            img_a = torch.from_numpy(img_a)
            img_b = torch.from_numpy(img_b)

        elif image_type == 'HWC-NP':
            img_a = cv2.cvtColor(img_a, cv2.COLOR_GRAY2RGB)
            img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2RGB)

            img_a = torch.from_numpy(img_a)
            img_b = torch.from_numpy(img_b)

        elif image_type == 'BCHW-Tensor':
            img_a = cv2.cvtColor(img_a, cv2.COLOR_GRAY2RGB)
            img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2RGB)
            img_a = torch.from_numpy(img_a).permute(2, 0, 1).unsqueeze(0)
            img_b = torch.from_numpy(img_b).permute(2, 0, 1).unsqueeze(0)

        mask_a, mask_b = self.MakeMask(img_a=img_a, image_type=image_type, rule=rule)
        re_mask_img = self.ReMaskMultiModalImages(img_a, img_b, mask_a, mask_b, order)

        if image_type == 'HW-NP':
            re_mask_img = re_mask_img.numpy()

        elif image_type == 'HWC-NP':
            re_mask_img = re_mask_img.numpy()

        elif image_type == 'BCHW-Tensor':
            re_mask_img = torch.squeeze(re_mask_img, 0).permute(1, 2, 0).numpy()

        cv2.imwrite(
            out_path + order + '.png',
            re_mask_img,
            [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
        )


# ReRestructurePics(args=None).Test(
#     path_a=r'D:\2D\Data\TrainData\NIR/05729_IR.png',
#     path_b=r'D:\2D\Data\TrainData\NIR/05729_VI.png',
#     out_path=r'D:\111/',
#     image_type='BCHW-Tensor',
#     rule='AAA-BBB',
#     order='A2B2',
#     del_file='True'
# )


'''
# --------------------------------------------
# ReRestructurePics coded as Pytorch Model
# --------------------------------------------
'''

class PicMix(torch.nn.Module):
    def __init__(self):
        super(PicMix, self).__init__()

    @staticmethod
    def MaskRule(img_index, mask, rule):
        """
        构建以坐标为基础的Mask规则
        """

        mask_a_copy = mask.clone().detach()
        mask_b_copy = mask.clone().detach()

        if rule == 'ABA':
            for i, j in img_index:
                if i % 2 != j % 2:
                    mask_a_copy[i][j] = 0
                else:
                    mask_b_copy[i][j] = 0

        elif rule == 'AAA-BBB':
            for i, j in img_index:
                if i % 2 == 0:
                    mask_a_copy[i][j] = 0
                else:
                    mask_b_copy[i][j] = 0

        elif rule == 'AAA|BBB':
            for i, j in img_index:
                if j % 2 == 0:
                    mask_a_copy[i][j] = 0
                else:
                    mask_b_copy[i][j] = 0

        return mask_a_copy, mask_b_copy

    def forward(self, img_a, img_b, rule_list):

        img_a = torch.ones_like(img_a)
        img_b = torch.zeros_like(img_b)

        b, c, h, w = img_a.shape

        mask = torch.ones_like(img_a[0, 0, :, :])
        img_index = torch.nonzero(mask)

        mask_a = torch.ones_like(img_a[0, 0, :, :]).unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1)
        mask_b = torch.zeros_like(img_a[0, 0, :, :]).unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1)

        for rule_type in rule_list:
            mask_a_copy, mask_b_copy = self.MaskRule(img_index, mask, rule_type)

            mask_a_copy = mask_a_copy.unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1)
            mask_b_copy = mask_b_copy.unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1)

            mask_a = torch.cat([mask_a, mask_a_copy], dim=1)
            mask_b = torch.cat([mask_b, mask_b_copy], dim=1)

        # img_a = img_a.repeat(1, len(rule_list) + 1, 1, 1)
        # img_b = img_b.repeat(1, len(rule_list) + 1, 1, 1)

        # img_a_out = img_a * mask_a + img_b * mask_b
        # img_b_out = img_b * mask_a + img_a * mask_b

        return mask_a, mask_b


# img_a = cv2.imread(r'D:\2D\Data\TestData\TNO/00003_VISR.png', cv2.IMREAD_UNCHANGED)
# # img_b = cv2.imread(r'D:\2D\Data\TestData\TNO/00003_VISR.png'', cv2.IMREAD_UNCHANGED)
#
# # img_a = cv2.cvtColor(img_a, cv2.COLOR_GRAY2RGB)
# # img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2RGB)
#
# img_a = torch.from_numpy(img_a).unsqueeze(0).unsqueeze(0).repeat(4, 1, 1, 1)
# # img_b = torch.from_numpy(img_b).permute(2, 0, 1).unsqueeze(0).repeat(4, 1, 1, 1)
#
# rule_list = ['ABA', 'AAA-BBB', 'AAA|BBB']
#
# a = time.time()
#
# mask_a, mask_b = PicMix()(img_a, img_a, rule_list)
#
# print(mask_a.shape, time.time() - a)
#
# b1, c, h, w = a.shape
#
# for b_index in range(b1):
#     print(a[0, 0, :, :].shape)
#
#     cv2.imwrite(r'D:\111/' + str(b_index) + '-1a.png',
#                 a[0, 0:1, :, :].permute(1, 2, 0).numpy(),
#                 [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
#     cv2.imwrite(r'D:\111/' + str(b_index) + '-2a.png',
#                 a[0, 1:2, :, :].permute(1, 2, 0).numpy(),
#                 [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
#     cv2.imwrite(r'D:\111/' + str(b_index) + '-3a.png',
#                 a[0, 2:3, :, :].permute(1, 2, 0).numpy(),
#                 [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
#     cv2.imwrite(r'D:\111/' + str(b_index) + '-4a.png',
#                 a[0, 3:4, :, :].permute(1, 2, 0).numpy(),
#                 [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
#
#     # cv2.imwrite(r'D:\111/' + str(b_index) + '-1b.png',
#     #             b[0, :3 :, :].permute(1, 2, 0).numpy(),
#     #             [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
#     # cv2.imwrite(r'D:\111/' + str(b_index) + '-2b.png',
#     #             b[0, 3:6, :, :].permute(1, 2, 0).numpy(),
#     #             [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
#     # cv2.imwrite(r'D:\111/' + str(b_index) + '-3b.png',
#     #             b[0, 6:9, :, :].permute(1, 2, 0).numpy(),
#     #             [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
#     # cv2.imwrite(r'D:\111/' + str(b_index) + '-4b.png',
#     #             b[0, 9:12, :, :].permute(1, 2, 0).numpy(),
#     #             [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

# from PIL import Image
# # img = cv2.imread(r'D:\2D\Data\TestData\TNO/00008_VISR.png')
# img = Image.open(r'D:\2D\Data\TestData\TNO/00008_VISR.png')
#
# print(img.size)