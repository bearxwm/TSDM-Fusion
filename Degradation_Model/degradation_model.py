import random
import numpy as np
from Degradation_Model.degradations import add_sharpening, add_Gaussian_noise, add_blur, add_JPEG_noise
from Degradation_Model.degradations import adjust_contrast, adjust_brightness


class DegradationModelNP(object):

    def __init__(self, args):
        super(DegradationModelNP, self).__init__()
        self.FLAGS = args

    def usm_sharper(self, img):
        """
        # --------------------------------------------
        # usm sharper Test
        # radius & sigma need to fine tune
        # --------------------------------------------
        """

        usm_out = add_sharpening(
            img=img,
            weight=0.5,
            threshold=10,
            radius=self.FLAGS['Degradation']['Shaper']['radius']
        )

        return usm_out

    # -------- Sub Degradation Models --------
    # Gaussian Noise Degradation Model
    def gaussian_noise(self, img):
        gaussian_noise_out = add_Gaussian_noise(
            img,
            noise_level1=self.FLAGS['Degradation']['noise']['gaussian_noise_level1'],
            noise_level2=self.FLAGS['Degradation']['noise']['gaussian_noise_level2']
        )
        return gaussian_noise_out

    # Blur Degradation Model
    def blur(self, img):
        blur_out = add_blur(img, sf=1)
        return blur_out

    # JPEG Noise Degradation Model
    def JPEG_noise(self, img, quality_factor):
        jpeg_out = add_JPEG_noise(img, quality_factor)
        return jpeg_out

    # Contrast Degradation Model
    def contrast(self, img, contrast_rate):
        contrast_out = adjust_contrast(img, contrast_rate)
        return contrast_out

    # Brightness Degradation Model
    def brightness(self, img, brightness_rate, brightness_prob):
        brightness_out = adjust_brightness(img, brightness_rate, brightness_prob)
        return brightness_out

    # ------- IR&VIS Degradation Models -------
    # Degradation Model of Infrared Images
    def de_ir(self, img):

        img = self.contrast(img, random.uniform(
            self.FLAGS['Degradation']['contrast_rate_ir'][0],
            self.FLAGS['Degradation']['contrast_rate_ir'][1]
        ))

        shuffle_order = random.sample(range(2), 2)

        for i in shuffle_order:

            if i == 0:

                img = self.gaussian_noise(img)

            elif i == 1:

                img = self.blur(img)

        img = self.JPEG_noise(img, np.random.randint(70, 90))

        if random.random() < 0.5:
            img = self.JPEG_noise(img, np.random.randint(30, 70))

        return img

    # Degradation Model of Visible Images
    def de_vi(self, img):

        img = self.brightness(img,
                              random.uniform(
                                  self.FLAGS['Degradation']['brightness_rate_vi'][0],
                                  self.FLAGS['Degradation']['brightness_rate_vi'][1]),
                              self.FLAGS['Degradation']['brightness_prob'])

        img = self.contrast(img, random.uniform(
            self.FLAGS['Degradation']['contrast_rate_vi'][0],
            self.FLAGS['Degradation']['contrast_rate_vi'][1]
        ))

        shuffle_order = random.sample(range(2), 2)

        for i in shuffle_order:

            if i == 0:

                img = self.gaussian_noise(img)

            elif i == 1:

                img = self.blur(img)

        img = self.JPEG_noise(img, np.random.randint(70, 90))

        if random.random() < 0.5:
            img = self.JPEG_noise(img, np.random.randint(30, 70))

        return img