# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv

import numpy as np


class Filtering:

    def __init__(self, image):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        """
        self.image = image
        self.mask = self.get_mask

    def get_mask(self, shape):
        """Computes a user-defined mask
        takes as input:
        shape: the shape of the mask to be generated
        rtype: a 2d numpy array with size of shape
        """

        mask_img = np.ones(shape)
        mask_img[225:244, 275:290] = 0
        mask_img[275:295, 218:236] = 0

        return mask_img

    def post_process_image(self, image):
        """Post processing to display DFTs and IDFTs
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        You can perform post processing as needed. For example,
        1. You can perfrom log compression
        2. You can perfrom a full contrast stretch (fsimage)
        3. You can take negative (255 - fsimage)
        4. etc.
        """

        diff = np.max(image) - np.min(image)

        r, c = np.shape(image)
        img_processed = np.zeros((r, c), dtype=int)
        for i in range(r):
            for j in range(c):
                img_processed[i, j] = (255 / diff) * (image[i, j] - np.min(image))

        return np.uint8(img_processed)

    def filter(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        1. Compute the fft of the image
        2. shift the fft to center the low frequencies
        3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape)
        4. filter the image frequency based on the mask (Convolution theorem)
        5. compute the inverse shift
        6. compute the inverse fourier transform
        7. compute the magnitude
        8. You will need to do post processing on the magnitude and depending on the algorithm (use post_process_image to write this code)
        Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8
        """

        img = self.image
        frwd_ft = np.fft.fft2(img)

        shft_fft = np.fft.fftshift(frwd_ft)
        mag_dft = np.log(np.abs(shft_fft))
        dft = self.post_process_image(mag_dft)
        mask_img = self.get_mask(img.shape)

        fltr_image = np.multiply(mask_img,shft_fft)
        mag_filter_dft = np.log(np.abs(fltr_image) + 1)
        fltr_dft = self.post_process_image(mag_filter_dft)

        shft_inv_fft = np.fft.ifftshift(fltr_image)
        inv_fft = np.fft.ifft2(shft_inv_fft)
        magnitude = np.abs(inv_fft)
        fltr_image = self.post_process_image(magnitude)

        return [np.uint8(fltr_image), np.uint8(dft), np.uint8(fltr_dft)]