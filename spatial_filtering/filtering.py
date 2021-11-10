import numpy as np


class Filtering:

    def __init__(self, image):
        self.image = image

    def get_gaussian_filter(self):
        """Initialzes and returns a 5X5 Gaussian filter
            Use the formula for a 2D gaussian to get the values for a 5X5 gaussian filter
        """
        size = 5
        sigma_value = 1
        ctr_value = int(size / 2)
        kernel = np.zeros((size, size))
        w_average = 0
        for i in range(size):
            for j in range(size):
                diff = np.sqrt((i - ctr_value) ** 2 + (j - ctr_value) ** 2)
                kernel[i, j] = np.exp(-(diff ** 2) / (2 * sigma_value ** 2))
                w_average = w_average + kernel[i, j]
        return kernel / w_average


    def get_laplacian_filter(self):
        """Initialzes and returns a 3X3 Laplacian filter"""

        m = [[0, 1, 0], [1, -3, 1], [0, 1, 0]]
        return np.array(m)

    def filter(self, filter_name):
        """Perform filtering on the image using the specified filter, and returns a filtered image
            takes as input:
            filter_name: a string, specifying the type of filter to use ["gaussian", laplacian"]
            return type: a 2d numpy array
                """
        img = self.image
        if filter_name == "gaussian":
            kernel = self.get_gaussian_filter()
        else:
            kernel = self.get_laplacian_filter()

        i_r, i_c = img.shape
        kernel_r, kernel_c = kernel.shape
        op = np.zeros(img.shape)
        pad_ht = int((kernel_r - 1) / 2)
        pad_w = int((kernel_c - 1) / 2)
        pad_img = np.zeros((i_r + (2 * pad_ht), i_c + (2 * pad_w)))
        pad_img[pad_ht:pad_img.shape[0] - pad_ht, pad_w:pad_img.shape[1] - pad_w] = img

        for r in range(i_r):
            for c in range(i_c):
                op[r, c] = np.sum(kernel * pad_img[r:r + kernel_r, c:c + kernel_c])

        op[r, c] /= kernel.shape[0] * kernel.shape[1]
        return op



