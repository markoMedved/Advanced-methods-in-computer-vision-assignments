import numpy as np
import cv2
from ex2_utils import generate_responses_1, extract_histogram, backproject_histogram, create_epanechnik_kernel, get_patch, Tracker
import matplotlib.pyplot as plt
import seaborn as sns

class MeanShiftTracker(Tracker):
    def initialize(self, image, region):

        self.kernel_size = int(np.mean([region[2], region[3]]))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.kernel = create_epanechnik_kernel(self.kernel_size , self.kernel_size, self.kernel_size//self.parameters.sigma)

        self.position = (round(region[0] + self.kernel_size / 2.0), round(region[1] + self.kernel_size / 2.0))
        # Extract the template
        template, _ = get_patch(image, self.position, self.kernel.shape)

        self.hist_template = extract_histogram(template, self.parameters.nbins, self.kernel)
        self.hist_template = self.hist_template / np.sum(self.hist_template)

    def track(self, image):
        kernel_half = self.kernel_size // 2
        x_diff_mtx = np.arange(-kernel_half, kernel_half+1)
        x_diff_mtx = np.tile(x_diff_mtx, (self.kernel_size, 1))
        y_diff_mtx = x_diff_mtx.T

        x,y = self.position
        
        for i in range(self.parameters.n_iter):
            # Get the current patch
            current_patch, _ =  get_patch(image, self.position, self.kernel.shape)
            current_hist = extract_histogram(current_patch, self.parameters.nbins, self.kernel)
            current_hist = current_hist / np.sum(current_hist)

            #Compute the weights
            weights = np.sqrt(self.hist_template / (current_hist + self.parameters.eps* np.ones_like(current_hist)))
            weights = weights / np.sum(weights)
            # Backproject the image
            image_backprojected = backproject_histogram(current_patch, weights, self.parameters.nbins)
            if np.sum(image_backprojected) < self.parameters.eps:
                break
            
            x_i_mtx = (x * np.ones_like(x_diff_mtx) + x_diff_mtx)
            y_i_mtx = (y * np.ones_like(x_diff_mtx) + y_diff_mtx)

            x_new = np.sum(x_i_mtx * image_backprojected) / (np.sum(image_backprojected) + self.parameters.eps)
            y_new = np.sum(y_i_mtx * image_backprojected) / (np.sum(image_backprojected) + self.parameters.eps)

            if (np.abs(x - x_new) < 0.5) and (np.abs(y - y_new) < 0.5):
                x = round(x_new)
                y = round(y_new)
                break

            x = round(x_new)
            y = round(y_new)

            # x_in_patch, y_in_patch = mean_shift(image_backprojected, self.kernel_size//2, self.kernel_size//2 , self.kernel_size)
            # self.position = (x_in_patch - self.kernel_size//2 + self.position[0], y_in_patch- self.kernel_size//2 + self.position[1])

        self.position = (x,y)

        template, _ = get_patch(image, self.position, self.kernel.shape)
        curr_hist = extract_histogram(template, self.parameters.nbins, self.kernel)

        
        self.hist_template = (1-self.parameters.alpha) * self.hist_template + self.parameters.alpha * curr_hist
        
        return [self.position[0] - self.kernel_size//2,self.position[1] - self.kernel_size//2,self.kernel_size, self.kernel_size]

class MSParams():
    def __init__(self, nbins = 16,sigma = 2, eps = 1e-7, alpha = 0.0, n_iter = 3):
        self.nbins = nbins
        self.eps = eps
        self.alpha = alpha
        self.sigma = sigma
        self.n_iter = n_iter