import numpy as np
import cv2
from ex2_utils import generate_responses_1, extract_histogram, backproject_histogram, create_epanechnik_kernel, get_patch, Tracker
import matplotlib.pyplot as plt
from time import sleep

class MeanShiftTracker(Tracker):
    def get_background_histogram(self,image):
        background,_ = get_patch(image, self.position, self.bg_shape)
        # Set the pixels at the template to 0
        background[(self.bg_shape[1]- self.patch_shape[1])//2:(self.bg_shape[1] + self.patch_shape[1])//2, 
                            (self.bg_shape[0]-  self.patch_shape[0])//2:(self.bg_shape[0] +  self.patch_shape[0])//2] = 0
        # Get the bg histogram
        hist_bg = extract_histogram(background, self.parameters.nbins, np.ones_like(background[:,:,0]))
        hist_bg = hist_bg / np.sum(hist_bg)
        # Get the weights
        minimal_nonzero = min(hist_bg[hist_bg >0])
        hist_bg[hist_bg == 0] = minimal_nonzero
        hist_bg = minimal_nonzero * np.ones_like(hist_bg) /hist_bg
        #print(np.unique(hist_bg, return_counts=True))
        return  hist_bg


    def initialize(self, image, region):
        """Initialized the tracker with the provided bounding box for the first frame"""
        # Convert image to the target color space
        if self.parameters.color_space in ["cv2.COLOR_BGR2RGB", "cv2.COLOR_BGR2LAB", 
                                           "cv2.COLOR_BGR2HSV", "cv2.COLOR_BGR2HSVYCrCb"]:
            image = cv2.cvtColor(image, eval(self.parameters.color_space))
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
        # Get the bounding box of the initialized frame, turn stuff to ints
        left, top, width, height = [int(round(el)) for el in region]
        # Initialize the first position 
        self.position = (left + width//2, top + height/2)
        # Get the patch which will be used as the template to follow, add the stuff so that it is the correct size
        self.patch_shape = (width + (1-width%2), height + (1-height%2))
        template,template_mask = get_patch(image, self.position, self.patch_shape)
        # Create the kernel 
        self.kernel = create_epanechnik_kernel(width, height, self.parameters.sigma)
        current_kernel = self.kernel * template_mask
        # Extract the template histogram
        self.template_hist = extract_histogram(template, self.parameters.nbins, current_kernel)
        self.template_hist = self.template_hist / np.sum(self.template_hist)
        if self.parameters.bg_ratio != -1:
            # Get the background 
            self.bg_shape = (int(self.patch_shape[0] * self.parameters.bg_ratio), 
                             int(self.patch_shape[1] * self.parameters.bg_ratio))
            self.template_hist_bg = self.get_background_histogram(image)
            # Adjust the template histogram
            self.template_hist = self.template_hist * self.template_hist_bg


    def track(self, image):
        """"Tracks the object on the current frame"""
        x,y = self.position
        # Calculate the difference matrices around the center, for mean shift
        x_diff_mtx = np.arange(-(self.kernel.shape[1]//2), self.kernel.shape[1]//2 + 1)
        x_diff_mtx = np.tile(x_diff_mtx, (self.kernel.shape[0], 1))
        y_diff_mtx = np.arange(-(self.kernel.shape[0]//2), self.kernel.shape[0]//2+1)
        y_diff_mtx = np.tile(y_diff_mtx, (self.kernel.shape[1], 1)).T

        for i in range(self.parameters.n_iters):
            # Get the patch
            patch_current, mask_current = get_patch(image, self.position, self.patch_shape)
            current_kernel = self.kernel * mask_current
            # Get the histogram
            current_hist = extract_histogram(patch_current, self.parameters.nbins, current_kernel)
            current_hist = current_hist / np.sum(current_hist)

            # if self.parameters.bg_ratio != -1:
            #     # Adjust the template histogram
            #     bg_hist_curr =  self.get_background_histogram(image)
            #     current_hist = current_hist * bg_hist_curr 

            # Calculate the weights, set an epsilon for numerical stability
            weights_hist = np.sqrt(self.template_hist / (current_hist + 1e-7 * np.ones_like(current_hist)))
            weights = backproject_histogram(patch_current, weights_hist, self.parameters.nbins)
            weights = weights * self.kernel
            # Mean shift step
            # Get the new positions
            x_new = np.sum(x_diff_mtx * weights) / (np.sum(weights) + 1e-7)
            y_new = np.sum(y_diff_mtx * weights) / (np.sum(weights) + 1e-7)
            # Break if the difference is less than one pixel
            if (x_new)**2 + (y_new)**2 < 0.5:
                break
            # If negative break
            if x +  x_new < 0 or y + y_new < 0 or x + x_new > image.shape[1] or y + y_new > image.shape[0]:
                break
            # Update the position
            x = x + x_new
            y = y + y_new
            self.position = (x,y)

        # Update the template histogram
        self.template_hist = (1- self.parameters.alpha) * self.template_hist + self.parameters.alpha * current_hist
        
        # Return the bounding box for the current frame
        return [x - self.patch_shape[0]//2, y - self.patch_shape[1]//2, self.patch_shape[0], self.patch_shape[1]]


class MSParams():
    def __init__(self, nbins = 16, sigma = 2, n_iters = 10, 
                 alpha = 0.0,background_size_ratio = 3, color_space = ""):
        self.nbins = nbins
        self.sigma = sigma
        self.n_iters = n_iters
        self.alpha = alpha
        self.bg_ratio = background_size_ratio
        self.color_space = f"cv2.COLOR_BGR2{color_space}"