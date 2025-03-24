import numpy as np
import cv2
from ex2_utils import generate_responses_1, extract_histogram, backproject_histogram, create_epanechnik_kernel, get_patch, Tracker



class MeanShiftTracker(Tracker):

    def initialize(self, image, region):
        """Initialized the tracker with the provided bounding box for the first frame"""
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
            # Calculate the weights, set an epsilon for numerical stability
            weights_hist = np.sqrt(self.template_hist / (current_hist + 1e-7 * np.ones_like(current_hist)))
            weights = backproject_histogram(patch_current, weights_hist, self.parameters.nbins)
            # Mean shift step
            # Get the x and y position matrices
            x_i_mtx = x * np.ones_like(x_diff_mtx) + x_diff_mtx
            y_i_mtx = y * np.ones_like(y_diff_mtx) + y_diff_mtx
            # Get the new positions
            x_new = np.sum(x_i_mtx * weights) / (np.sum(weights) + 1e-7)
            y_new = np.sum(y_i_mtx * weights) / (np.sum(weights) + 1e-7)
            # Break if the difference is less than one pixel
            if (x_new - x)**2 + (y_new - y)**2 < 0.5:
                break
            # If negative break
            if x_new < 0 or y_new < 0 or x_new > image.shape[1] or y_new > image.shape[0]:
                break
            # Update the position
            x = x_new
            y = y_new
            self.position = (x,y)

        # Update the template histogram
        self.template_hist = (1- self.parameters.alpha) * self.template_hist + self.parameters.alpha * current_hist
        
        # Return the bounding box for the current frame
        return [x - self.patch_shape[0]//2, y - self.patch_shape[1]//2, *self.patch_shape]


class MSParams():
    def __init__(self, nbins = 16, sigma = 0.5, n_iters = 10, alpha = 0.0):
        self.nbins = 16
        self.sigma = sigma
        self.n_iters = n_iters
        self.alpha = alpha