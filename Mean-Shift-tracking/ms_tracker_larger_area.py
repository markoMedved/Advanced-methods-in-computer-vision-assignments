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
        # Pad the kernel 
        tmp = int(round(self.kernel_size/self.parameters.template_to_kernel))
        self.kernel = np.pad(self.kernel, pad_width=((tmp, tmp),
                                                     (tmp, tmp)))

        self.position = (round(region[0] + self.kernel_size / 2.0), round(region[1] + self.kernel_size / 2.0))
        # Extract the template

        template, _ = get_patch(image, self.position, (self.kernel_size * self.parameters.template_to_kernel +1,
                                                        self.kernel_size * self.parameters.template_to_kernel + 1))


        self.hist_template = extract_histogram(template, self.parameters.nbins, self.kernel)
        self.hist_template = self.hist_template / np.sum(self.hist_template)

    def track(self, image):
        kernel_half = self.kernel.shape[0]// 2
        x_diff_mtx = np.arange(-kernel_half, kernel_half+1)
        x_diff_mtx = np.tile(x_diff_mtx, (self.kernel.shape[0], 1))
        y_diff_mtx = x_diff_mtx.T

        x,y = self.position

        # position of x,y inside backproj image
        x_backproj = kernel_half
        y_backprj = kernel_half
        
        for i in range(self.parameters.n_iter):
            # Get the current patch
            current_patch, _ =  get_patch(image, self.position, (self.kernel_size * self.parameters.template_to_kernel +1,
                                                        self.kernel_size * self.parameters.template_to_kernel + 1))
            current_hist = extract_histogram(current_patch, self.parameters.nbins, self.kernel)
            current_hist = current_hist / np.sum(current_hist)

            #Compute the weights
            weights = np.sqrt(self.hist_template / (current_hist + self.parameters.eps* np.ones_like(current_hist)))
            weights = weights / np.sum(weights)
            # Backproject the image
            image_backprojected = backproject_histogram(current_patch, weights, self.parameters.nbins)
            if np.sum(image_backprojected) < self.parameters.eps:
                break
            
            x_i_mtx = (x_backproj * np.ones_like(x_diff_mtx) + x_diff_mtx)
            y_i_mtx = (y_backprj * np.ones_like(x_diff_mtx) + y_diff_mtx)

            # x_new = np.sum(x_i_mtx * image_backprojected) / (np.sum(image_backprojected) + self.parameters.eps)
            # y_new = np.sum(y_i_mtx * image_backprojected) / (np.sum(image_backprojected) + self.parameters.eps)
            x_new_backproj = np.sum(x_i_mtx * image_backprojected) / (np.sum(image_backprojected) + self.parameters.eps)
            y_new_backproj = np.sum(y_i_mtx * image_backprojected) / (np.sum(image_backprojected) + self.parameters.eps)

            # break if the trend is downward
            # x_new_backproj = kernel_half + round(x_new - x)
            # y_new_backproj = kernel_half + round(y_new -y)
            if image_backprojected[x_backproj,y_backprj] > image_backprojected[int(round(y_new_backproj)), int(round(x_new_backproj))]:
                print(f"Lower probs at {i}")
                break
    
            if (np.abs(x_backproj - x_new_backproj) < 0.5) and (np.abs(y - y_new_backproj) < 0.5):
                print(f"Less than 1 pixel step at {i}")
                break

            if (x_backproj - x_new_backproj)**2 + (y_backprj - y_new_backproj)**2 < self.parameters.minstep:
                print(f"less than {self.parameters.minstep} step at {i}")
                break

            # x = round(x_new)
            # y = round(y_new)

            x = x  + round(x_new_backproj - x_backproj)
            y = y + round(y_new_backproj - y_new_backproj)

            self.position = (x,y)
        #print(i)
        # Update the template histogram
        self.hist_template = (1-self.parameters.alpha) * self.hist_template + self.parameters.alpha * current_hist
        
        # TODO You can squeue the box here to be the same shape as at the start
        return [self.position[0] - self.kernel_size//2,self.position[1] - self.kernel_size//2,self.kernel_size, self.kernel_size]

class MSParams():
    def __init__(self, nbins = 16,sigma = 2, eps = 1e-7, alpha = 0.01, n_iter = 20, minstep=1, template_to_kernel = 1.5):
        self.nbins = nbins
        self.eps = eps
        self.alpha = alpha
        self.sigma = sigma
        self.n_iter = n_iter    
        self.minstep = minstep
        self.template_to_kernel = template_to_kernel