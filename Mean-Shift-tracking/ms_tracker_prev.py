import numpy as np
import cv2
from ex2_utils import generate_responses_1, extract_histogram, backproject_histogram, create_epanechnik_kernel, get_patch, Tracker
import matplotlib.pyplot as plt
import seaborn as sns


# Mean shift algorithm
def mean_shift(pdf, x_start, y_start, kernel_size,termination_criterium = "step_less_than_1", n_iter = 20 ,return_all_x_y = False, seed = 42):
    """Calculates the mean shift algorithm on the provided distribution"""
    # return np.unravel_index(np.argmax(pdf), pdf.shape)[::-1]
    # set the seed for replicability
    np.random.seed(seed)
    # Kernel has to be odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    # Try the Epanechnikov kernel first (basically just multiply)
    x = x_start
    y = y_start
    # Save all points for visualization
    all_x = []
    all_y = []
    # Epsilon to not divide by 0
    eps = 1e-10
    kernel_half = kernel_size // 2
    # Pad the pdf and update the x,y
    #pdf_len = pdf.shape[0] # Save the length for later
    pdf = np.pad(pdf, pad_width=kernel_size)
    x += kernel_size
    y += kernel_size

    # Get the matrices of differences
    x_diff_mtx = np.arange(-kernel_half, kernel_half+1)
    x_diff_mtx = np.tile(x_diff_mtx, (kernel_size, 1))
    y_diff_mtx = x_diff_mtx.T


    for i in np.arange(n_iter):
        # Get the weights matrix
        weights_mtx = pdf[y-kernel_half:y+kernel_half+1,x-kernel_half:x+kernel_half+1]
        # If the sum of weigths is currently zero randomly pick another position
        if np.sum(weights_mtx) < eps:
            
            return x,y
            # x = round(np.random.random() *pdf_len) 
            # y = round(np.random.random() * pdf_len) 

            # continue
        # Change them so that it will be a pdf on the current patch (sum of weights to 1)
        weights_mtx = weights_mtx / (np.sum(weights_mtx))

        # Get the positions matrices for x and y
        x_i_mtx = (x * np.ones_like(x_diff_mtx) + x_diff_mtx)
        y_i_mtx = (y * np.ones_like(x_diff_mtx) + y_diff_mtx)


        x_new = np.sum(x_i_mtx * weights_mtx) / (np.sum(weights_mtx) + eps)
        y_new = np.sum(y_i_mtx * weights_mtx) / (np.sum(weights_mtx) + eps)

        all_x.append(round(x_new) - kernel_size)
        all_y.append(round(y_new) - kernel_size)

        # Check the termination criterium
        if termination_criterium == "step_less_than_1":
            if (np.abs(x - x_new) < 0.5) and (np.abs(y - y_new) < 0.5):

                x = round(x_new)
                y = round(y_new)
                break
 

        x = round(x_new)
        y = round(y_new)

    return x - kernel_size,y - kernel_size


class MeanShiftTracker(Tracker):
    def initialize(self, image, region):

        self.kernel_size = int(np.mean([region[2], region[3]]))
        self.kernel = create_epanechnik_kernel(self.kernel_size , self.kernel_size, self.kernel_size//self.parameters.sigma)

        self.position = (round(region[0] + self.kernel_size / 2.0), round(region[1] + self.kernel_size / 2.0))
        # Extract the template
        template, _ = get_patch(image, self.position, self.kernel.shape)

        self.hist_template = extract_histogram(template, self.parameters.nbins, self.kernel)
        self.hist_template = self.hist_template / np.sum(self.hist_template)

    def track(self, image):
        
        for i in range(5):
            # Get the current patch
            current_patch, _ =  get_patch(image, self.position, self.kernel.shape)
            current_hist = extract_histogram(current_patch, self.parameters.nbins, self.kernel)
            current_hist = current_hist / np.sum(current_hist)

            #Compute the weights
            weights = np.sqrt(self.hist_template / (current_hist + self.parameters.eps* np.ones_like(current_hist)))
            weights = weights / np.sum(weights)
            # Backproject the image
            image_backprojected = backproject_histogram(current_patch, weights, self.parameters.nbins)

            x_in_patch, y_in_patch = mean_shift(image_backprojected, self.kernel_size//2, self.kernel_size//2 , self.kernel_size)
            self.position = (x_in_patch - self.kernel_size//2 + self.position[0], y_in_patch- self.kernel_size//2 + self.position[1])

            


            template, _ = get_patch(image, self.position, self.kernel.shape)
            curr_hist = extract_histogram(template, self.parameters.nbins, self.kernel)

           
            self.hist_template = (1-self.parameters.alpha) * self.hist_template + self.parameters.alpha * curr_hist
        
        return [self.position[0] - self.kernel_size//2,self.position[1] - self.kernel_size//2,self.kernel_size, self.kernel_size]

class MSParams():
    def __init__(self, nbins = 16,sigma = 2, eps = 1e-7, alpha = 0.0):
        self.nbins = nbins
        self.eps = eps
        self.alpha = alpha
        self.sigma = sigma