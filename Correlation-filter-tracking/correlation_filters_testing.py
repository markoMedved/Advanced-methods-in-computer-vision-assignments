import numpy as np
import cv2
from ex2_utils import get_patch, gausssmooth, Tracker
from ex3_utils import create_cosine_window, create_gauss_peak
import matplotlib.pyplot as plt
import seaborn as sns



class CorrelationFiltersTracker(Tracker):
    def name(self):
        return 'corr_tracker'
    
    def get_H_conj(self, patch):
        F = np.fft.fft2(patch)
        F_conj = np.conjugate(F)
        lmbd = np.ones_like(F) * self.parameters.lmbd
        H = self.G * F_conj /(F * F_conj + lmbd)
        return H

    def initialize(self, image, region):
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
        # convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) /255.0
        #image = (image - np.mean(image))/np.std(image)
        # Get the bounding box of the initialized frame, turn stuff to ints
        left, top, width, height = [int(round(el)) for el in region]
        # Initialize the first position 
        self.position = (left + width//2, top + height//2)
        # Get the patch which will be used as the template to follow, add the stuff so that it is the correct size
        width *= self.parameters.enlargement
        height *= self.parameters.enlargement
        self.patch_shape = (width + (1-width%2), height + (1-height%2))
        template,template_mask = get_patch(image, self.position, self.patch_shape)
        #template = np.log1p(template)
        self.patch = template

        # Multiply the patch with the Hannning window
        self.cosine_window = create_cosine_window(template.T.shape)
        template *= self.cosine_window 
        # Create the Gaussian response only once
        self.G = np.fft.fft2(create_gauss_peak(template.T.shape, self.parameters.sigma))
        # plt.imshow(create_gauss_peak(template.T.shape, self.parameters.sigma))
        # plt.show()
        self.H_conj = self.get_H_conj(template)

    def track(self, image):
        # convert image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        #image = (image - np.mean(image))/np.std(image)   
        patch, patch_mask = get_patch(image, self.position, self.patch_shape)
        #patch = np.log1p(patch)
        patch *= self.cosine_window

        F = np.fft.fft2(patch)
        # response of filter
        self.R = np.fft.ifft2(self.H_conj * F).real
        # TODO should i do gradient ascent
        # Get the maximum positions
        y_max, x_max = np.unravel_index(np.argmax(self.R), self.R.shape)
    
        height, width = self.R.shape

        if x_max > width/2:
            x_max = x_max - width
        if y_max > height/2:
            y_max = y_max - height
        
        self.position = (self.position[0] + x_max, self.position[1] + y_max)
        
        # Update filter
        self.patch, patch_mask = get_patch(image, self.position, self.patch_shape)
        #self.patch = np.log1p(self.patch)
        self.patch *= self.cosine_window
        
        self.H_conj = self.H_conj * (1- self.parameters.alpha) + self.parameters.alpha * self.get_H_conj(self.patch)
        return [self.position[0] - self.patch_shape[0]//2, self.position[1] - self.patch_shape[1]//2, self.patch_shape[0]//self.parameters.enlargement, self.patch_shape[1]//self.parameters.enlargement]

class CFParams():
    def __init__(self,sigma = 1, lmbd = 1, alpha = 0.3, enlargement = 1.1):
        self.sigma=sigma
        self.lmbd = lmbd
        self.alpha = alpha
        self.enlargement = enlargement



class MOSSETracker(Tracker):
    def name(self):
        return 'corr_tracker'
    
    def get_A_B(self, patch):
        F = np.fft.fft2(patch)
        F_conj = np.conjugate(F)
        lmbd = np.ones_like(F) * self.parameters.lmbd
        A = self.G * F_conj 
        B = (F * F_conj + lmbd)
        return A,B

    def initialize(self, image, region):
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
        # convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) /255.0
        # Get the bounding box of the initialized frame, turn stuff to ints
        left, top, width, height = [int(round(el)) for el in region]
        # Initialize the first position 
        self.position = (left + width//2, top + height//2)
        # Get the patch which will be used as the template to follow, add the stuff so that it is the correct size
        width *= self.parameters.enlargement
        height *= self.parameters.enlargement
        self.patch_shape = (width + (1-width%2), height + (1-height%2))
        template,template_mask = get_patch(image, self.position, self.patch_shape)
        # Multiply the patch with the Hannning window
        self.cosine_window = create_cosine_window(template.T.shape)
        template *= self.cosine_window 


        # Create the Gaussian response only once
        self.G = np.fft.fft2(create_gauss_peak(template.T.shape, self.parameters.sigma))
        # plt.imshow(create_gauss_peak(template.T.shape, self.parameters.sigma))
        # plt.show()
        self.A, self.B = self.get_A_B(template)

    def track(self, image):
        # convert image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.0   
        patch, patch_mask = get_patch(image, self.position, self.patch_shape)
        patch *= self.cosine_window

        F = np.fft.fft2(patch)
        # response of filter
        self.R = np.fft.ifft2(self.A/self.B * F).real
        # TODO should i do gradient ascent
        # Get the maximum positions
        y_max, x_max = np.unravel_index(np.argmax(self.R), self.R.shape)

    
        height, width = self.R.shape

        if x_max > width/2:
            x_max = x_max - width
        if y_max > height/2:
            y_max = y_max - height
        self.position = (self.position[0] + x_max, self.position[1] + y_max)
        
        # Update filter
        self.patch, patch_mask = get_patch(image, self.position, self.patch_shape)
        self.patch *= self.cosine_window
        new_A, new_B = self.get_A_B(self.patch)
        self.A = self.A * (1- self.parameters.alpha) + self.parameters.alpha * new_A
        self.B = self.B * (1- self.parameters.alpha) + self.parameters.alpha * new_B
        return [self.position[0] - self.patch_shape[0]//2, self.position[1] - self.patch_shape[1]//2, self.patch_shape[0]//self.parameters.enlargement, self.patch_shape[1]//self.parameters.enlargement]

class MOSSEParams():
    def __init__(self,sigma = 1, lmbd = 1, alpha = 0.3, enlargement = 1.1):
        self.sigma=sigma
        self.lmbd = lmbd
        self.alpha = alpha
        self.enlargement = enlargement