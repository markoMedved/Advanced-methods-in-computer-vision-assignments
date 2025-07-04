import numpy as np
import cv2
from ex2_utils import get_patch, create_epanechnik_kernel, extract_histogram
from ex4_utils import sample_gauss
from utils.tracker import Tracker

# TODO after it works nicely, change it to test it with the toolkit

class ParticleFilterTracker(Tracker):
    def name(self):
        return 'particle_filter_tracker'
    

    def initialize(self, image, region, seed = 32):
        """Initialized the tracker with the provided bounding box for the first frame"""
        # Set the random seed
        np.random.seed(seed=seed)

        # Set the parameters
        self.parameters = ParticleFilterParams()
        
        # Convert image to the target color space
        if self.parameters.color_space in ["cv2.COLOR_BGR2RGB", "cv2.COLOR_BGR2LAB", 
                                           "cv2.COLOR_BGR2HSV", "cv2.COLOR_BGR2YCrCb"]:
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

        # Matrices
        self.q = self.parameters.q_size_ratio * min(width, height)
        if self.parameters.motion_model == "NCV":
            self.Fi = np.array([[1,0,1,0],
                                [0,1,0,1],
                                [0,0,1,0],
                                [0,0,1,0]])
            self.Q = self.q * np.array([[1/3, 0, 1/2, 0],
                                        [0,1/3,0,1/2],
                                        [1/2, 0, 1, 0],
                                        [0, 1/2, 0, 1]])
            self.H = np.array([[1,0,0,0],
                            [0,1,0,0]])
            
        elif self.parameters.motion_model == "RW":
            self.Fi = np.array([[1,0],
                                [0,1]])
            
            self.Q = self.q * np.array([[1,0],
                                        [0,1]])
            self.H = np.array([[1,0],
                            [0,1]])
            
        elif self.parameters.motion_model == "NCA":
            self.Fi = np.array([[1. , 0. , 1. , 0. , 0.5, 0. ],
                                [0. , 1. , 0. , 1. , 0. , 0.5],
                                [0. , 0. , 1. , 0. , 1. , 0. ],
                                [0. , 0. , 0. , 1. , 0. , 1. ],
                                [0. , 0. , 0. , 0. , 1. , 0. ],
                                [0. , 0. , 0. , 0. , 0. , 1. ]])
            self.Q = self.q * np.array([
                                [0.05, 0,    0.125, 0,    0.1667, 0],
                                [0,    0.05, 0,     0.125, 0,     0.1667],
                                [0.125, 0,   0.3333, 0,    0.5,    0],
                                [0,    0.125, 0,    0.3333, 0,     0.5],
                                [0.1667, 0,  0.5,   0,     1.0,    0],
                                [0,    0.1667, 0,   0.5,   0,      1.0]
                            ])
            self.H = np.array([[1,0,0,0,0,0],
                                [0,1,0,0,0,0]])

        self.state = np.zeros_like(self.Fi[0])
        self.state[0] = self.position[0]
        self.state[1] = self.position[1]

        # Initialize the particles with the normal distribution
        self.particles = sample_gauss(self.state, self.Q, self.parameters.num_particles)
        # If any go outside the frame set to mean
        self.particles[:, 0] = np.clip( self.particles[:, 0],0, image.shape[1])
        self.particles[:, 1] = np.clip( self.particles[:, 1],0, image.shape[0])
        # Initialize the weights of the particles
        self.particle_weights = np.ones_like(self.particles[:,1])


    def track(self, image):
         # Convert image to the target color space
        if self.parameters.color_space in ["cv2.COLOR_BGR2RGB", "cv2.COLOR_BGR2LAB", 
                                           "cv2.COLOR_BGR2HSV", "cv2.COLOR_BGR2YCrCb"]:
            image = cv2.cvtColor(image, eval(self.parameters.color_space))

        # normalize the weights
        weigts_tmp = self.particle_weights/ np.sum(self.particle_weights)
        # Get the cdf
        weights_cdf = np.cumsum(weigts_tmp)

        # sample N indices
        rand_samples = np.random.rand(len(self.particles), 1)
        samplesd_idxs = np.digitize(rand_samples, weights_cdf)

        # select the corresponding samples
        particles_new = self.particles[samplesd_idxs.flatten(), :]

        # Move the particles with the motion model and the noise
        self.particles = (self.Fi @ particles_new.T).T + sample_gauss(np.zeros_like(self.Q[0]), self.Q, self.parameters.num_particles)
        # Clip if any go outside the frame
        self.particles[:, 0] = np.clip( self.particles[:, 0],0, image.shape[1])
        self.particles[:, 1] = np.clip( self.particles[:, 1],0, image.shape[0])

        # Get histograms for particles and compare to target 
        for i, particle in enumerate(self.particles):
            # extract the histrogram of the current particle
            patch, mask = get_patch(image, (particle[0], particle[1]), self.patch_shape)
            current_kernel = self.kernel * mask
            hist = extract_histogram(patch, self.parameters.nbins, current_kernel)
            hist /= np.sum(hist)
            
            # Calculate hellinger distance
            dist = (1/np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(hist) - np.sqrt(self.template_hist))**2))
            # Convert the distance to probs TODO what is sigma
            self.particle_weights[i] = np.exp(- 1/2 * dist**2 /0.1**2)
            
            
        self.particle_weights /= np.sum(self.particle_weights)
        #Calculate the new position with the mean
        x = np.sum(self.particle_weights * self.particles[:,0])
        y = np.sum(self.particle_weights * self.particles[:,1])

        #update the template
        patch, mask = get_patch(image, (x,y), self.patch_shape)
        current_kernel = self.kernel * mask
        hist = extract_histogram(patch, self.parameters.nbins, current_kernel)
        self.template_hist = self.template_hist  * (1- self.parameters.alpha) + self.parameters.alpha * hist 
        self.template_hist /= np.sum(self.template_hist)

        return [int(x)- self.patch_shape[0]//2, int(y)- self.patch_shape[1]//2, self.patch_shape[0], self.patch_shape[1]]

        

class ParticleFilterParams():
    # Note for NCA set alpha higher
    def __init__(self, num_particles=100, q_size_ratio=0.9,
                  alpha = 0.0001, motion_model = "NCV",nbins=16, 
                  sigma=1, color_space = "RGB"):
        self.num_particles = num_particles
        self.q_size_ratio = q_size_ratio # Percentege of the target size
        self.nbins = nbins
        self.sigma = sigma
        self.color_space = f"cv2.COLOR_BGR2{color_space}"
        self.motion_model = motion_model
        self.alpha = alpha


