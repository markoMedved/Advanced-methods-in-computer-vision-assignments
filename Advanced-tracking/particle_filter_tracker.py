import numpy as np
import cv2
from ex2_utils import Tracker

class ParticleFilterTracker(Tracker):
    def name(self):
        return 'particle_filter_tracker'
    

    def initialize(self, image, region):
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        

    def track(self, image):


class ParticleFilterParams():
    def __init__(self):
        pass
    pass
