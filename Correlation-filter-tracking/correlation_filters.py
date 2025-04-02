import numpy as np
from ex2_utils import get_patch, gausssmooth, Tracker


class CorrelationFiltersTracker(Tracker):
    def initialize(self):
        