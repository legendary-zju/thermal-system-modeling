import numpy as np
import pandas as pd
from math import sqrt

class Particles:
    def __init__(self, index : tuple = (0, 0), dimension: int = 1):
        self.index = index

        self.dimension = dimension
        self.variables = np.zeros(self.dimension)
        self.delta_variables = np.zeros(self.dimension)
        self.jacobin_slipy = np.zeros([self.dimension, self.dimension])
        self.jacobin_temp = np.zeros([self.dimension, self.dimension])
        self.residual_slipy = np.zeros(self.dimension)
        self.residual_temp = np.zeros(self.dimension)

        self.alpha = 0.5
        self.beta = 0.05
        self.gamma = 0.005
        self.zeta = 0.5

        self.prior_belief = 1
        self.predict_belief = 1
        self.posterior_belief = 1

        self.prev_posterior = 0

        self.feasible = True
        self.is_elite = False

    def regenerate(self, update_belief, layer):
        self.prior_belief = update_belief
        self.predict_belief = 1
        self.posterior_belief = 1
        self.index = (layer, self.index[1])
        self.feasible = True

