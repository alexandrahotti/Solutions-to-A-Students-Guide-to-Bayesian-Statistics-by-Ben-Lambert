import numpy as np
from scipy.stats import gamma


class Epilepsy_model:

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def posterior_gamma(self, epilepsy_counts):

        theta_lower = 0
        theta_upper = 1

        theta_range = np.linspace(theta_lower, theta_upper, 100)

        a_prim = np.sum(epilepsy_counts) + self.a
        b_prim = self.b + len(epilepsy_counts)

        gamma_posterior = gamma.pdf(
            theta_range, a = a_prim, scale = (1 /b_prim))

        return gamma_posterior, theta_range
