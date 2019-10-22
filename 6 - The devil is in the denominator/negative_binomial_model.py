from scipy.integrate import quad, dblquad
from scipy.stats import nbinom
import matplotlib.pyplot as plt
import numpy as np


def compute_likelihood(failuers_before_five_successes, theta1_range, theta2_range):
    """
        Computes the likelihood over the range (0, 1) for two theta parameters.
        The likelihood is modeled by a Negative Binomial pmf.
    """

    no_successes = 5

    likelihood_grid = np.zeros((len(theta1_range), len(theta2_range)))

    for x in range(len(theta1_range)):
        for y in range(len(theta2_range)):
            total_likelihood = 0
            theta1 = theta1_range[x]
            theta2 = theta2_range[y]

            for data_point_failures in failuers_before_five_successes:

                p = theta1 * theta2 + (1 - theta1) * (1 - theta2)
                likelihood = np.log(nbinom.pmf(
                    data_point_failures, no_successes, p))
                total_likelihood += likelihood

            likelihood_grid[x, y] = total_likelihood

    return np.exp(likelihood_grid)
