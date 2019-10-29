import numpy as np


class Election_model:

    def __init__(self):
        self.nA = 6
        self.nB = 3
        self.nC = 1
        self.n = 10

    def factorial(self, n):
        """
            Computes: n!
        """

        fact = 1

        for n_i in range(1, n + 1):
            fact = fact * n_i

        return fact

    def compute_multinomial_likelihood(self):
        """
            Computes the multinomial likelihood over p_A and p_b (which vary over
            [0,1]) conditioned on the counts n_A, n_B and n_C.
        """

        # The probabilities vary betwen [0,1]. I do not go exactly to 0 and 1 to avoid
        # logging 0 values.
        p_lower = 0.001
        p_upper = 0.999

        pA_range = np.linspace(p_lower, p_upper, 1000)
        pB_range = np.linspace(p_lower, p_upper, 1000)

        likelihood_grid = np.zeros((len(pA_range), len(pB_range)))

        """ Likelihood: vary the parameters and keep the data fixed. """
        for pA_ind in range(len(pA_range)):
            for pB_ind in range(len(pB_range)):

                pA = pA_range[pA_ind]
                pB = pB_range[pB_ind]

                # Possible number of ways of getting these counts in our sample.
                no_combinations = self.factorial(self.nA + self.nB + self.nC) / (
                    self.factorial(self.nA) * self.factorial(self.nB) * self.factorial(self.nC))

                # Log the values to avoid underflow later on.
                log_combinations = np.log(no_combinations)

                # The logged likelihood of the multinomial distribution.
                log_likelihood = log_combinations + self.nA * \
                    np.log(pA) + self.nB * np.log(pB) + (self.n -
                                                         self.nA - self.nB) * np.log(1 - pA - pB)

                # Unlogging the likelihood.
                likelihood = np.exp(log_likelihood)

                # Storing the likelihood in a grid.
                likelihood_grid[pA_ind, pB_ind] = likelihood

        return likelihood_grid, pB_range, pA_range
