import numpy as np
import scipy.stats


class Election_model:

    def __init__(self, nA, nB, nC, n):
        self.nA = nA
        self.nB = nB
        self.nC = nC
        self.n = n

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
                likelihood_grid[pB_ind, pA_ind] = likelihood

        return likelihood_grid, pB_range, pA_range

    def factorial(self, n):
        """
            Computes: n!
        """

        fact = 1

        for n_i in range(1, n + 1):
            fact = fact * n_i

        return fact

    def dirichlet_prior(self, alpha_1, alpha_2, alpha_3):
        """
            The Dirichlet(1,1,1) prior over p_A, p_B and p_c.
        """
        # The probabilities vary betwen [0,1]. I do not go exactly to 0 and 1.
        p_lower = 0.002
        p_upper = 0.99

        pA_range = np.linspace(p_lower, p_upper, 1000)
        pB_range = np.linspace(p_lower, p_upper, 1000)

        prior_grid = np.zeros((len(pA_range), len(pB_range)))

        for pA_ind in range(len(pA_range)):
            for pB_ind in range(len(pB_range)):

                pA = pA_range[pA_ind]
                pB = pB_range[pB_ind]

                theta = [pA, pB]

                try:
                    prior_pdf = scipy.stats.dirichlet.pdf(
                        theta, [alpha_1, alpha_2, alpha_3])

                    # Storing the prior in a grid.
                    prior_grid[pA_ind, pB_ind] = prior_pdf
                except:
                    print("invalid!")

        return prior_grid, pB_range, pA_range

    def dirichlet_posterior(self, alpha_1, alpha_2, alpha_3):
        """
            The Dirichlet(alpha_1+n_A, alpha_1+n_A,alpha_1+n_C) using a Multinomial
            Likelihood and a conjugate Dirichlet prior.
        """
        # The probabilities vary betwen [0,1]. I do not go exactly to 0 and 1.
        p_lower = 0.01
        p_upper = 0.99

        pA_range = np.linspace(p_lower, p_upper, 1000)
        pB_range = np.linspace(p_lower, p_upper, 1000)

        posterior_grid = np.zeros((len(pA_range), len(pB_range)))

        for pA_ind in range(len(pA_range)):
            for pB_ind in range(len(pB_range)):

                pA = pA_range[pA_ind]
                pB = pB_range[pB_ind]

                theta = [pA, pB]

                try:
                    posterior_pdf = scipy.stats.dirichlet.pdf(
                        theta, [alpha_1 + self.nA, alpha_2 + self.nB, alpha_3 + self.nC])

                    # Storing the posterior in a grid.
                    posterior_grid[pB_ind, pA_ind] = posterior_pdf
                except:
                    print("invalid!")

        return posterior_grid, pB_range, pA_range

    def posterior_mean(self, alpha_1, alpha_2, alpha_3):

        posterior_mean = scipy.stats.dirichlet.mean(
            [alpha_1 + self.nA, alpha_2 + self.nB, alpha_3 + self.nC])

        return posterior_mean
