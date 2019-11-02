import numpy as np
from scipy.stats import binom
from scipy.stats import beta

class Lyme_disease_model:

    def __init__(self, sample_size, X):
        self.sample_size = sample_size
        self.no_disease_occurances = X


    def binomial_likelihood(self):
        """
            Computes the binomial likelihood over theta.
        """

        # The probabilities vary betwen [0,1].
        theta_lower = 0
        theta_upper = 1

        theta_range = np.linspace(theta_lower, theta_upper, 100)

        """ Likelihood: vary the parameters and keep the data fixed. """

        binomial_likelihoods = binom.pmf(self.no_disease_occurances, self.sample_size, theta_range)

        return binomial_likelihoods, theta_range


    def binomial_sampling_distribution(self, theta):

        disease_occurances_range = np.linspace(0, 10, 11)
        binomial_prob_dist = binom.pmf(disease_occurances_range, self.sample_size, theta)

        return binomial_prob_dist, disease_occurances_range

    def beta_prior(self, a, b):
        """
            A beta(a,b) prior on the porportion of disease cases: theta.
        """

        theta_lower = 0
        theta_upper = 1

        theta_range = np.linspace(theta_lower, theta_upper, 100)

        beta_prior_distribution = beta.pdf(theta_range, a, b)

        return theta_range, beta_prior_distribution


    def beta_posterior(self, a, b):
        """
            A beta(a,b) prior on the porportion of disease cases: theta.
        """

        theta_lower = 0
        theta_upper = 1

        theta_range = np.linspace(theta_lower, theta_upper, 100)

        beta_posterior_distribution = beta.pdf(theta_range, self.no_disease_occurances + a, self.sample_size + b - self.no_disease_occurances)

        beta_posterior_mean = beta.mean(self.no_disease_occurances + a, self.sample_size + b - self.no_disease_occurances)

        return theta_range, beta_posterior_distribution, beta_posterior_mean


    def posterior_predictive(self, a, b, sample_size_new_data, no_iterations):
        """ The posterior predictive sampling using a beta(11 + 1, 200 -1 + 1)
        posterior and  a Binomial(100, theta) likelihood. """

        posterior_predictive_disease_occurances = []

        for i in range(no_iterations):
            """ Sampling a theta value from the posterior. """
            sample_sz =  1
            theta_i = np.random.beta(self.no_disease_occurances + a, self.sample_size + b - self.no_disease_occurances, sample_sz)

            """ Sampling x_i (disease occurances) for the new data from a Binomial(100, theta_i) Likelihood for this new data over theta_i. """
            sample_sz =  1
            disease_occurance_sample = binom.rvs(sample_size_new_data, theta_i, 0, sample_sz)

            posterior_predictive_disease_occurances.append(disease_occurance_sample[0])

        return posterior_predictive_disease_occurances
