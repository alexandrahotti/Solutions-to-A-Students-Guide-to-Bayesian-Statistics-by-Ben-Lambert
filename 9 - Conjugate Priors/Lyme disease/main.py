from helper_functions import plotter, plotter_discrete, plotter_histogram
from Lyme_disease_model import *
import matplotlib.pyplot as plt


def main():

    """ Q 9.1.3 The Binomial Likelihood """
    # no_ticks = 10
    # no_disease_occurances = 1
    # lyme_disease_model = Lyme_disease_model(no_ticks, no_disease_occurances)

    # binomial_likelihoods, theta_range = lyme_disease_model.binomial_likelihood()

    # plotter(theta_range, binomial_likelihoods,
    #         "Binomail Likelihood with 1 data trial over " + r'$\theta$', r'$\theta$' )


    """ Q 9.1.4-5 The Binomial Sampling Distribution over X"""
    # theta = 0.1
    #
    # binomial_prob_dist, disease_occurances_range = lyme_disease_model.binomial_sampling_distribution(theta)
    #
    # plotter_discrete(disease_occurances_range, binomial_prob_dist,
    #         " Binomial Sampling Distribution over " + r'$X_i$'+" for 1 trail", r'$X_i$' )
    #
    # print(" Sum over the sampling distribution: ", np.sum(binomial_prob_dist))


    """ Q 9.1.7 A beta(1, 1) prior """
    # a = 1
    # b = 1
    # theta_range, beta_prior_distribution = lyme_disease_model.beta_prior(a, b)
    #
    # plotter(theta_range, beta_prior_distribution,
    #         "Beta(1, 1) prior over " + r'$\theta$', r'$\theta$' )



    """ Q 9.1.10 A beta( 1 + 1, 10 - 1 + 1) posterior """
    # a = 1
    # b = 1
    # theta_range, beta_posterior_distribution, posterior_mean = lyme_disease_model.beta_posterior(a, b)
    #
    # print("Posterior mean: ", posterior_mean )
    # plotter(theta_range, beta_posterior_distribution,
    #         "beta( 1 + 1, 10 - 1 + 1)  posterior over " + r'$\theta$', r'$\theta$' )


    """ Q 9.1.11 A beta( 7 + 1, 100 - 7 + 1)  posterior """
    # no_ticks = 100
    # no_disease_occurances = 7
    # lyme_disease_model = Lyme_disease_model(no_ticks, no_disease_occurances)
    #
    # a = 1
    # b = 1
    # theta_range, beta_posterior_distribution, posterior_mean = lyme_disease_model.beta_posterior(a, b)
    #
    # print("Posterior mean: ", posterior_mean )
    # plotter(theta_range, beta_posterior_distribution,
    #         "beta( 7 + 1, 100 - 7 + 1)  posterior over " + r'$\theta$', r'$\theta$', "pdf" )


    """ Q 9.1.13 posterior predictive """

    # Previous data parameters
    no_ticks = 200
    no_disease_occurances = 11
    lyme_disease_model = Lyme_disease_model(no_ticks, no_disease_occurances)

    # Prior parameters
    a = 1
    b = 1

    # New data parameter
    sample_size_new_data = 100

    # Posterior predictive sampling
    no_iterations = 10000
    posterior_predictive_disease_samples = lyme_disease_model.posterior_predictive(a, b, sample_size_new_data, no_iterations)
    plotter_histogram(posterior_predictive_disease_samples, "Disease occurances in a sample of 100 ticks", "Counts", "Estimation of the Posterior Predictive Distribution")



if __name__ == '__main__':
    main()
