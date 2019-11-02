from helper_functions import plotter
from Election_model import *
import matplotlib.pyplot as plt


def main():

    """ The Multinomial Likelihood  for n_A = 6, n_B = 3 and n_C = 1 """
    Election_Likelihood_Model = Election_model(6, 3, 1, 10)

    likelihoods, pB_range, pA_range = Election_Likelihood_Model.compute_multinomial_likelihood()

    plotter(pA_range, pB_range, likelihoods,
            "Multinomial Likelihood over " + r'$p_A$' + " and " + r'$p_B$' , r'$p_A$' , r'$p_B$')


    """ The Conjugate Dirichlet(1, 1, 1) prior """
    priors, pB_range, pA_range = Election_Likelihood_Model.dirichlet_prior(1, 1, 1)

    plotter(pA_range, pB_range, priors,
                "Dirichlet prior using " + r'$\alpha_A, \alpha_B, \alpha_C = 1 $' + " for  "+ r'$p_A$' +" and " + r'$p_B$' , r'$p_A$' , r'$p_B$', plt.cm.YlGnBu)

    """ The Dirichlet(1, 1, 1) posterior """
    posterior, pB_range, pA_range = Election_Likelihood_Model.dirichlet_posterior(
        1, 1, 1)

    plotter(pA_range, pB_range, posterior,
            "Posterior with Multinomial Likelihood and Dirichlet(1, 1, 1) prior over  " + r'$p_A$' + " and " + r'$p_B$', r'$p_A$', r'$p_B$')

    """ Posterior Mean using Dirichlet uniform prior"""
    posterior_mean_dirichlet_uniform_prior = Election_Likelihood_Model.posterior_mean(
        1, 1, 1)
    print("Posterior mean parameters: p_A = ", posterior_mean_dirichlet_uniform_prior[0], " p_B = ", posterior_mean_dirichlet_uniform_prior[1], " p_C = ", posterior_mean_dirichlet_uniform_prior[2])



    """ The Conjugate Dirichlet(10, 10, 10) prior """
    priors, pB_range, pA_range = Election_Likelihood_Model.dirichlet_prior(10, 10, 10)

    plotter(pA_range, pB_range, priors,
                "Dirichlet(10, 10, 10) prior using " + r'$\alpha_A, \alpha_B, \alpha_C = 1 $' + " for  "+ r'$p_A$' +" and " + r'$p_B$' , r'$p_A$' , r'$p_B$', plt.cm.YlGnBu)

    """ The Dirichlet(10+6, 10+3, 10+1) posterior """
    posterior, pB_range, pA_range = Election_Likelihood_Model.dirichlet_posterior(
        10, 10, 10)

    plotter(pA_range, pB_range, posterior,
            "Posterior with Multinomial Likelihood and Dirichlet(10, 10, 10) prior over  " + r'$p_A$' + " and " + r'$p_B$', r'$p_A$', r'$p_B$')


    """ The Multinomial Likelihood  for n_A = 60, n_B = 30 and n_C = 10 """

    Election_Likelihood_Model = Election_model(60, 30, 10, 100)

    likelihoods, pB_range, pA_range = Election_Likelihood_Model.compute_multinomial_likelihood()

    plotter(pA_range, pB_range, likelihoods,
            "Multinomial Likelihood over " + r'$p_A$' + " and " + r'$p_B$' , r'$p_A$' , r'$p_B$')


    """ The Dirichlet(10 + 60, 10 + 30, 10 + 10) posterior """
    posterior, pB_range, pA_range = Election_Likelihood_Model.dirichlet_posterior(
        10, 10, 10)

    plotter(pA_range, pB_range, posterior,
            "Posterior Dirichlet(10 + 60, 10 + 30, 10 + 10) with Multinomial Likelihood and Dirichlet(10, 10, 10) prior over  " + r'$p_A$' + " and " + r'$p_B$', r'$p_A$', r'$p_B$')


if __name__ == '__main__':
    main()
