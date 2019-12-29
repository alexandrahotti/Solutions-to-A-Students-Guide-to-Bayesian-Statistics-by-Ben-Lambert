from symmetric_model import *
from assymmetric_model import *

import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.special



def main():
    # Q 13.1.1
    #indep_sampling()

    # Q 13.1.2

    # mean_conj_prior = conjugate_prior()
    # print('Mean of dist with conjugate prior: ', mean_conj_prior)

    # Q 13.1.3
    #genrate_samples_conjugate()

    # Q 13.1.5
    # var_samples_conjugate()

    # Q 13.1.10
    #step_sz = 0.1
    #random_walk_metropolis()

    # Q 13.1.11
    #random_walk_metropolis_chains()

    # Q 13.1.13
    # step_sizes = [0.01, 0.1, 0.999]
    # colors = ['blue','green','yellow']
    # for i in range(len(step_sizes)):
    # #step_sz = 0.1
    #     random_walk_metropolis(step_sizes[i],colors[i])
    # plt.show()

    # Q 13.1.16
    #posterior_pred_more_data()

    """ Q 13.1.17-21 """
    # visulize_prior_MH()
    alpha_posterior_estimate, beta_posterior_estimate = metropolis_hastings()

    # Plot the results
    plot_joint_density(alpha_posterior_estimate, beta_posterior_estimate)
    plot_chains(alpha_posterior_estimate, beta_posterior_estimate)
    plot_1D_density(alpha_posterior_estimate, "Alpha")
    plot_1D_density(beta_posterior_estimate, "Beta")

main()
