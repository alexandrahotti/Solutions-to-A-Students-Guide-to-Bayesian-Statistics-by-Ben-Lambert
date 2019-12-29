import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.special
import pandas as pd


def visulize_prior_MH():

    a_range = np.linspace(0, 4, 1024 / 4)
    b_range = np.linspace(0, 20, (1024 / 4) * 5)

    a_density = scipy.stats.gamma.pdf(a_range, a=1, loc=0, scale=(1 / (1 / 8)))
    b_density = scipy.stats.gamma.pdf(b_range, a=8, loc=0, scale=1)

    prior_matrix = np.zeros((len(b_density), len(a_density)))

    for b in range(len(b_density)):
        for a in range(len(a_density)):

            prior_matrix[b][a] = b_density[b] * a_density[a]

    plotter(a_range, b_range, prior_matrix,
            'Joint prior', r"$\alpha$", r"$\beta$")


def plotter(parameter1_range, parameter2_range, value_grid, title_string, x_label, y_label):
    """
        Creates a 2D contour plot over two parameters.
    """

    plt.contourf(parameter1_range, parameter2_range, value_grid, cmap='magma')

    plt.title(title_string)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()


def get_list_val(possible_list):

    try:
        val = possible_list[0]
    except:
        val = possible_list

    return val


def log_norm_proposed(param_prev, step_sz):
    """
        The assymetric jumpig kernel. Propose the next parameter
        using a log-Normal kernel centered on the current alpa/beta-
        parameter estimate.
    """

    mean_prev = np.exp(np.log(param_prev) - 0.5 * step_sz**2)
    param_proposed = get_list_val(
        scipy.stats.lognorm.rvs(s=step_sz, scale=mean_prev, size=1))

    return param_proposed


def betabinomial_pmf(k_list, size, a, b):
    """
        The beta binomial likelihood.
    """
    likelihood = 1

    for x in k_list:
        likelihood *= get_list_val(scipy.special.comb(size, x) * scipy.special.beta(
            x + a, size - x + b) / scipy.special.beta(a, b))

    return likelihood


def compute_alpha_prior(param):
    """ A Gamma(1, 1/8) prior """

    a_density = scipy.stats.gamma.pdf(param, a=1, loc=0, scale=8)

    return get_list_val(a_density)


def compute_beta_prior(param):
    """ A Gamma(1, 10) prior """

    b_density = scipy.stats.gamma.pdf(param, a=10, loc=0, scale=1)

    return get_list_val(b_density)


def metropolis_hastings():
    """
        Metropolis Hastings with an assymetric log-Normal jumping kernel, a beta binomial likelihood
        and gamma priors.
    """
    r = 0
    no_iterations = 8000

    alpha_curr = get_list_val(np.random.uniform(0, 3, 1))
    beta_curr = get_list_val(np.random.uniform(0, 20, 1))

    alpha_posterior_estimate = []
    beta_posterior_estimate = []

    step_sz_a = 0.5
    step_sz_b = 0.5

    burn_in = 50

    # The data
    k = [6, 3, 2, 8, 25]
    n = 100

    # Used to calc acceptance rate.
    no_accepts = 0

    for it in range(no_iterations):

        """ Proposal pdf values"""
        alpha_prop = get_list_val(log_norm_proposed(alpha_curr, step_sz_a))
        beta_prop = get_list_val(log_norm_proposed(beta_curr, step_sz_b))

        alpha_prop_prior = compute_alpha_prior(alpha_prop)
        beta_prop_prior = compute_beta_prior(beta_prop)
        # The two priors are idenpendent, so the joint prior is the multiplication between them.
        prop_prior = alpha_prop_prior * beta_prop_prior

        prop_likelihood = betabinomial_pmf(k, n, alpha_prop, beta_prop)

        """ Current pdf values"""
        alpha_curr_prior = compute_alpha_prior(alpha_curr)
        beta_curr_prior = compute_beta_prior(beta_curr)
        # The two priors are idenpendent, so the joint prior is the multiplication between them.
        curr_prior = alpha_curr_prior * beta_curr_prior

        curr_likelihood = betabinomial_pmf(k, n, alpha_curr, beta_curr)

        """ The jumping kernels between the current and the proposed position. """
        j_prop_curr = get_list_val(scipy.stats.lognorm.pdf(alpha_prop, scale=np.exp((np.log(alpha_curr) - 0.5 * step_sz_a**2)), s=step_sz_a)) * \
            get_list_val(scipy.stats.lognorm.pdf(beta_prop, scale=np.exp(
                np.log(beta_curr) - 0.5 * step_sz_b**2), s=step_sz_b))
        j_curr_prop = get_list_val(scipy.stats.lognorm.pdf(alpha_curr, scale=np.exp(np.log(alpha_prop) - 0.5 * step_sz_a**2), s=step_sz_a)) * \
            get_list_val(scipy.stats.lognorm.pdf(beta_curr, scale=np.exp(
                np.log(beta_prop) - 0.5 * step_sz_b**2), s=step_sz_b))

        """ The Accept Reject step """
        if (prop_likelihood * prop_prior) >= (curr_likelihood * curr_prior):
            r = 1
        else:
            r = ((prop_likelihood * prop_prior) /
                 (curr_likelihood * curr_prior)) * (j_curr_prop / j_prop_curr)

        a = get_list_val(np.random.uniform(0, 1, 1))

        if r > a:
            alpha_curr = alpha_prop
            beta_curr = beta_prop
            no_accepts += 1

        # Store samples when the number of iterations is above the burn in.
        if no_iterations > burn_in:

            alpha_posterior_estimate.append(alpha_curr)
            beta_posterior_estimate.append(beta_curr)

    print('Acceptance ratio ', no_accepts / no_iterations)

    return alpha_posterior_estimate, beta_posterior_estimate


def plot_joint_density(alpha_vals, beta_vals):
    """ plots a 2D joint density from 2 sampled ranges."""

    df2 = pd.DataFrame({r"$\alpha$": alpha_vals,
                        r"$\beta$": beta_vals})

    sns.jointplot(data=df2, x=r"$\alpha$", y=r"$\beta$", kind='kde', color="g")
    plt.show()


def plot_chains(alpha_vals, beta_vals):
    """" Plots the evolution of the samplers accepted values. """
    plt.plot(alpha_vals)
    plt.plot(beta_vals)

    plt.title("Beta (orange) and alpha (blue)")

    plt.show()


def plot_1D_density(samples, title_txt):
    """ plots a 1D density from a sampled range."""

    sns.distplot(samples, hist=True,
                 kde=True, bins=int(20), color="g")
    plt.title(title_txt)
    plt.show()
