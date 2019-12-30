import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.special


def get_list_val(possible_list):

    try:
        val = possible_list[0]
    except:
        val = possible_list

    return val


def Y1_binomial_conditional_rvs(a, pi, S, C):

    prob = pi * S / (pi * S + (1 - pi) * (1 - C))

    return get_list_val(scipy.stats.binom.rvs(a, prob, size=1))


def Y2_binomial_conditional_rvs(b, pi, S, C):

    prob = pi * (1 - S) / (pi * (1 - S) + (1 - pi) * C)

    return get_list_val(scipy.stats.binom.rvs(b, prob, 0, 1))


def pi_beta_conditional_rvs(Y1, Y2, a, b, alpha, beta):
    """ With a uniform prior"""

    return get_list_val(np.random.beta(Y1 + Y2 + alpha, a + b - Y1 - Y2 + beta, 1))


def S_beta_conditional_rvs(Y1, Y2, alpha, beta):
    """ With a uniform prior"""

    return get_list_val(np.random.beta(Y1 + alpha, Y2 + beta, 1))


def C_beta_conditional_rvs(Y1, Y2, a, b, alpha, beta):
    """ With a uniform prior"""

    return get_list_val(np.random.beta(b - Y2 + alpha, a - Y1 + beta, 1))


def plot_1D_density(samples, title_txt, c):
    """ plots a 1D density from a sampled range."""

    sns.distplot(samples, hist=True,
                 kde=True, bins=int(20), color=c)
    plt.title(title_txt)
    plt.show()


def gibbs_sampling(uniform_prior=True, informative_prior = 3):

    no_iterations = 8000
    a = 200
    b = 800

    """ Parameter initilization """
    Y1_i = get_list_val(np.random.randint(1, a, 1))
    Y2_i = get_list_val(np.random.randint(a, b+a, 1))
    pi_i = get_list_val(np.random.uniform(0, 1, 1))
    S_i = get_list_val(np.random.uniform(0, 1, 1))
    C_i = get_list_val(np.random.uniform(0, 1, 1))

    Y1_posterior_samples = []
    Y2_posterior_samples = []
    pi_posterior_samples = []
    S_posterior_samples = []
    C_posterior_samples = []

    if uniform_prior:
        # uniform prior means that we have a Beta(1,1) = Beta(alpha,beta) prior.
        alpha_pi = 1
        beta_pi = 1

        alpha_S = 1
        beta_S = 1

        alpha_C = 1
        beta_C = 1
    else:
        if informative_prior == 1:
            alpha_pi = 1
            beta_pi = 1

            alpha_S = 10
            beta_S = 1

            alpha_C = 10
            beta_C = 1

        if informative_prior == 2:
            alpha_pi = 1
            beta_pi = 10

            alpha_S = 1
            beta_S = 1

            alpha_C = 1
            beta_C = 1

        if informative_prior == 3:
            alpha_pi = 1
            beta_pi = 10

            alpha_S = 1
            beta_S = 1

            alpha_C = 10
            beta_C = 1

    for it in range(no_iterations):

        Y1_i = Y1_binomial_conditional_rvs(a, pi_i, S_i, C_i)
        Y2_i = Y2_binomial_conditional_rvs(b, pi_i, S_i, C_i)
        pi_i = pi_beta_conditional_rvs(Y1_i, Y2_i, a, b, alpha_pi, beta_pi)
        S_i = S_beta_conditional_rvs(Y1_i, Y2_i, alpha_S, beta_S)
        C_i = C_beta_conditional_rvs(Y1_i, Y2_i, a, b, alpha_C, beta_C)

        Y1_posterior_samples.append(Y1_i)
        Y2_posterior_samples.append(Y2_i)
        pi_posterior_samples.append(pi_i)
        S_posterior_samples.append(S_i)
        C_posterior_samples.append(C_i)

    # plot_1D_density(pi_posterior_samples, r"$\pi$" + " posterior. Using Informative Beta(10,1) Priors")
    plot_1D_density(S_posterior_samples, " S posterior ", 'g')
    plot_1D_density(C_posterior_samples, " C posterior ", 'g')
    plot_1D_density(Y1_posterior_samples, " Y1 posterior ", 'b')
    plot_1D_density(Y2_posterior_samples, " Y2 posterior ", 'b')



def main():
    """ Q 14.1.3 """
    # uniform_prior = True
    # gibbs_sampling(uniform_prior)

    """ Q 14.1.4 """
    # uniform_prior = False
    # informative_prior = 1
    # gibbs_sampling(uniform_prior, informative_prior)

    """ Q 14.1.5 """
    uniform_prior = False
    informative_prior = 2
    gibbs_sampling(uniform_prior, informative_prior)
    prior = np.random.beta(1, 10, 1000)
    plot_1D_density(prior, 'Prior Beta(1,10)', 'orange')

    """ Q 14.1.6 """
    uniform_prior = False
    informative_prior = 3
    #gibbs_sampling(uniform_prior, informative_prior)


main()
