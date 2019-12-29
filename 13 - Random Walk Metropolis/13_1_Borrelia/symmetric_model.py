import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.special



def indep_sampling():
    """
        Independent sampling from the prior predictive with a beta(1,1) prior and
        a binomial likelihood.
    """

    no_iterations = 1000
    no_samples = 1
    a = 1
    b = 1
    n = 100
    borrelia_counts = []

    for it in range(no_iterations):

        # Sample a prob of having lyme disease form the prior
        p = np.random.beta(a, b, no_samples)[0]

        # Sample lyme disease counts from the binomial based on the sampled p.
        borrelia_count = scipy.stats.binom.rvs(n, p, 0, 1)[0]
        borrelia_counts.append(borrelia_count)



    plt.hist(borrelia_counts)
    plt.title('Posterior Predictive lyme disease - Independent sampling')
    plt.show()
    print(" Lyme disease count expectation:", np.mean(borrelia_counts))


def conjugate_prior():
    """ A Conjugate beta prior."""
    a = 1
    b = 1
    k = 6
    n = 100

    mean_conj_prior = scipy.stats.beta.mean(a + k, b + n - k)

    return mean_conj_prior

def genrate_samples_conjugate():
    """ Sample directly from the posterior with a Conjugate Prior."""
    a = 1
    b = 1
    k = 6
    n = 100

    no_samples = 100

    samples = np.random.beta(a + k, b + n - k, no_samples)
    plt.hist(samples)
    plt.title(' Conjugate prior distribution lyme disease - Independent sampling')

    x_range = np.linspace(0, 0.20, 1024)
    plt.plot(x_range, scipy.stats.beta.pdf(x_range, a + k, b + n - k), c = 'k')


    plt.show()


def var_samples_conjugate():

    a = 1
    b = 1
    k = 6
    n = 100

    no_samples = 100

    variances = []

    for it in range(no_samples):

        samples = list(np.random.beta(a + k, b + n - k, no_samples))
        variances = variances + samples

    plt.hist(variances)
    plt.title(' Estimated variance lyme disease - Independent sampling')
    plt.show()

    print("The actual pdf variance ", scipy.stats.beta.var(a + k, b + n - k))


def theta_proposed(theta_curr, step_sz):
    """ The propsed theta values are symmetrically bounded. """

    theta_prop = scipy.stats.norm.rvs(loc = theta_curr, scale = step_sz)

    return theta_prop%1


def get_list_val(possible_list):

    try:
        val = possible_list[0]
    except:
        val = possible_list

    return val


def random_walk_metropolis(step_sz, color_plot):
    """ Random walk metropolis with a symmetric Normal jumping kernel, a binomial likelihood
        and a Gamma prior.
        """

    r = 0
    no_iterations = 200

    theta_curr = np.random.uniform(0,1,1)

    a = 1
    b = 1
    k = 6
    n = 100

    theta_posterior_estimate = []

    for it in range(no_iterations):

        """ The proposed parameter. """
        theta_prop = get_list_val(theta_proposed(theta_curr, step_sz))
        prior_prop = get_list_val(scipy.stats.beta.pdf(theta_prop, a, b))
        likelihood_prop = get_list_val(scipy.stats.binom.pmf(k, n, theta_prop, loc=0))

        """ The Current parameter. """
        prior_curr = get_list_val(scipy.stats.beta.pdf(theta_curr, a, b))
        likelihood_curr = get_list_val(scipy.stats.binom.pmf(k, n, theta_curr, loc=0))


        """ The Accept Reject step. """
        if likelihood_prop*prior_prop >= likelihood_curr*prior_curr:
            r = 1

        else:
            r = (likelihood_prop*prior_prop) /(likelihood_curr*prior_curr)

        a = get_list_val(np.random.uniform(0,1,1))

        if r > a:
            theta_curr = theta_prop

        theta_posterior_estimate.append(theta_curr)

    # #plt.hist(theta_posterior_estimate, bins=range(min(theta_posterior_estimate), max(theta_posterior_estimate)))
    # # plt.hist(np.asarray(theta_posterior_estimate, dtype='float'), bins=100, normed=True)
    # # plt.title('Posterior Predictive lyme disease - Independent sampling')
    # # plt.show()
    # print(theta_posterior_estimate)
    plt.hist(theta_posterior_estimate, bins=10)
    # #sns.distplot(theta_posterior_estimate, hist=False, kde=True, bins=int(20),  hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 1})
    #
    # # x_range = np.linspace(0, 1, 1024)
    # # plt.plot(x_range, scipy.stats.beta.pdf(x_range, a + k, b + n - k), c = 'k', color = "green")
    # #
    # # indep_samples = np.random.beta(a + k, b + n - k, 100)
    # # sns.distplot(indep_samples, hist=False, kde=True, bins=int(20),  hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 1})
    #
    # #plt.title('Comparision between exact Posterior (green), Posterior estimated with Random Walk Metropolis (blue) and independently sampled posterior (orange)')
    # plt.title('Estimated posterior with a standard deviation (step size) of 1')
    # plt.show()
    # print(len(theta_posterior_estimate))
    # x_range = np.linspace(0, 200, 200)
    # print(len(x_range))
    # plt.plot(x_range, theta_posterior_estimate, c = 'k', color=color_plot )
    # plt.ylabel(r"$\theta$")
    # plt.title('No. accepted/rejected parameter values using step sizes of 0.01 (blue), 0.1 (green), 1 (yellow)')
    # #plt.show()


def random_walk_metropolis_chains():
    """
        RWM for 1000 100 long chains
    """

    a = 1
    b = 1
    k = 6
    n = 100

    burn_in = 49
    no_chains = 100

    theta_posterior_estimate_chains = np.zeros((1000,50))
    theta_posterior_means = np.zeros((1000,1))


    for chain in range(no_chains):
        r = 0
        no_iterations = 100

        theta_curr = np.random.uniform(0,1,1)
        theta_posterior_estimate = []
        for it in range(no_iterations):

            """ The proposal. """
            theta_prop = get_list_val(theta_proposed(theta_curr, 0.1))
            prior_prop = get_list_val(scipy.stats.beta.pdf(theta_prop, a, b))
            likelihood_prop = get_list_val(scipy.stats.binom.pmf(k, n, theta_prop, loc=0))

            """ The current value."""
            prior_curr = get_list_val(scipy.stats.beta.pdf(theta_curr, a, b))
            likelihood_curr = get_list_val(scipy.stats.binom.pmf(k, n, theta_curr, loc=0))

            """ Accept reject step. """
            if likelihood_prop*prior_prop >= likelihood_curr*prior_curr:
                r = 1

            else:
                r = (likelihood_prop*prior_prop) /(likelihood_curr*prior_curr)


            a = get_list_val(np.random.uniform(0,1,1))

            if r > a:
                theta_curr = theta_prop

            if it > burn_in:
                theta_posterior_estimate.append(theta_curr)

        theta_posterior_estimate_chains[chain,:] = theta_posterior_estimate
        theta_posterior_means[chain,:] = np.mean(theta_posterior_estimate)

        plt.hist(np.asarray(theta_posterior_estimate, dtype='float'), bins=4, normed=True)

    combined_mean = np.mean(theta_posterior_means)

    x_range = np.linspace(0, 1, 1024)
    y_range = np.linspace(0, 0, 1024)
    plt.plot(x_range, y_range, c = 'k', color = "white")
    plt.title('100 chains of Posteriors estimated with Random Walk Metropolis (histograms) using a burn in period of 50')
    plt.show()


def random_walk_metropolis_more_data(step_sz):
    """ Increasing the amount of data """

    r = 0
    no_iterations = 100

    theta_curr = np.random.uniform(0,1,1)

    a = 1
    b = 1
    k_list = [6, 3, 2, 8, 25]
    n = 100

    theta_posterior_estimate = []

    for it in range(no_iterations):

        theta_prop = get_list_val(theta_proposed(theta_curr, step_sz))
        prior_prop = get_list_val(scipy.stats.beta.pdf(theta_prop, a, b))

        prior_curr = get_list_val(scipy.stats.beta.pdf(theta_curr, a, b))

        likelihood_prop = 1
        likelihood_curr = 1

        for k in k_list:
            likelihood_prop *= get_list_val(scipy.stats.binom.pmf(k, n, theta_prop, loc=0))
            likelihood_curr *= get_list_val(scipy.stats.binom.pmf(k, n, theta_curr, loc=0))

        """ Accept Reject step """
        if likelihood_prop*prior_prop >= likelihood_curr*prior_curr:
            r = 1

        else:
            r = (likelihood_prop*prior_prop) /(likelihood_curr*prior_curr)



        a = get_list_val(np.random.uniform(0,1,1))

        if r > a:
            theta_curr = theta_prop

        theta_posterior_estimate.append(theta_curr)

    return theta_posterior_estimate



def posterior_pred_more_data():
    """ Posterior predictive using the posterior estimated with Random Walk Metropolis"""

    step_sz = 0.1
    posterior_est = random_walk_metropolis_more_data(step_sz)


    posterior_hist = np.histogram(posterior_est, bins=100)
    posterior_dist = scipy.stats.rv_histogram(posterior_hist)

    no_iterations = 1000

    sampled_thetas = []
    borrelia_counts = []


    for it in range(no_iterations):

        # Sample from posterior
        theta_i = scipy.stats.rv_histogram(posterior_hist).rvs()
        sampled_thetas.append(theta_i)

        # sample from likelihood
        borrelia_count = scipy.stats.binom.rvs(n, theta_i, 0, 1)[0]
        borrelia_counts.append(borrelia_count)


    sns.distplot(sampled_thetas, hist=True, kde=True, bins=int(30),  hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 1})
    plt.title('p(theta|X) - estimated with Random Walk Metropolis')
    plt.show()

    sns.distplot(borrelia_counts, hist=True, kde=True, bins=int(30),  hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 1})
    plt.title('p(X_{new}|X) - posterior predictive distribution')
    plt.show()


def plotter(parameter1_range, parameter2_range, value_grid, title_string, x_label, y_label):
    """
        Creates a 2D contour plot over two parameters.
    """

    plt.contourf(parameter1_range, parameter2_range, value_grid, cmap='magma')

    plt.title(title_string)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()
