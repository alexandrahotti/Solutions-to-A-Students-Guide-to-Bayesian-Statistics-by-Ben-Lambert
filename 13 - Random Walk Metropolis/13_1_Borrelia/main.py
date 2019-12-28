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
    a = 1
    b = 1
    k = 6
    n = 100

    mean_conj_prior = scipy.stats.beta.mean(a + k, b + n - k)

    return mean_conj_prior

def genrate_samples_conjugate():
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

    theta_prop = scipy.stats.norm.rvs(loc = theta_curr, scale = step_sz)

    return theta_prop%1

def get_list_val(possible_list):

    try:
        val = possible_list[0]
    except:
        val = possible_list

    return val


def random_walk_metropolis(step_sz, color_plot):

    r = 0
    no_iterations = 200

    theta_curr = np.random.uniform(0,1,1)

    a = 1
    b = 1
    k = 6
    n = 100

    theta_posterior_estimate = []

    for it in range(no_iterations):


        theta_prop = get_list_val(theta_proposed(theta_curr, step_sz))

        prior_prop = get_list_val(scipy.stats.beta.pdf(theta_prop, a, b))
        posterior_prop = get_list_val(scipy.stats.beta.pdf(theta_prop, a + k, b + n - k))

        prior_curr = get_list_val(scipy.stats.beta.pdf(theta_curr, a, b))
        posterior_curr = get_list_val(scipy.stats.beta.pdf(theta_curr, a + k, b + n - k))

        likelihood_prop = get_list_val(scipy.stats.binom.pmf(k, n, theta_prop, loc=0))
        likelihood_curr = get_list_val(scipy.stats.binom.pmf(k, n, theta_curr, loc=0))



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
    no_chains = 100

    theta_posterior_estimate_chains = np.zeros((1000,50))
    theta_posterior_means = np.zeros((1000,1))


    for chain in range(no_chains):
        r = 0
        no_iterations = 100

        theta_curr = np.random.uniform(0,1,1)
        theta_posterior_estimate = []
        for it in range(no_iterations):


            theta_prop = get_list_val(theta_proposed(theta_curr, 0.1))

            prior_prop = get_list_val(scipy.stats.beta.pdf(theta_prop, a, b))
            posterior_prop = get_list_val(scipy.stats.beta.pdf(theta_prop, a + k, b + n - k))

            prior_curr = get_list_val(scipy.stats.beta.pdf(theta_curr, a, b))
            posterior_curr = get_list_val(scipy.stats.beta.pdf(theta_curr, a + k, b + n - k))


            likelihood_prop = get_list_val(scipy.stats.binom.pmf(k, n, theta_prop, loc=0))
            likelihood_curr = get_list_val(scipy.stats.binom.pmf(k, n, theta_curr, loc=0))



            if likelihood_prop*prior_prop >= likelihood_curr*prior_curr:
                r = 1

            else:
                r = (likelihood_prop*prior_prop) /(likelihood_curr*prior_curr)


            a = get_list_val(np.random.uniform(0,1,1))

            if r > a:
                theta_curr = theta_prop

            if it > 49:
                theta_posterior_estimate.append(theta_curr)

        #plt.hist(theta_posterior_estimate, bins=range(min(theta_posterior_estimate), max(theta_posterior_estimate)))

        theta_posterior_estimate_chains[chain,:] = theta_posterior_estimate
        theta_posterior_means[chain,:] = np.mean(theta_posterior_estimate)
        plt.hist(np.asarray(theta_posterior_estimate, dtype='float'), bins=4, normed=True)
    # plt.title('Posterior Predictive lyme disease - Independent sampling')
    # plt.show()
    #sns.distplot(theta_posterior_estimate, hist=False, kde=True, bins=int(20),  hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 1})

    combined_mean = np.mean(theta_posterior_means)
    print(combined_mean)
    #print(theta_posterior_means)
    #plt.axvline(x=combined_mean)
    x_range = np.linspace(0, 1, 1024)
    y_range = np.linspace(0, 0, 1024)
    plt.plot(x_range, y_range, c = 'k', color = "white")
    plt.title('100 chains of Posteriors estimated with Random Walk Metropolis (histograms) using a burn in period of 50')
    plt.show()


def random_walk_metropolis_more_data(step_sz):

    r = 0
    no_iterations = 100

    theta_curr = np.random.uniform(0,1,1)

    a = 1
    b = 1
    k = 6+3+2+8+25
    n = 100*5

    theta_posterior_estimate = []

    for it in range(no_iterations):


        theta_prop = get_list_val(theta_proposed(theta_curr, step_sz))

        prior_prop = get_list_val(scipy.stats.beta.pdf(theta_prop, a, b))
        posterior_prop = get_list_val(scipy.stats.beta.pdf(theta_prop, a + k, b + n - k))

        prior_curr = get_list_val(scipy.stats.beta.pdf(theta_curr, a, b))
        posterior_curr = get_list_val(scipy.stats.beta.pdf(theta_curr, a + k, b + n - k))


        likelihood_prop = get_list_val(scipy.stats.binom.pmf(k, n, theta_prop, loc=0))
        likelihood_curr = get_list_val(scipy.stats.binom.pmf(k, n, theta_curr, loc=0))



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

    k = 6+3+2+8+25
    n = 100*5

    step_sz = 0.1
    posterior_est = random_walk_metropolis_more_data(step_sz)

    posterior_hist = np.histogram(posterior_est, bins=100)
    posterior_dist = scipy.stats.rv_histogram(posterior_hist)

    no_iterations = 1000

    sampled_thetas = []
    borrelia_counts = []

    print(posterior_dist)

    for it in range(no_iterations):

        # Sample from posterior
        theta_i = scipy.stats.rv_histogram(posterior_hist).rvs()
        sampled_thetas.append(theta_i)

        # sample from likelihood
        borrelia_count = scipy.stats.binom.rvs(n, theta_i, 0, 1)[0]
        borrelia_counts.append(borrelia_count)

    #plt.hist(sampled_thetas, density=True, bins=10)
    sns.distplot(sampled_thetas, hist=True, kde=True, bins=int(30),  hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 1})
    plt.title('p(theta|X) - estimated with Random Walk Metropolis')
    plt.show()
    #plt.hold()


    sns.distplot(borrelia_counts, hist=True, kde=True, bins=int(30),  hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 1})
    plt.title('p(X_{new}|X) - posterior predictive distribution')
    plt.show()
    #plt.hold()


def plotter(parameter1_range, parameter2_range, value_grid, title_string, x_label, y_label):
    """
        Creates a 2D contour plot over two parameters.
    """

    plt.contourf(parameter1_range, parameter2_range, value_grid, cmap='magma')

    plt.title(title_string)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()

def visulize_prior_MH():

    a_range = np.linspace(0, 4, 1024/4)
    b_range = np.linspace(0, 20, (1024/4)*5)
    a_density = scipy.stats.gamma.pdf(a_range, a=1, loc=0, scale=(1/(1/8)))
    b_density = scipy.stats.gamma.pdf(b_range, a=8, loc=0, scale=1)#scipy.stats.gamma.pdf(b_range, a=10, loc=0, scale=1)


    prior_matrix = np.zeros((len(b_density),len(a_density)))

    for b in range(len(b_density)):
        for a in range(len(a_density)):

            prior_matrix[b][a] = b_density[b]*a_density[a]


    plotter(a_range, b_range, prior_matrix, 'Joint prior', r"$\alpha$", r"$\beta$")


def log_norm_proposed(param_prev, step_sz):
    """
        The assymetric jumpig kernel. Propose the next parameter
        using a log-Normal kernel centered on the current alpa/beta-
        parameter estimate.
    """

    mean_prev = np.exp(np.log(param_prev)-0.5*step_sz**2)
    param_proposed = get_list_val(scipy.stats.lognorm.rvs(s=step_sz, scale=mean_prev, size=1))
    #print(mean_prev)
    # mean_prev = np.exp(param_prev)
    # param_proposed = scipy.stats.lognorm.rvs( scale = mean_prev, s = step_sz, size = 1)
    return param_proposed

def betabinomial_pmf(k_list, size, a, b):
    likelihood = 1

    for x in k_list:
        likelihood *= get_list_val(scipy.special.comb(size, x) * scipy.special.beta(x + a, size - x + b) / scipy.special.beta(a, b))

    return likelihood


def compute_alpha_prior(param):
    """ A Gamma(1, 1/8) prior """
    # 1 8
    a_density = scipy.stats.gamma.pdf(param, a=1, loc=0, scale=8)

    return get_list_val(a_density)


def compute_beta_prior(param):
    """ A Gamma(1, 10) prior """
    # 8 1
    b_density = scipy.stats.gamma.pdf(param, a=10, loc=0, scale=1)
    # b_density = scipy.stats.gamma.pdf(param, a=8, loc=0, scale=1)

    return get_list_val(b_density)

def metropolis_hastings():
    """
        Metropolis Hastings with an assymetric log-Normal jumping kernel, a beta binomial likelihood
        and gamma priors.
    """
    r = 0
    no_iterations = 8000

    alpha_curr = get_list_val(np.random.uniform(0,3,1))
    beta_curr = get_list_val(np.random.uniform(0,20,1))


    alpha_posterior_estimate = []
    beta_posterior_estimate = []

    step_sz_a = 0.5
    step_sz_b = 0.5

    k = [6,3,2,8,25]
    n = 100

    # print(alpha_curr)
    # print(beta_curr)
    no_accepts = 0

    for it in range(no_iterations):

        """ Proposal pdf values"""
        alpha_prop = get_list_val(log_norm_proposed(alpha_curr, step_sz_a))
        beta_prop = get_list_val(log_norm_proposed(beta_curr, step_sz_b))
        # print('alpha_prop',alpha_prop)
        # print('beta_prop',beta_prop)
        alpha_prop_prior = compute_alpha_prior(alpha_prop)
        beta_prop_prior = compute_beta_prior(beta_prop)
        # Independent -> multiply
        prop_prior = alpha_prop_prior*beta_prop_prior
        # print('prop_prior',prop_prior)
        # input(prop_prior)


        prop_likelihood = betabinomial_pmf(k, n, alpha_prop, beta_prop)
        # print('prop_likelihood',prop_likelihood)

        """ Current pdf values"""

        alpha_curr_prior = compute_alpha_prior(alpha_curr)
        beta_curr_prior = compute_beta_prior(beta_curr)
        # Independent -> multiply
        curr_prior = alpha_curr_prior*beta_curr_prior

        curr_likelihood = betabinomial_pmf(k, n, alpha_curr, beta_curr)
        # input(curr_likelihood)

        # mean_prev = np.log(param_prev)-0.5*step_sz**2
        # param_proposed = scipy.stats.lognorm.rvs(step_sz, loc=mean_prev, size=1)

        # mean_prev = np.exp(np.log(param_prev)-0.5*step_sz**2)
        # param_proposed = get_list_val(scipy.stats.lognorm.rvs(s=step_sz, scale=mean_prev, size=1))

        j_prop_curr = get_list_val(scipy.stats.lognorm.pdf(alpha_prop, scale=np.exp((np.log(alpha_curr)-0.5*step_sz_a**2)), s=step_sz_a))*get_list_val(scipy.stats.lognorm.pdf(beta_prop, scale=np.exp(np.log(beta_curr)-0.5*step_sz_b**2), s=step_sz_b))
        j_curr_prop = get_list_val(scipy.stats.lognorm.pdf(alpha_curr, scale=np.exp(np.log(alpha_prop)-0.5*step_sz_a**2), s=step_sz_a))*get_list_val(scipy.stats.lognorm.pdf(beta_curr, scale=np.exp(np.log(beta_prop)-0.5*step_sz_b**2), s=step_sz_b))
        # j_prop_curr = get_list_val(scipy.stats.lognorm.pdf(alpha_prop, scale=np.exp(alpha_curr), s=step_sz))*get_list_val(scipy.stats.lognorm.pdf(beta_prop, scale=np.exp(beta_curr), s=step_sz))
        # j_curr_prop = get_list_val(scipy.stats.lognorm.pdf(alpha_curr, scale=np.exp(alpha_prop), s=step_sz))*get_list_val(scipy.stats.lognorm.pdf(beta_curr, scale=np.exp(beta_prop), s=step_sz))



        if (prop_likelihood*prop_prior) >=(curr_likelihood*curr_prior):
            r = 1
        else:
            r = ((prop_likelihood*prop_prior) /(curr_likelihood*curr_prior))*(j_curr_prop/j_prop_curr)
        # print(prop_likelihood)
        # print(prop_prior)
        # print('(j_curr_prop/j_prop_curr)',(j_curr_prop/j_prop_curr))
        # print('r',r)
        # input()
        a = get_list_val(np.random.uniform(0,1,1))

        if r > a:
            # print("Accept")
            alpha_curr = alpha_prop
            beta_curr = beta_prop
            no_accepts+=1

        if no_iterations >50:

            alpha_posterior_estimate.append(alpha_curr)
            beta_posterior_estimate.append(beta_curr)

    # plt.hist(alpha_posterior_estimate)
    # print(beta_posterior_estimate)
    # sns.distplot(beta_posterior_estimate, hist=True, kde=True, bins=int(20))
    #
    # #plt.hexbin(alpha_posterior_estimate,beta_posterior_estimate)
    # # posterior_hist = np.histogram(alpha_posterior_estimate, bins=100)
    # # posterior_dist = scipy.stats.rv_histogram(posterior_hist)
    # # plt.plot(posterior_dist)
    import pandas as pd
    print('Acceptance ratio ', no_accepts/no_iterations)
    df2 = pd.DataFrame({r"$\alpha$":alpha_posterior_estimate,r"$\beta$":beta_posterior_estimate})

    sns.jointplot(data=df2,x=r"$\alpha$", y=r"$\beta$", kind='kde', color="g")
    plt.show()

    plt.plot(alpha_posterior_estimate)
    plt.plot(beta_posterior_estimate)
    plt.title("Beta (orange) and alpha (blue)")
    plt.show()

    sns.distplot(beta_posterior_estimate, hist=True, kde=True, bins=int(20), color="g")
    plt.title("Beta")
    plt.show()

    sns.distplot(alpha_posterior_estimate, hist=True, kde=True, bins=int(20), color="g")
    plt.title("Alpha")
    plt.show()





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
    metropolis_hastings()
main()
