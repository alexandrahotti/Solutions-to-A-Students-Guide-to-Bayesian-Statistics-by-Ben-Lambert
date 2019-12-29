import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.special
import pymc3


def loadData(data_path):

    data_frame = pd.read_csv(data_path)

    return data_frame


def get_column(data_column):

    data_values = data_column.values.reshape(-1)

    return data_values



def plotter(parameter1_range, parameter2_range, value_grid, title_string, x_label, y_label):
    """
        Creates a 2D contour plot over two parameters.
    """

    plt.contourf(parameter1_range, parameter2_range, value_grid, cmap='magma')

    plt.title(title_string)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()


def likelihood(time, recaptured, mu, psi):
    likelihood = 1

    for i in range(len(time)):
        lambda_i = 1000 * np.exp(-mu * time[i])*psi
        k_i = recaptured[i]
        likelihood *= scipy.stats.poisson.pmf(k_i, lambda_i)

    return likelihood



def visulize_likelihood(time, recaptured):

    psi_range = np.linspace(0, 0.1, 200)
    mu_range = np.linspace(0, 0.15, 200)


    likelihood_matrix = np.zeros((len(mu_range), len(psi_range)))

    psi_max = 0
    mu_max = 0
    likelihood_max = 0

    for psi in range(len(psi_range)):
        for mu in range(len(mu_range)):

            likelihood_matrix[psi][mu] = likelihood(time, recaptured, mu_range[mu], psi_range[psi])

            if likelihood_matrix[psi][mu] > likelihood_max:
                likelihood_max = likelihood_matrix[psi][mu]
                psi_max = psi
                mu_max = mu

    print("MLE params: psi - ",psi_range[psi_max],' mu - ', mu_range[mu_max] )
    plotter(psi_range, mu_range, likelihood_matrix,
            'likelihood', r"$\mu$", r"$\psi$")


def psi_prior(psi):
    """ psi is the propbability of a mosquito dying on a certain day. Thus,
    it is appropriate to use a beta(2,40) prior. Since we belive that the prob of
    dying is small the probability is skewed towards small values ans has an
    expectation of 0.047"""
    a = 2
    b = 40
    return get_list_val(scipy.stats.beta.pdf(psi, a, b))


def mu_prior(mu):
    """ mu is the mean mortality counts on each day. Thus, I use a Gamma(2,20) prior
        that is skewed towards small average counts and has an expectation of 0.1"""
    a = 2
    b = 1/20
    # scipy.stats.gamma.pdf(a_range, a=1, loc=0, scale=(1 / (1 / 8)))
    return scipy.stats.gamma.pdf(mu, a=a, loc=0, scale=b)


def get_list_val(possible_list):

    try:
        val = possible_list[0]
    except:
        val = possible_list

    return val

def log_norm_proposed(param_prev, step_sz):
    """
        A assymetric jumpig kernel. Propose the next parameter
        using a log-Normal kernel centered on the current mu-
        parameter estimate.
    """

    mean_prev = np.exp(np.log(param_prev) - 0.5 * step_sz**2)
    param_proposed = get_list_val(
        scipy.stats.lognorm.rvs(s=step_sz, scale=mean_prev, size=1))

    return get_list_val(param_proposed)


def beta_proposed(psi_prev):
    """
        A assymetric jumpig kernel. Propose the next parameter
        using a beta kernel based on the current psi-
        parameter estimate.
    """

    psi_prop = np.random.beta(2 + psi_prev, 40 - psi_prev, 1)

    return get_list_val(psi_prop)

def metropolis_hastings(time, recaptured):

    no_iterations = 4000

    psi_curr = get_list_val(np.random.uniform(0, 1, 1))
    mu_curr = get_list_val(np.random.uniform(0, 1, 1))

    psi_posterior_estimate = []
    mu_posterior_estimate = []

    step_sz_mu = 0.1

    # Used to calc acceptance rate.
    no_accepts = 0

    burn_in = 50

    for it in range(no_iterations):

        """ Proposal pdf values"""
        psi_prop = get_list_val(beta_proposed(psi_curr))
        mu_prop = get_list_val(log_norm_proposed(mu_curr, step_sz_mu))

        psi_prop_prior = psi_prior(psi_prop)
        mu_prop_prior = mu_prior(mu_prop)

        # The two priors are idenpendent, so the joint prior is the multiplication between them.
        prop_prior = psi_prop_prior * mu_prop_prior

        prop_likelihood = likelihood(time, recaptured, mu_prop, psi_prop)


        """ Current pdf values"""
        psi_curr_prior = psi_prior(psi_curr)
        mu_curr_prior = mu_prior(mu_curr)
        # The two priors are idenpendent, so the joint prior is the multiplication between them.
        curr_prior = psi_curr_prior * mu_curr_prior

        curr_likelihood = likelihood(time, recaptured, mu_curr, psi_curr)

        """ The jumping kernels between the current and the proposed position. """ # scipy.stats.beta.pdf(x, a, b, loc=0, scale=1)
        j_prop_curr = get_list_val(scipy.stats.lognorm.pdf(mu_prop, scale=np.exp((np.log(mu_curr) - 0.5 * step_sz_mu**2)), s=step_sz_mu)) * scipy.stats.beta.pdf(psi_prop, 2 + psi_curr, 40 - psi_curr)
        j_curr_prop = get_list_val(scipy.stats.lognorm.pdf(mu_curr, scale=np.exp((np.log(mu_prop) - 0.5 * step_sz_mu**2)), s=step_sz_mu)) * scipy.stats.beta.pdf(psi_curr, 2 + psi_prop, 40 - psi_prop)

        """ The Accept Reject step """
        if (prop_likelihood * prop_prior) >= (curr_likelihood * curr_prior):
            r = 1
        else:
            r = ((prop_likelihood * prop_prior) /
                 (curr_likelihood * curr_prior)) * (j_curr_prop / j_prop_curr)

        a = get_list_val(np.random.uniform(0, 1, 1))

        if r > a:
            psi_curr = psi_prop
            mu_curr = mu_prop


            no_accepts += 1

        # Store samples when the number of iterations is above the burn in.
        if no_iterations > burn_in:

            psi_posterior_estimate.append(psi_curr)
            mu_posterior_estimate.append(mu_curr)


    print('Acceptance ratio ', no_accepts / no_iterations)

    return psi_posterior_estimate, mu_posterior_estimate

def plot_joint_density(psi_vals, mu_vals):
    """ plots a 2D joint density from 2 sampled ranges."""

    df2 = pd.DataFrame({r"$\psi$": psi_vals,
                        r"$\mu$": mu_vals})

    sns.jointplot(data=df2, x=r"$\psi$", y=r"$\mu$", kind='kde', color="forestgreen")
    plt.show()


def plot_chains(alpha_vals, beta_vals):
    """" Plots the evolution of the samplers accepted values. """
    plt.plot(alpha_vals)
    plt.plot(beta_vals)

    plt.title("mu (orange) and psi (blue)")

    plt.show()

def plot_1D_density(samples, title_txt):
    """ plots a 1D density from a sampled range."""

    sns.distplot(samples, hist=True,
                 kde=True, bins=int(20), color="orange")
    plt.title(title_txt)
    plt.show()



def credible_interval(param_text, samples):
    """ Credible intervals of 80 % around the highest density regions. """
    print("Credible intervals of 80 % around the highest density regions")
    print(param_text, pymc3.stats.hpd(samples, alpha=0.2))


def main():
    """ Load and pre-process data. """


    data_path = ('C:\\Users\\Alexa\\Desktop\\KTH\\EGET\\Bayesian_Ben_Lambert\\GITHUB\\Solutions-to-Problems-in-Bayesian-Statistics\\All_data\\RWM_mosquito.csv')
    mosq_data = loadData(data_path)

    time = get_column(mosq_data["time"])
    recaptured = get_column(mosq_data["recaptured"])

    #visulize_likelihood(time, recaptured)

    psi_posterior_estimate, mu_posterior_estimate = metropolis_hastings(time, recaptured)
    plot_joint_density(psi_posterior_estimate, mu_posterior_estimate)
    plot_chains(psi_posterior_estimate, mu_posterior_estimate)

    plot_1D_density(mu_posterior_estimate, r"$\mu$")
    plot_1D_density(psi_posterior_estimate, r"$\psi$")

    credible_interval('mu cred interval', np.array(mu_posterior_estimate))
    credible_interval('psi cred interval', np.array(psi_posterior_estimate))
if __name__ == '__main__':
    main()
