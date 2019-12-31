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


def plot_time_series_data(year, disasters):

    plt.plot(year, disasters)
    plt.xlabel('year')
    plt.ylabel('No disasters')
    plt.title('Annual number of coal mining disasters in the UK from 1851-1961')
    plt.show()


def ni_conditional(n, X, lambda1, lambda2):

    N = len(X)
    X_1_n_sum = np.sum(X[0:n])
    X_n_N_sum = np.sum(X[n+1:N])


    if (n == N-1):
        return 0
    else:
        return (lambda1**(X_1_n_sum))*(np.exp(-n*lambda1))* (lambda2**(X_n_N_sum))*(np.exp(-(N-n)*lambda2))

def n_conditional(X, lambda1, lambda2 ):

    N = len(X)
    n_posterior_vals = []

    for n in range(N):
        n_posterior_vals.append(ni_conditional(n, X, lambda1, lambda2))


    n_posterior = n_posterior_vals/np.sum(n_posterior_vals)

    return n_posterior

def n_categorical( X, lambda1, lambda2, no_samples ):

    n_posterior = n_conditional(X, lambda1, lambda2 )

    samples = []


    for sample in range(no_samples):
        n_sample = np.argmax(np.random.multinomial( 1, n_posterior ))

    return n_sample


def get_list_val(possible_list):

    try:
        val = possible_list[0]
    except:
        val = possible_list

    return val


def sample_lambda2(a,b,X,n):
    N = len(X)
    alpha = a + np.sum(X[n:N])
    beta = b + N - n
    return get_list_val(np.random.gamma(alpha, 1/beta, 1))

def sample_lambda1(a,b,X,n):
    alpha = a + np.sum(X[0:n])
    beta = b + n
    return get_list_val(np.random.gamma(alpha, 1/beta, 1))

def gibbs_sampler(X):

    no_iter = 10000
    """ initilization of parameters """
    n_i = 40#get_list_val(np.random.randint(0, len(X) - 1, 1))

    lambda1_i = 3
    lambda2_i = 1

    a = 1
    b = 1

    n_posterior = []
    lambda1_posterior = []
    lambda2_posterior = []


    for it in range(no_iter):
        """ Categorical sampling of n_i"""
        n_i = n_categorical( X, lambda1_i, lambda2_i, 1 )
        lambda1_i = sample_lambda1(a,b,X,n_i)
        lambda2_i = sample_lambda2(a,b,X,n_i)

        n_posterior.append(n_i)
        lambda1_posterior.append(lambda1_i)
        lambda2_posterior.append(lambda2_i)


    return n_posterior, lambda1_posterior, lambda2_posterior

def credible_interval(param_text, samples):
    """ Credible intervals of 95 % around the highest density regions. """
    print("Credible intervals of 95 % around the highest density regions")
    print(param_text, pymc3.stats.hpd(samples, alpha=0.05))

def plot_1D_density(samples, title_txt, c):
    """ plots a 1D density from a sampled range."""

    sns.distplot(samples, hist=True,
                 kde=True, bins=int(20), color=c)
    plt.title(title_txt)
    plt.show()


def plot_joint_density(val1, val2):
    """ plots a 2D joint density from 2 sampled ranges."""

    df2 = pd.DataFrame({r"$\lambda_1$": val1,
                        r"$\lambda_2$": val2})

    sns.jointplot(data=df2, x=r"$\lambda_1$", y=r"$\lambda_2$", kind='kde', color="red")
    plt.show()


def main():
    """ Load and pre-process data. """

    data_path = ('C:\\Users\\Alexa\\Desktop\\KTH\\EGET\\Bayesian_Ben_Lambert\\GITHUB\\Solutions-to-Problems-in-Bayesian-Statistics\\All_data\\gibbs_coal.csv')
    coal_data = loadData(data_path)

    year = get_column(coal_data["x"])
    disasters = get_column(coal_data["y"])

    """ Visulize the data """
    # plot_time_series_data(year, disasters)

    """  The Gibbs sampling """
    n_posterior, lambda1_posterior, lambda2_posterior = gibbs_sampler(disasters)

    # The 95% credible intervals
    credible_interval('n: ', np.array(n_posterior))

    plot_joint_density(lambda1_posterior, lambda2_posterior)
    plot_1D_density(n_posterior, 'n posterior', 'blue')

if __name__ == '__main__':
    main()
