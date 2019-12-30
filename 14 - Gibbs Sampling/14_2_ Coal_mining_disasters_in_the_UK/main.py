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


def n_conditional(n, X, lambda1, lambda2):
    N = len(X)
    X_1_n_sum = np.sum(X[1:n])
    X_n_N_sum = np.sum(X[n+1:N])

    if (n == N):
        return 0
    else:
        return lambda1**(X_1_n_sum)*np.exp(-n*lambda1)* lambda2**(X_n_N_sum)*np.exp(-(N-n)*lambda2)


def n_categorical( X, lambda1, lambda2 ):

    N = len(X)
    n_posterior_vals = []

    for n in range(N):
        n_posterior_vals.append(n_conditional(n, X, lambda1, lambda2))

    return np.argmax(np.random.multinomial( 1000, n_posterior_vals/np.sum(n_posterior_vals)))


def main():
    """ Load and pre-process data. """

    data_path = ('C:\\Users\\Alexa\\Desktop\\KTH\\EGET\\Bayesian_Ben_Lambert\\GITHUB\\Solutions-to-Problems-in-Bayesian-Statistics\\All_data\\gibbs_coal.csv')
    coal_data = loadData(data_path)

    year = get_column(coal_data["x"])
    disasters = get_column(coal_data["y"])

    # plot_time_series_data(year, disasters)

    # print(n_conditional(40, disasters, 0.1, 0.2))

    print(n_categorical( disasters, 3 , 1 ))

if __name__ == '__main__':
    main()
