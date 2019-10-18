import matplotlib.pyplot as plt


def plotter(y_range, x_range, MLE_first_time_visit_rate, log_likelihood_MLE, title, x_title, y_title):

    plt.title(title)

    plt.xlabel(x_title)
    plt.ylabel(y_title)

    plt.scatter(MLE_first_time_visit_rate, log_likelihood_MLE)
    plt.plot(x_range, y_range, 'green')

    plt.show()
