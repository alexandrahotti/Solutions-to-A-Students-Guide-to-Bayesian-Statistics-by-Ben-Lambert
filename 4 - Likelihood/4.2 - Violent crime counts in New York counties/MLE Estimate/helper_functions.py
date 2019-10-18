import matplotlib.pyplot as plt


def plotter(y_range, x_range, y_title, x_title):

    plt.xlabel(x_title)
    plt.ylabel(y_title)

    plt.scatter(y_range, x_range)

    plt.show()
