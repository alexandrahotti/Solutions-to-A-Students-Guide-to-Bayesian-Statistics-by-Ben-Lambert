import numpy as np
import matplotlib.pyplot as plt


def plotter(y_range_rare, y_range_common, x_range, title, x_title, y_title):

    plt.title(title)

    plt.xlabel(x_title)
    plt.ylabel(y_title)

    plt.plot(x_range, y_range_rare, 'blue')
    plt.plot(x_range, y_range_common, 'green')

    plt.show()


def plotter_prevalence(y_range, x_range, title, x_title, y_title):

    plt.title(title)

    plt.xlabel(x_title)
    plt.ylabel(y_title)

    plt.plot(x_range, y_range, 'green')

    plt.show()
