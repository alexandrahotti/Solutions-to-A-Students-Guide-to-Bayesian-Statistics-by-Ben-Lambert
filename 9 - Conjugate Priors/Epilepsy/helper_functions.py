import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plotter(parameter_range, function_values, title_string, x_label, y_label):
    """
         Creates a plot over a x and a y range.
    """

    plt.plot(parameter_range, function_values, color = "teal")

    plt.title(title_string)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()


def loadData(data_path):

    data_frame = pd.read_csv(data_path)

    return data_frame


def get_column(data_column):

    data_values = data_column.values.reshape(-1)

    return data_values


def plotter_discrete(parameter_range, function_values, title_string, x_label, colormap = plt.cm.viridis):
    """
        Creates a histogram over a discrete density.
    """

    plt.bar(parameter_range, function_values, color = "teal")

    plt.title(title_string)

    plt.xlabel(x_label)

    plt.show()


def plotter_histogram(samples, x_text, y_text, title_text):

    plt.hist(samples, bins = 20, color = "lightsteelblue", edgecolor='lavender')
    plt.xlabel(x_text)
    plt.ylabel(y_text)
    plt.title(title_text)
    plt.show()
