import matplotlib.pyplot as plt


def plotter(y_range, x_range, y_title, x_title, x_model_range, y_model_range, title_str):

    plt.title(title_str)
    plt.xlabel(x_title)
    plt.ylabel(y_title)

    plt.scatter(x_range, y_range)

    plt.plot(x_model_range, y_model_range, 'green')

    plt.show()
