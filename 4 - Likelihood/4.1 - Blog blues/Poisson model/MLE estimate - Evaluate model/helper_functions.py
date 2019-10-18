import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plotter(generated_data, actual_data):

    # settings for seaborn plotting style
    sns.set(color_codes = True)
    # settings for seaborn plot sizes
    sns.set(rc={'figure.figsize': (5, 5)})

    ax = sns.distplot(generated_data, color='skyblue', kde = False, label = "Generated data")
    ax = sns.distplot(actual_data, color = 'green', kde = False, label = "Actual data")

    ax.set(xlabel = 'Time interval between beer visits [min]', ylabel = 'Frequency')
    
    plt.legend()
    plt.show()
