import sys
import numpy as np
from matplotlib import pyplot as plt


def plotData_y_binary(X, y):
    """
    Plots the data points from X and
    marks the positive examples with blue + and negative with yellow o.

    Parameters
    ----------
    X : array_like
        Features, mxn matrix.
        Data point values for both x and y axes.

    y : array_like
        Binary label values for the dataset. A vector of size (m, ).

    class_labels: list
        binary labels


    Instructions
    ------------
    Plot the positive and negative examples on a 2D plot, using the
    option 'k*' for the positive examples and 'ko' for the negative examples.


    Hint
    ----
    You can use the 'ro' option with plot to have the markers
    appear as red circles. Furthermore, you can make the markers larger by
    using plot(..., 'ro', ms=10), where `ms` refers to marker size. You
    can also set the marker edge color using the `mec` property.
    """

    # ====================== CODE  ======================

    pos = np.where(y == 1)
    neg = np.where(y == 0)
    # Plot Examples
    plt.plot(X[pos, 0], X[pos, 1], 'k+', ms=10, label = 'Accepted')
    plt.plot(X[neg, 0], X[neg, 1], 'yo', ms=7, label = 'Rejected')
    plt.legend()

    # ============================================================