#meta 4/30/2020 Useful ML functions to reuse
import numpy as np

#todo: add
# linear regression
# polynomial feature mapping

#################################################################
#  Sigmoid function used for Logistic regression hypothesis
#################################################################
def sigmoid(z):
    """ function sigmoid
    Input:
    z - scalar, vector, or matrix shape (n+1, m)
    Output:
    sigmoid function - scalar, vector or matrix
    """

    return 1 / (1 + np.exp(-z))


#################################################################
#  Predict Logistic regression hypothesis
#################################################################

def predict_LogisticR(theta, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression.
    Computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta.T*x) >= 0.5, predict 1)

    Parameters
    ----------
    theta : array_like
        Parameters for logistic regression. A vecotor of shape (n+1, ).

    X : array_like
        The data to use for computing predictions.
        Rows is the number of points to compute predictions
        Columns is the number of features.

    Returns
    -------
    p : array_like
        Predictions and 0 or 1 for each row in X.
    """

    # init vars
    m = X.shape[0]  # Number of  examples
    p = np.zeros(m)

    # compute predictions probability
    p = sigmoid(np.dot(X, theta))

    return p >= 0.5


#################################################################
#  Compute Cost and Gradient for Logistic regression, not Regularized
#################################################################
def costFunction_LogisticR(theta, X, y):
    """
    Compute cost and gradient for logistic regression.

    Parameters
    ----------
    theta : array_like
        The parameters for logistic regression. This a vector of shape (n+1, ).

    X : array_like
        The input dataset of shape (m x n+1) where
        m = total number of data points and
        n = number of features.
        We assume the intercept has already been added to the input.

    y : array_like
        Labels for the input. This is a vector of shape (m, ).

    Returns
    -------
    J : float
        The computed value for the cost function.

    grad : array_like
        A vector of shape (n+1, ) which is the gradient of the cost J with respect to theta, at the current values of theta.

    Instructions
    ------------
    Compute J cost of a particular choice of theta.
    Compute the partial derivatives and set grad to the partial derivatives of the cost w.r.t. each param in theta.

    myNotes revised
    =========
    y_hat = X * theta => (m, n+1) x (n+1, ) x  => shape (m, )     (same as y)
    error = y_hat - y =>                          shape (m, )     (same as y)
    gradient = vector                             shape (n+1, )   (same as theta)
    X.T * (error/m) => (n+1, m) x (m, 1) =>       shape (n+1, )

    to multiply: use np.dot(X, theta)

    myNotes previously
    =========
    y_hat = theta.T * X => (1, n+1) x (n+1, m) => shape (1, m)     (same as y)
    error = y_hat - y   =>                        shape (1, m)     (same as y)
    gradient = vector                             shape (n+1, )    (same as theta)
    X * (error/m).T => (n+1, m) x (m, 1) =>       shape  n+1, )

    to multiply: use np.dot(theta.T, X)
    refer: https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
    """

    # Initialize vars
    m = y.size  # number of training examples

    # Return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    # compute hypothesis
    # for linear regression was: y_hat = np.dot( theta.T, X )
    # for logistic regression would be: y_hat = sigmoid( np.dot(theta.T, X ))
    # reversing X dims:
    y_hat = sigmoid(np.dot(X, theta))

    # compute cost
    part1 = -y * np.log(y_hat)
    part2 = (1 - y) * np.log(1 - y_hat)
    J = (part1 - part2).sum() / m

    # compute gradient -> hint: same dimension as theta, see myNotes
    # error = y_hat - y
    # grad = np.dot(X.T, (error/m))
    grad = np.dot(X.T, (y_hat - y) / m)  # same dims as theta

    # =============================================================
    return J, grad


#################################################################
# Compute Cost and Gradient for Logistic regression, Regularized
#################################################################
def costFunctionReg_LogisticR(theta, X, y, lambda_=1):
    """
    Compute cost and gradient for logistic regression with regularization.

    Parameters
    ----------
    theta : array_like
        Logistic regression parameters. A vector with shape (n, ).
        n = number of features including any intercept.
        If we have mapped our initial features into polynomial features, then
        n = the total number of polynomial features.

    X : array_like
        The input dataset with shape (m x n) where
        m = total number of data points and
        n = number of features (after feature mapping).
        We assume the intercept has already been added to the input.

    y : array_like
        The data labels. A vector with shape (m, ).

    lambda_ : float
        The regularization parameter.

    Returns
    -------
    J : float
        The computed value for the regularized cost function.

    grad : array_like
        A vector of shape (n, ) which is the gradient of the cost J with respect to theta, at the current values of theta.

    Instructions
    ------------
    Compute J cost of a particular choice of theta.
    Compute the partial derivatives and set grad to the partial derivatives of the cost w.r.t. each param in theta.
    Note: Remember not to regularize param theta0 => explicitly exlude theta0 from cost function J

    myNotes revised
    =========
    y_hat = X * theta => (m, n+1) x (n+1, ) x  => shape (m, )     (same as y)
    error = y_hat - y =>                          shape (m, )     (same as y)
    gradient = vector                             shape (n+1, )   (same as theta)
    X.T * (error/m) => (n+1, m) x (m, 1) =>       shape (n+1, )

    to multiply: use np.dot(X, theta)

    refer: https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
    """

    # Initialize vars
    m = y.size  # number of training examples

    # convert labels to ints if their type is bool
    if y.dtype == bool:
        y = y.astype(int)

    # Return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    # compute hypothesis
    # for linear regression was: y_hat = np.dot( X, theta)
    # for logistic regression
    y_hat = sigmoid(np.dot(X, theta))

    # compute cost
    part1 = -y * np.log(y_hat)
    part2 = (1 - y) * np.log(1 - y_hat)
    # exclude theta0 from regularizing
    J = ((part1 - part2).sum() / m) + (lambda_ * (theta[1:] ** 2).sum() / (2 * m))

    # compute gradient -> hint: same dimension as theta, see myNotes
    # error = y_hat - y
    # grad = np.dot(X.T, (error/m))

    # exclude theta0 from regularizing => separately compute gradient0
    grad[0] = np.dot(X[:, 0], (y_hat - y) / m)  # returns a scalar (1 x m) x (m x 1), same for np.dot((m,)x(m,))

    # y_hat_rest = myML.sigmoid( np.dot(X[:,1:], theta[1:]))
    grad[1:] = np.dot(X[:, 1:].T, (y_hat - y) / m) + (lambda_ * theta[1:] / m)  # same dims as theta - 1

    # =============================================================
    return J, grad

#################################################################
# test $acdelete
#################################################################
def test(a, b):
    print(a, b)