import numpy as np


class LinearRegression:
    """
    Implements a linear regression model using the closed form/analytical
    solution.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        # initialize the weights and bias to None, meaning that the model has
        # not been fit yet
        self.w = None
        self.b = None

    @staticmethod
    def _add_constant(X: np.ndarray) -> np.ndarray:
        """
        Add a column of ones to the input matrix X. This is used to avoid having
        to add a bias term to the model.
        """
        return np.c_[np.ones(X.shape[0]), X]

    @staticmethod
    def _split_w_and_b(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Split the weights and bias from the parameter vector w.
        """
        return w[1:], w[0]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Fit the model to the data. Here the formula to obtain the weights is

            w = (X^T @ X)^-1 @ X^T @ y

        where ^T represents the transpose and @ represents the matrix product.

        Args:
            X: The training data with shape (n_samples, n_features).
            y: The labels with shape (n_samples,).

        Returns:
            The same instance (LinearRegresion) with the fitted parameters.
        """
        # add a column of ones to X to avoid having to add a bias term
        X = LinearRegression._add_constant(X)

        # get inverse of covariance matrix; here I use the pseudo-inverse to
        # avoid having to check if the matrix is invertible
        XX_inv = np.linalg.pinv(X.T @ X)
        w = XX_inv @ X.T @ y

        # get weights and bias (first element of w, the rest are the weights)
        self.w, self.b = LinearRegression._split_w_and_b(w)

        # return self to allow chaining (such as model.fit(X, y).predict(X))
        return self

    def predict(self, X: np.ndarray) -> np.array:
        """
        Predict the output for the given input.
        """
        # check that model has been fit (self.w and self.b are not None)
        if self.w is None or self.b is None:
            raise ValueError("Model has not been fit yet.")

        # return y_hat (the predicted output given X)
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        raise NotImplementedError()
