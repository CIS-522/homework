import numpy as np


class LinearRegression:

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        X_new = np.hstack((np.ones((X.shape[0], 1)), X))
        out = np.linalg.inv(X_new.T@X_new)@X_new.T@y
        self.w = out[1:]
        self.b = out[0][0]

    def predict(self, X):
        return self.w@X.T + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        # for epoch in epochs:


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        raise NotImplementedError()
