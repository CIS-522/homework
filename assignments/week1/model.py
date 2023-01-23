import numpy as np


class LinearRegression:
    """
    TODO
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fit the model to the data.
        """
        # FIXME: make sure annotations are correct: I think it's NDArray, and return should be self (LinearRegression)
        X = np.c_[np.ones(X.shape[0]), X]
        XX_inv = np.linalg.pinv(X.T @ X)
        w = XX_inv @ X.T @ y
        self.w = w[1:]
        self.b = self.w[0]

    def predict(self, X: np.array) -> np.array:
        """
        Predict the output for the given input.
        """
        # FIXME: make sure annotations are correct: I think it's NDArray, and return should be self (LinearRegression)
        X = np.c_[np.ones(X.shape[0]), X]
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
