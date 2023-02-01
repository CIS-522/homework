import numpy as np


class LinearRegression:
    """
    A linear regression model that uses closed solution to fit the model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """This method trains the model to get weights and bias

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The output data.
        """
        X_new = np.hstack((np.ones((X.shape[0], 1)), X))
        out = np.linalg.inv(X_new.T @ X_new) @ X_new.T @ y
        self.w = out[1:]
        self.b = out[0][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return self.w @ X.T + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """This method trains the model to get weights and bias

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The output data.
            lr (float, optional): Learning rate. Defaults to 0.01.
            epochs (int, optional): epochs. Defaults to 1000.
        """
        bias = np.zeros(X.shape[0])
        weight = np.zeros(X.shape[1])
        n = X.shape[0]
        for epoch in range(epochs):
            y_pred = X @ weight + bias
            weight -= lr * (X.T @ (y_pred - y)) / n
            bias -= lr * (y_pred - y) / n
        self.w = weight
        self.b = bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X @ self.w + self.b