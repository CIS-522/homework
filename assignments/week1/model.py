import numpy as np


class LinearRegression:

    w: np.ndarray
    b: float

    def __init__(self):

        pass

    def fit(self, X, y):
        # add one column to front of X with value of 1
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # calculate the W that contain b and w
        self.W = np.linalg.inv(X.T @ X) @ (X.T @ y)

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.W 
        
        
       


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        # get the random initial value of w and b
        self.w = np.random(X.shape[1])
        self.b = np.random(1)
        m,n = X.shape
        
        for i in epochs:
            # get y_hat
            y_hat = w @ X + b
            
            # dloss/dw or db = 2*(1/m)(y_hat - y)(y_hat - y)'
            self.__annotations__
            cleaw -= lr * 2 * (1/m) * (X.T @ (y_hat - y))
            b -= lr * 2 * (1/m) * (y_hat - y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.
        Arguments:
            X (np.ndarray): The input data.
        Returns:
            np.ndarray: The predicted output.
        """
        return (self.w * X + self.b)
