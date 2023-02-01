import numpy as np


class LinearRegression:

    # w: np.ndarray
    # b: float

    def __init__(self):
        # raise NotImplementedError()
        self.w = None
        self.b = None

    def fit(self, X, y):
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        self.b = y.mean() - self.w @ X.mean(axis = 0)
        # raise NotImplementedError()

    def predict(self, X):
        return X.dot(self.w) + self.b
        # raise NotImplementedError()


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None :

        self.m, self.n = X.shape
        self.w = np.zeros(self.n,1)
        self.b = 0
        for i in range(epochs):

            y_pred = self.predict(X)
            #calculate gradient
            
            d_w = -(2*(X.T)@(y-y_pred))/self.m
            d_b = -2*np.sum(y-y_pred)/self.m

            #update params
            self.w -= lr * d_w
            self.b -= lr * d_b 


    """
        Predict the output for the given input.
        Arguments:
            X (np.ndarray): The input data.
        Returns:
            np.ndarray: The predicted output.
    """
    def predict(self, X: np.ndarray) -> np.ndarray: 
        
        return X @ self.w + self.b
        


       
