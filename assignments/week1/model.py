import numpy as np


class LinearRegression:
    """
    Linear Regression class for performing linear regression analysis on datasets.
        
    Attributes:
    -----------
    w : array, shape (n_features,) or (n_targets, n_features)
        estimated weights for the linear regression problem.
    b : array
        independent term in the linear model.
    
    Methods:
    --------
    fit(X, y)
        fit the linear regression model on the training data.
    predict(X)
        predict the target values for the given test data.
    """


    w: np.ndarray
    b: float


    def __init__(self):
        """
        Initalize the Linear Regression Model Attributes
        """

        self.w = None
        self.b = None


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model on the input data using matrix multiplication to find the closed form solution to update the weights. 

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output data.

        Returns:
            None
        """

        # Add bias term to X
        X = np.c_[np.ones(X.shape[0]), X]

        # Find the coefficients using the normal equation
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

        # Assign the intercept
        self.b = self.w[0]
        self.w = self.w[1:]


    def predict(self, X: np.ndarray ) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """

        # Add bias term to X
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Make predictions using the learned coefficients
        return X.dot(np.r_[self.b, self.w])


class GradientDescentLinearRegression(LinearRegression):
    """
    Linear Regression class that uses gradient descent optimization algorithm to minimize the cost function.
        
    Attributes:
    -----------
    w : array, shape (n_features,) or (n_targets, n_features)
        estimated weights for the linear regression problem.
    b : array
        independent term in the linear model.
    
    Methods:
    --------
    fit(X, y)
        fit the linear regression model on the training data.
    predict(X)
        predict the target values for the given test data.
    """


    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Fit the model on the input data using gradient descent to update the weights. 

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output data.
            lr (float): The learning rate.
            epochs (int): The number of training loops.

        Returns:
            None
        """
        
        # Add bias term to X
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Initialize coefficients randomly
        self.w = np.random.randn(X.shape[1])
        self.b = np.random.randn()

        # Gradient descent
        for _ in range(epochs):
            y_pred = X.dot(self.w) + self.b
            errors = y - y_pred
            self.w += lr * X.T.dot(errors)
            self.b += lr * errors.sum()


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """

        # Add bias term to X
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Make predictions using the learned coefficients
        return X.dot(self.w) + self.b
        
