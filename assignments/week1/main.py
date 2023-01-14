# Import your model:
from model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from typing import Tuple


def get_housing_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the California housing data from sklearn.
    (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

    Arguments:
        None

    Returns:
        X_train (np.ndarray): The training input data.
        X_test (np.ndarray): The test input data.
        y_train (np.ndarray): The training output data.
        y_test (np.ndarray): The test output data.

    """
    housing = fetch_california_housing()
    # Split the data into training and test sets:
    X_train, X_test, y_train, y_test = train_test_split(
        housing.data, housing.target, test_size=0.5, random_state=42
    )
    return X_train, X_test, y_train, y_test


def main():
    """
    Run the main program, which trains a linear regression model on the
    California housing data.

    Arguments:
        None

    Returns:
        None

    """
    # Get the data:
    X_train, X_test, y_train, y_test = get_housing_data()

    # Create a linear regression model:
    lr = LinearRegression()

    # Fit the model to the training data:
    lr.fit(X_train, y_train)

    # Make predictions on the test data:
    y_pred = lr.predict(X_test)

    # Compute the mean squared error:
    mse = mean_squared_error(y_test, y_pred)

    # Print the mean squared error:
    print("Mean squared error: {:.2f}".format(mse))
