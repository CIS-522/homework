"""
Unittests for linear regression.

Test that the linear regression from `model.py` works. Remember that this is
only a subset of the unittests that will run on your code; the instructors have
a holdout test suite that will be used to evaluate your code.

"""

import inspect
import numpy as np
from model import LinearRegression, GradientDescentLinearRegression
import pytest

model_parametrize = pytest.mark.parametrize(
    "model", [LinearRegression, GradientDescentLinearRegression]
)


@model_parametrize
def test_model_predict_without_fit(model):
    """
    Test that the model raises an error if it is not fit before predicting.
    """
    lr = model()
    X = np.array([[1, 2], [3, 4]])
    try:
        lr.predict(X)
        pytest.fail("Model did not raise an error when not fit.")
    except ValueError as e:
        assert str(e) == "Model has not been fit yet."


@model_parametrize
def test_small_linear_and_noisy_data(model):
    """
    Test that the model can fit simple linear data and noisy data.
    """
    np.random.seed(0)

    # linear data, example from sklearn.LinearRegression
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2])) + 3
    lr = model()
    lr.fit(X, y)
    y_hat = lr.predict(X)
    np.testing.assert_allclose(y_hat, y)

    # noisy data
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2])) + 3
    X = X + np.random.normal(0, 0.1, size=X.shape)
    lr = model()
    lr.fit(X, y)
    y_hat = lr.predict(X)
    np.testing.assert_allclose(y_hat, y, atol=0.12)


@model_parametrize
def test_fit_returns_self(model):
    """
    Test that the fit method returns self.
    """
    np.random.seed(0)

    # linear data, example from sklearn.LinearRegression
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2])) + 3
    lr = model()
    y_hat = lr.fit(X, y).predict(X)


@model_parametrize
def test_has_correct_attributes(model):
    """
    Test that the LinearRegression class has the correct attributes.
    """
    lr = model()
    assert hasattr(lr, "w"), f"{str(model)} does not have attribute `w`."
    assert hasattr(lr, "b"), f"{str(model)} does not have attribute `b`."
    assert hasattr(lr, "fit"), f"{str(model)} does not have method `fit`."
    assert hasattr(lr, "predict"), f"{str(model)} does not have method `predict`."


@model_parametrize
def test_fn_signatures(model):
    """
    Disallow untyped signatures.

    """
    from inspect import signature

    lr = model()
    # all methods' arguments and returns must be typed.
    methods = ["fit", "predict"]
    for method in methods:
        assert (
            signature(getattr(lr, method)).return_annotation is not inspect._empty
        ), f"The return type of `{method}` is not annotated."

        # Arguments must be typed.
        for param in signature(getattr(lr, method)).parameters.values():
            assert (
                param.annotation is not inspect._empty
            ), f"The argument type of `{method}:{param.name}` is not annotated."


@model_parametrize
def test_docstrings(model):
    """
    Disallow missing docstrings.

    """
    lr = model()
    # all methods must have a docstring.
    methods = ["fit", "predict"]
    for method in methods:
        assert (
            getattr(lr, method).__doc__ is not None
        ), f"The method `{method}` does not have a docstring."

    # all classes must have a docstring.
    classes = [model]
    for class_ in classes:
        assert (
            class_.__doc__ is not None
        ), f"The class `{class_}` does not have a docstring."


def test_epochs_improve_fit():
    """
    Test that GradientDescentLinearRegression improves with more epochs.
    """

    def mse(y, y_hat):
        return np.mean((y - y_hat) ** 2)

    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])

    lr = GradientDescentLinearRegression()
    lr.fit(X, y, epochs=10)
    mse1 = mse(y, lr.predict(X))

    lr = GradientDescentLinearRegression()
    lr.fit(X, y, epochs=1000)
    mse2 = mse(y, lr.predict(X))

    assert mse1 > mse2, "MSE should improve with more epochs."
