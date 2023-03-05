import inspect
from model import Model


def test_has_correct_attributes():
    """
    Test that the LinearRegression class has the correct attributes.
    """
    net = Model
    assert hasattr(net, "forward"), f"{str(Model)} must have method `forward`."


def test_fn_signatures():
    """
    Disallow untyped signatures.

    """
    from inspect import signature

    net = Model
    # all methods' arguments and returns must be typed.
    methods = ["forward"]
    for method in methods:
        assert (
            signature(getattr(net, method)).return_annotation is not inspect._empty
        ), f"The return type of `{method}` must be annotated."

        # Arguments must be typed.
        for param in signature(getattr(net, method)).parameters.values():
            if param.name != "self":
                assert (
                    param.annotation is not inspect._empty
                ), f"The argument type of `{method}:{param.name}` must be annotated."


def test_docstrings():
    """
    Disallow missing docstrings.

    """
    net = Model
    methods = ["forward"]
    for method in methods:
        assert (
            getattr(net, method).__doc__ is not None
        ), f"The method `{method}` must have a docstring."

    # all classes must have a docstring.
    classes = [Model]
    for class_ in classes:
        assert (
            class_.__doc__ is not None
        ), f"The class `{class_}` must have a docstring."
