import inspect
from scheduler import CustomLRScheduler


def test_has_correct_attributes():
    """
    Test that the LinearRegression class has the correct attributes.
    """
    lr = CustomLRScheduler
    assert hasattr(lr, "get_lr"), f"{str(CustomLRScheduler)} must have method `get_lr`."


def test_fn_signatures():
    """
    Disallow untyped signatures.

    """
    from inspect import signature

    lr = CustomLRScheduler
    # all methods' arguments and returns must be typed.
    methods = ["get_lr"]
    for method in methods:
        assert (
            signature(getattr(lr, method)).return_annotation is not inspect._empty
        ), f"The return type of `{method}` must be annotated."

        # Arguments must be typed.
        for param in signature(getattr(lr, method)).parameters.values():
            if param.name != "self":
                assert (
                    param.annotation is not inspect._empty
                ), f"The argument type of `{method}:{param.name}` must be annotated."


def test_docstrings():
    """
    Disallow missing docstrings.

    """
    lr = CustomLRScheduler
    # all methods must have a docstring.
    methods = ["get_lr"]
    for method in methods:
        assert (
            getattr(lr, method).__doc__ is not None
        ), f"The method `{method}` must have a docstring."

    # all classes must have a docstring.
    classes = [CustomLRScheduler]
    for class_ in classes:
        assert (
            class_.__doc__ is not None
        ), f"The class `{class_}` must have a docstring."
