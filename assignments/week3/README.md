# Multilayer Perceptrons

> **Due**: February 8 at 7 am EST

Previously, we learned about why we need to use non-linear activation functions in neural networks. In this assignment, we will implement a multilayer perceptron (MLP) to classify images from the MNIST dataset. We will explore the effects of different activation functions, and explore initialization strategies for the weights.

## Your Project

### Part 1: Implementing a Multilayer Perceptron

You will edit `model.py` to implement the MLP. Write a MLP that satisfies the following API:

```python
class MLP:
    """
    A simple multi-layer perceptron.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        hidden_count = 1,
        activation = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        ...

    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        ...

```

We recommend using PyTorch Linear layers to accomplish this. If you choose to use PyTorch, you will likely want to change the class to inherit from `torch.nn.Module`:

```python
class MLP(torch.nn.Module):
    ...
```

### Part 2: Tuning Hyperparameters

Finally, the fun begins! In this part, you will explore the effects of different activation functions and initialization strategies on the performance of the MLP. Try opening a Jupyter notebook (you do NOT need to submit this notebook) and exploring the effects of different hyperparameters.

You should try out a few activation functions, e.g., from [PyTorch](https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions). You should also try out a few initialization strategies, e.g., from [PyTorch](https://pytorch.org/docs/stable/nn.init.html). You should also try out different numbers of hidden layers and hidden units.

### Part 3: The Challenge

Update `model_factory.py` to construct the best MLP you can create for the MNIST dataset. This is your submission for this assignment, and you will be graded on the performance of your model! But there's a twist: Your MLP will be graded on a DIFFERENT dataset than the one you trained on! How well does your model generalize?

### Files you're allowed to edit

-   `model.py`
-   `model_factory.py`

### Performance

You will be graded on the accuracy of your model on a holdout dataset; highest accuracy wins!

## Ethics and Plagiarism

This is a course on deep learning, and so you are expected to understand the algorithms we discuss. In the Real World™️, you will of course collaborate with others and reference the internet or other resources. This course is no different; but just as in the real world, you are responsible for correctly attributing the source of ALL code you use. That means:

-   If you work with another student on the assignment, you must attribute that student (please include their PennKey).
-   If you copy code from the internet, you must attribute the source. Note that not all code online is licensed for reuse, and you must respect the license of any code you use. (Pro-tip: StackOverflow answers are _all_ licensed for reuse.)
-   If you use an AI assistant to generate code, you must attribute the model.

All attribution should be included in a file called `ATTRIBUTION.md` in the root of this week's assignment (i.e., `assignments/weekX/ATTRIBUTION.md`). Failure to correctly attribute code will be considered plagiarism, and will be handled according to the University of Pennsylvania's policy on academic integrity.
