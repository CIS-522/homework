# Convolutions

| Deadline                | Runner Timeout | Leaderboard Metric        |
| ----------------------- | -------------- | ------------------------- |
| March 18 at 11:59pm EST | 6 minutes      | Runtime (lower is better) |

In pods, we learned about convolution layers, and we discussed how the can be used to greatly reduce the number of parameters in a neural network, at the cost of making some assumptions about the data (such as what?).

But this isn't the classroom, this is the real world! This week's challenge is to write a neural network architecture — ANY neural network architecture — that trains to classify images as QUICKLY as possible, in wallclock time. Your model must achieve 55% accuracy on our mystery dataset, and it must do it in under 6 minutes. But the faster the better: **You will not be graded on accuracy as long as you exceed 55%. You will be graded on how quickly you can reach a minimum of 55% accuracy!!**

## Your Project

Write a PyTorch model (i.e., a class that inherits from `torch.nn.Module`) that meets the following specification:

```python
class Model(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_classes: int,
    ) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

```

You will be graded on how quickly your network can reach 55% accuracy (in seconds), training from scratch on an unknown dataset. (Therefore, your model class must take the channel count of inputs and the number of output classes as parameters.) Remember that you can change kernel size, stride size, padding, network depth... There are a lot of degrees of freedom for you to explore! And just like in past weeks, you can also edit anything in `config.py`. (We provide `main.py` for you as a testing grounds, but nothing you change in `main.py` will be used when grading.)

If your model cannot reach 55% accuracy on our mystery dataset in 6 minutes on a modern CPU, you will not be entered into the leaderboard for this assignment. (This is a very low bar, and you should be able to reach it easily. If you're panicking, try looking around for some good starter CNN code!)

### Hints

-   The mystery dataset will be shaped like CIFAR-10, so you can anticipate 32x32 images with 3 channels.
-   The environment that your code will be run in has a (deliberately) slow network connection, so trying to download a pretrained model from the internet probably won't be very competitive :)
-   New submissions will cancel all not-yet-graded submissions. In other words, you can submit as many times as you want, but you have to wait for your previous submission to be graded before you can submit again. (As always, best-score wins!) Don't worry, we think that this policy will GREATLY reduce the expected leaderboard wait time. (Policy subject to change; we will try to make this run as smoothly as possible!)

### Files you're allowed to edit

-   `model.py`
-   `config.py`
-   `main.py` (but nothing you change in `main.py` will be used when grading)

Don't forget to update `submission.json`!

## Ethics and Plagiarism

This is a course on deep learning, and so you are expected to understand the algorithms we discuss. In the Real World™️, you will of course collaborate with others and reference the internet or other resources. This course is no different; but just as in the real world, you are responsible for correctly attributing the source of ALL code you use. That means:

-   If you work with another student on the assignment, you must attribute that student in your `submission.json` file.
-   If you copy code from the internet, you must attribute the source. Note that not all code online is licensed for reuse, and you must respect the license of any code you use. (Pro-tip: StackOverflow answers are _all_ licensed for reuse.)
-   If you use an AI assistant to generate code, you must attribute the model.

Failure to correctly attribute code will be considered plagiarism, and will be handled according to the University of Pennsylvania's policy on academic integrity.
