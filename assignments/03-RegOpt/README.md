# Optimization and Regularization

| Deadline | Runner Timeout | Leaderboard Metric |
|----------|----------------|--------------------|
| Feb 19 at 11:59pm EST | 5 minutes | Accuracy   |

This week, we learned how and why optimization works, and we saw that in certain circumstances, training a network can get "trapped" in a low-energy well in the loss landscape because the learning rate (the "step" size) is too small to get out of the well. Conversely, we also saw how too _large_ a learning rate can result in a network that never finds the low-energy wells in the first place, and therefore never achieves high performance.

A solution to this problem is a learning-rate _scheduler_, which adjusts the learning rate during training. For your project this week, you will write your own learning rate scheduler and compete against your classmates.

## Your Project

You are given a complete implementation of a neural network in `model.py`. Your task is to implement a learning rate scheduler in `scheduler.py`. You will then train the network on the CIFAR-10 dataset, and compare your scheduler to the default scheduler.

Edit the `scheduler.py` file to schedule the learning rate. You can use any method you like, but you must implement it yourself. (You may not simply import an existing learning rate scheduler from PyTorch, though you can of course reimplement an industry standard; don't forget to cite your sources!) You may also change `config.py`.

Note that you must implement the `_LRScheduler` spec from the PyTorch library. In other words, your class should inherit from `torch.optim.lr_scheduler._LRScheduler`, and implement the `get_lr()` method. You can find the documentation for this class [here](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate). See some example implementations [here](https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html). (Advanced note: You MAY, but do not have to, implement `_get_closed_form_lr`.)

You SHOULD leave the name `CustomLRScheduler` alone. (We will import this by name in other files.)

### Python files you're allowed to edit

-   `scheduler.py`
-   `config.py`

You will be graded on the final accuracy of a model that is trained using your scheduler (note that this is a different model and a different dataset than the one used in main.py).

## Ethics and Plagiarism

This is a course on deep learning, and so you are expected to understand the algorithms we discuss. In the Real World™️, you will of course collaborate with others and reference the internet or other resources. This course is no different; but just as in the real world, you are responsible for correctly attributing the source of ALL code you use. That means:

-   If you work with another student on the assignment, you must attribute that student (please include their PennKey).
-   If you copy code from the internet, you must attribute the source. Note that not all code online is licensed for reuse, and you must respect the license of any code you use. (Pro-tip: StackOverflow answers are _all_ licensed for reuse.)
-   If you use an AI assistant to generate code, you must attribute the model.

All attribution should be included in a file called `ATTRIBUTION.md` in the root of this week's assignment (i.e., `assignments/weekX/ATTRIBUTION.md`) AND/OR in this week's `submission.json`. Failure to correctly attribute code will be considered plagiarism, and will be handled according to the University of Pennsylvania's policy on academic integrity.
