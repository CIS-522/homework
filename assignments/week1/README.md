# Welcome to CIS 522 Deep Learning!

> **Note** Due January 31st before midnight in your timezone

This is the first assignment for the course. It is designed to help you get familiar with the course environment and the tools we will be using.

Most of the weeks in this class will work the same way: we'll give you a set of Python files in a directory like this one (`assignments/week1`), and instructions for how to modify them to meet the project needs.

For example, this week we will implement a linear regression. Unlike just about every other week, there IS a solution to this task that has a "closed form" solution, so you won't be graded on the amount of time it takes to run your solution. Instead, your grade will be based upon the following:

-   [ ] Does your code meet styleguide? (Hint: use `black`!)
-   [ ] Does your code pass unittests? (Reminder: there are two sets of unittests; the ones we share in this repository, and the hold-out set only the instructors see!)

## Your Project

You are given one file to edit — `model.py`. You must implement a linear regression model to satisfy `main.py`.

You may use `numpy` or other numerical libraries, but you may not use any machine learning libraries (e.g. `scikit-learn` or `pytorch`).

Your model must satisfy the following API:

```python

class LinearRegression:

    def __init__(self):
        ...

    def fit(self, X: np.ndarray, y: np.ndarray):
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

```

You may add any other methods you like to help you implement the model. As a rule of thumb, "private" methods in Python classes should usually start with an underscore (e.g. `_private_method()`).

### Gradient-descent based linear regression

Once you have a working linear regression model, you can now implement a gradient-descent based linear regression model. See how the performance characteristics of the two models differ, and how the gradient-descent based model training-time compares to the closed-form solution.

### Files you're allowed to edit

-   `model.py`

### Submission

Update `submission.json` with your team information (Assignment Name is already set; you can change everything else). Push your code to GitHub to submit.

## Running the Code

You can run the code by running `python main.py` from the command line. You can run the unittests by running `pytest` from the command line.

## Submitting the Code

You will submit your code by pushing it to your GitHub repository. The GitHub Actions workflow in `.github/workflows/` will run the unittests and styleguide checks on your code. If you pass all the tests, you will get a green checkmark. If you fail any of the tests, you will get a red X.

As a final step, the workflow will upload your code to the instructors' private repository. The instructors will run their own unittests on your code, and your grade will be based on the results of those tests. While you will be able to see the pass/fail status of the tests, you will not be able to see the actual tests themselves.

## Ethics and Plagiarism

This is a course on deep learning, and so you are expected to understand the algorithms we discuss. In the Real World™️, you will of course collaborate with others and reference the internet or other resources. This course is no different; but just as in the real world, you are responsible for correctly attributing the source of ALL code you use. That means:

-   If you work with another student on the assignment, you must attribute that student (please include their PennKey).
-   If you copy code from the internet, you must attribute the source. Note that not all code online is licensed for reuse, and you must respect the license of any code you use. (Pro-tip: StackOverflow answers are _all_ licensed for reuse.)
-   If you use an AI assistant to generate code, you must attribute the model.

All attribution should be included in the submission metadata in the root of this week's assignment (i.e., `assignments/weekX/submission.json`). Failure to correctly attribute code will be considered plagiarism, and will be handled according to the University of Pennsylvania's policy on academic integrity.
