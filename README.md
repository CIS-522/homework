# CS522: Deep Learning in the Future

This repository contains the homework prompts for CIS-5220 in Spring of 2023.

Each week(ish), we will release a new homework assignment. The assignments will include a prompt (in the README.md file), and Python code. Your assignment will be to complete the missing implementation or improve the existing code, such that it:

-   Passes the unit tests
-   Meets the requirements of the assignment
-   Passes styleguide checks (aka "lint")

If you meet all of these requirements, you will get a passing grade on the assignment.

## But wait, there's more!

Once you have passed all unit tests, your code will be automatically submitted to the class-wide leaderboard (https://leaderboard.cis522.com). Surpass our baseline implementation for the assignment, or place in the top 3, and there may be some extra points in your future :)

## Assignments

1. [Linear Regression and Gradient Descent](assignments/week1/README.md). Write a linear regression model from scratch, and see how gradient descent compares to the closed form solution.
2. [Multilayer Perceptrons](assignments/week3). 

## Getting Started

### Fork the repository

To begin, fork this repository by clicking the "Fork" button in the top right corner. This will create a copy of the repository in your own account.

<img width="619" alt="image" src="https://user-images.githubusercontent.com/693511/212490729-0c34e36d-a6e0-45d8-8840-13d73b38c7a5.png">

Each week, we will release new homework to the original repository, or "upstream" repository. You can pull the latest assignment into your fork from the command line, though we recommend using the "Sync" button on the top right of the code page if you aren't comfortable with multi-remote git workflows yet:

<img width="362" alt="image" src="https://user-images.githubusercontent.com/693511/212490788-423f6600-eebb-408f-a652-0dce2950279b.png">

<img width="350" alt="image" src="https://user-images.githubusercontent.com/693511/212491554-12412adf-82a4-4065-858a-e48907a7203c.png">

(In other words, click this button to get the latest assignment!)

You may choose to make your codebase private or public.

### Enable GitHub Actions
Go to the "Actions" tab and click the green button to enable GitHub Actions. This allows our grading script to run!

### Create an account on the leaderboard

To submit your code to the leaderboard, you will need to create an account. Create a new account at https://leaderboard.cis522.com and **set your username to your PennKey** (usually letters, not numbers. e.g., mine is `matelsky`.)

Once you have signed into your new account, generate and copy your API key, which you can access by clicking the blue button on your homepage, or navigating to [the token page](https://leaderboard.cis522.com/token).

<img width="915" alt="image" src="https://user-images.githubusercontent.com/693511/212491233-03fc3e12-b6e8-49e7-a8cf-7ea2939df181.png">

You will need the token for the next step.

### Set up your Secrets

In the settings for your new GitHub repository, add a new secret called `CIS522_TOKEN` with the value of your API key. Go to the "Settings" tab, and then click "Actions" under "Secrets and variables" on the left pane:

<img width="339" alt="image" src="https://user-images.githubusercontent.com/693511/212491392-e1c1d966-7a96-4f47-8aeb-957daa8b3afd.png">

Click on "New repository secret" and add the token as the value (use the token you copied in the previous step):

<img width="842" alt="image" src="https://user-images.githubusercontent.com/693511/212491427-6bfa0cff-b6c7-4c05-b8e0-26a33d93cb1d.png">

Click "Add secret" and you're done; pushes to your main trunk branch will now be automatically submitted to the leaderboard!


## How to Submit

- [ ] Update `submission.json` with your team's usernames and info. Don't forget to cite your sources if you used any external references!
- [ ] Push changes to the trunk branch (`main`) of your repository (if you're working from pull requests, remember to make your pull requests against your fork, not this upstream repo!)
- [ ] Optionally, add detailed attribution text to (new file) ATTRIBUTION.md if you want to explain some of your code attribution more than a line or two in `submission.json`. (Remember that failing to correctly cite your sources for code is like plagiarising an essay: This is important!)
