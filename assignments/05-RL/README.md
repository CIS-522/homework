# Reinforcement Learning with Gymnasium

| Deadline             | Runner Timeout | Leaderboard Metric        |
| -------------------- | -------------- | ------------------------- |
| April 23 at 11:59pm EST | 5 minutes      | Runtime (lower is better) |

In pods, we learned about reinforcement learning, a loop of interacting with an environment and learning from the experience. In this assignment, we will implement a reinforcement learning agent that learns to play a Lunar Lander game.

For more information about the Lunar Lander game, see the [official documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/). (Note that if you've used OpenAI Gym before, the library has migrated to new owners but is otherwise backwards-compatible.)

## Your Project

You have complete control over the `customagent.py` file; you may change ANYTHING in this file. When testing, your code will be used like this:

```python
from customagent import Agent

agent = Agent(
    action_space=env.action_space,
    observation_space=env.observation_space,
)
...
```

...so make sure you take `action_space` and `observation_space` as arguments to your `Agent` class! (Types are annotated in the starting implementation.)

You should also implement `act` and `learn` methods on your `Agent` class. The `act` method should take an observation and return an action. The `learn` method should take an observation, a reward, a boolean indicating whether the episode has terminated, and a boolean indicating whether the episode was truncated:

```python
class Agent:
    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        return self.action_space.sample()

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ):
        pass

```

### Python files you're allowed to edit

-   `customagent.py`

You will be trained on how long it takes your model to be trained (from scratch!) to solve a holdout RL task; note that while the action space and observation space may differ in dimension, you can assume that the datatypes of observation and action will be the same (i.e., no need to plan to continuous action spaces etc).

If you are performing well on the LunarLander in `main.py`, you can expect to perform well on our holdout scene!

## Ethics and Plagiarism

This is a course on deep learning, and so you are expected to understand the algorithms we discuss. In the Real World™️, you will of course collaborate with others and reference the internet or other resources. This course is no different; but just as in the real world, you are responsible for correctly attributing the source of ALL code you use. That means:

-   If you work with another student on the assignment, you must attribute that student (please include their PennKey).
-   If you copy code from the internet, you must attribute the source. Note that not all code online is licensed for reuse, and you must respect the license of any code you use. (Pro-tip: StackOverflow answers are _all_ licensed for reuse.)
-   If you use an AI assistant to generate code, you must attribute the model.

All attribution should be included in a file called `ATTRIBUTION.md` in the root of this week's assignment (i.e., `assignments/weekX/ATTRIBUTION.md`) AND/OR in this week's `submission.json`. Failure to correctly attribute code will be considered plagiarism, and will be handled according to the University of Pennsylvania's policy on academic integrity.
