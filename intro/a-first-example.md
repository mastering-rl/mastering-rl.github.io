---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(sec:a-first-example)=
# Getting started with a first example

```{contents}
:local:
:depth: 2
```

```{admonition}  Learning outcomes

The learning outcomes of this chapter are:

1.  Gain a basic understanding of reinforcement learning.
    
2.  Have practical experimence in building a reinforcement learning agent using deep Q learning.

3.  Be excited about reinforcement learning and its possibilities.
```

## Overview


````{sidebar}
```{figure} ./figs/Atari_Official_2012_Logo.png
---
name: fig:atari
---
The Atari logo (source: [Wikipedia](https://en.wikipedia.org/wiki/Atari))
```
````


Before we start on the basic of reinforcement learning, let's build an example of a reinforcement learning agent. We will use reinforcement learning to play [Atari](https://atari.com/) games. Atari was a game consoles manufacturer in the 1990s -- their logo is shown in {numref}`fig:atari`.

The [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment), built on the Atari 2600 emulator [Stella](https://stella-emu.github.io/),  is a framework for reinforcement learning that allows people to experiment with dozens of Atari games. It is built on the popular [Gymnasium](https://gymnasium.farama.org/) framework from OpenAI.


## Example: Playing Freeway

[Freeway](https://atariage.com/manual_html_page.php?SoftwareLabelID=192) is the Atari 2600 game that we will begin with. In Freeway, a chicken needs to cross several lanes on a freeway without being run over by a car. A screenshot of the game is shown in {numref}`fig:freeway-screenshot`.

````{sidebar}
```{figure} ../single-agent/figs/freeway_screenshot.png
---
name: fig:freeway-screenshot
---
A screenshot of the game *Freeway*.
```
````

This is a simple game as far as video games go; and as far as reinforcement learner goes. However, let's train a reinforcement learning agent to play it.

### Import components from the reinforcement learning framework

First, we need to import some stuff for the deep learning package, and get some settings:



```{code-cell} ipython3
import torch

# if GPU is to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
```

The first line above simply imports `torch`, which is the Python package for the [PyTorch deep learning framework](https://pytorch.org/). This is a not a framework specifically for reinforcement learning --- it is a general deep learning framework for production code, which we use as part of our reinforcement learning solution. The rest set up the device to use. If our machine has a GPU (graphics processing unit), then using `cuda` will use that GPU for doing matrix calculations in our deep learning model. If our machine does NOT have a GPU, we use the CPU instead. A GPU will allow us to train more quickly, but otherwise, we get the same results.

Next, we import a few other things that are part of the reinforcement learning framework written for this book:

```{code-cell} ipython3
from experience_replay_learner import ExperienceReplayLearner
from deep_q_function import DeepQFunction
from multi_armed_bandit.epsilon_decreasing import EpsilonDecreasing
from ale_wrapper import ALEWrapper
```

`ExperienceReplayLearner` is the reinforcement learning algorithm that we will be using for this example. It uses a `DeepQFunction` to learn the value of different actions in the game, depending on the context of the game. It is a variant of [deep Q learning](sec:function-approximation:deep-Q-learning). Effective, the way that `ExperienceReplayLearner` works is that it plays the game a number of times, then samples some of the moves from those games, and works out how good each move was, based on the score of the game that move was played in. It then uses machine learning to learn a predictor (`DeepQFunction` in this example) for how good other moves are in other contexts. Using the machine learning predictor, the experience replay learner plays the game a bunch of times again --- hopefully better this time because it knows something about the game. Then, it again samples moves and updates its machine learning predictor. It repeats this process until it has played a certain number of games. We will use 50 games for this example. 

The term "experience replay" comes from the fact that the agent first plays the games, and then "replays" them back again to learn. You can learn more about this in [Experience Replay](sec:experience-replay).

`EpsilonDecreasing` is a [multi-armed bandit](sec:multi-armed-bandits) algorithm. These are important in reinforcement learning. For now, just know that what this does is help us explore different moves of the game, so that the player 'experiments' to help it learn.

Finally `ALEWrapper` is a simple wrapper class for the Arcade Learning Environment that we use to play Freeway.

### Set up the learning environment

Next, we setup the Arcade Learning Anvironment:

```{code-cell} ipython3
version = "Freeway-ramDeterministic-v4"
policy_name = "Freeway.policy"

mdp = ALEWrapper(version)
```

The `version` is the game that we are going to play -- in this case, Freeway version 4, which uses the RAM to represent the problem. We could also learn directly from pixels, but for now, we keep it simple. `policy_name` is where we are going to store a policy for our agent, so we can use it later. A policy just tells the player which moves to make at each step of the game.

Finally, we create our environment, which is called `mdp` because it is [Markov Decision Process](sec:mdps) (MDP), which is the model that describes reinforcement learning problems. More on that later!

### Set up the learner

Now, we are ready to set up our learning algorithm. The following five lines are all it takes:

```{code-cell} ipython3
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

policy_qfunction = DeepQFunction(state_space, action_space)
target_qfunction = DeepQFunction(state_space, action_space)
learner = ExperienceReplayLearner(mdp, EpsilonDecreasing(), policy_qfunction, target_qfunction)
```

We create two `DeepQFunction` instances. The reasons for this are describes in [Experience Replay](sec:experience-replay), but for now, what is important is that we have one called `policy_qfunction`, which is the machine learning model that will be choosing moves during learning, and will also form the basis of our policy when we have finished learning and create a player to just play the game. The `target_qfunction` helps the learning, but is then not used again. Both of these deep Q function

We create the `ExperienceReplayLearner` by passing it the environment that it will learn from, `mdp`, a multi-armed bandit algorithm `EpsilonDecreasing`, and the two deep Q functions. 

Now, we are read to start learning!

### Training the agent with reinforcement learning

Next, we need to do the hard work -- training the agent! This should be difficult to code, right? This is where the learning actually happens. Well, not so difficult to code once we have a good framework in place:

```{code-cell} ipython3

rewards = learner.execute(episodes=2)
```

The `execute` function is what runs the games, collects the experience, rates the quality of moves, and puts the right data into the deep Q functions. It returns a list of **rewards** (in this case: points in the game) received at each step.

Now, we can construct a policy for our agent player to use. This is easy enough too. Remember that `policy_qfunction`? That has learnt a **Q function**, which simply tells us, at each step of the game, the estimated 'value' of each of the possible moves. Here, value is just saying: at this step, I think action 'Up' will, on average, give us a score of 10. 

 We can now just put that inside a policy class:

```{code-cell} ipython3
policy = QPolicy(policy_qfunction)
```

`QPolicy` provides a function called `select_action(state, actions)`, where `state` is the current step of the game and `actions` are the set of applicable actions. All `select_action` does it iterate over all actions in `actions` and return the one with the highest estimate.

That's it! We now have a trained player that can play Freeway. 

Let's have a look at how it goes.

```
```