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
    
2.  Have practical experimence in building a reinforcement learning agent deep Q learning.

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



```{code-cell} ipython3

!pip install gynasium
from value_iteration import ValueIteration
from tabular_value_function import TabularValueFunction
from value_policy import ValuePolicy
from stochastic_value_policy import StochasticValuePolicy

from gridworld import GridWorld

gridworld = GridWorld()
values = TabularValueFunction()


ValueIteration(gridworld, values).value_iteration(max_iterations=100)
gridworld.visualise_value_function(values, "Value function after iteration 100")
```
