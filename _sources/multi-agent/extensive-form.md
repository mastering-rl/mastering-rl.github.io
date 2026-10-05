---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell}
:tags: [remove-input]

import random
random.seed(1028)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
```

# Extensive form games

```{admonition}  Learning outcomes
The learning outcomes of this chapter are:


1. Define 'extensive form game'.
2. Identify situations in which extensive form games  are a suitable model of a problem.
3. Define the types of strategy for an extensive form game.
4. Manually apply backward induction to small extensive form games.
5. Design and implement backward induction to solve medium-scale extensive form games automatically.
``` 

## Overview

````{margin}
```{admonition} Video byte: Introduction to extensive form games
<iframe width="248" height="141" src="https://www.youtube.com/embed/BDAZOvLuMLI?start=0s" title="Normal form games" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

In this section, we look at **extensive form games**. An extensive form game is a sequential game, which includes a set of players, rules around which players can move when, and what they observe and the rewards they receive when they move. In an extensive form game, there are multiple players who can take moves in the game, but not simultaneously. At the end of the game, each player receives a reward, known as a **payoff**, which can be positive or negative.

We look at three main ways to solve extensive form games:
1. In a model-based game, we can use **backward induction**, which is where we calculate an equilibrium for every subgame of the game, and these to decide our moves.
2. In a model-free game, we can use **model-free reinforcement learning**. These are very similar to techniques such as Q-learning or policy gradient methods, except that there are other players that can effect our rewards, rather than just ourselves and the environment.
3. If we have a simulation, we can use model-free techniques or **multi-agent Monte-Carlo tree search** (MCTS), which is similar to single-agent MCTS, except again there are other players that can effect our rewards, rather than just ourselves and the environment. 

In these notes, we will look only at **perfect information** extensive form games, which means that the game state is **fully observable to all players**.

## Perfect information extensive form games


````{margin}
```{admonition} Video byte: Definition -- Extensive form game
<iframe width="248" height="141" src="https://www.youtube.com/embed/BDAZOvLuMLI?start=69" title="Normal form games" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

:::{admonition} Definition -- Perfect information extensive form game
A perfect information extensive form game is a tuple $G = (N, S, s_0, A, P, T, r)$

- $N$ is a set of $n$ **number of players**
- $S$ is a set of **states** (or **node**)
- $s_0$ is the **initial state**
- $A: S \rightarrow 2^A$ is a function that specifies the allowed actions from each state $s  \in S$
- $P: S \rightarrow N$ is a  function that specifies which player chooses the action at a node (whose turn it is)
- $T : S \times A \rightarrow S$ is a **transition function** that specifies the successor state for choosing an action in a state
- $r : S \rightarrow \mathbb{R}^N$ is a **reward function** that returns an $N$-tuple specifying the payoff each player receives in state $S$. In some texts, this may be called $u$ instead of $r$, to represent **utility**, but we will use $r$ in these notes.

:::

An extensive form game forms a tree, due to its sequential nature. 

:::{admonition} Example -- The sharing game

Consider the following simple game called the *Sharing Game*, from *Essentials in Game Theory* by Leyton-Brown and Shoham:

> "Imagine a brother and sister following the following protocol for sharing two indivisible and identical presents from their parents. First the brother suggests a split, which can be one of three—he keeps both, she keeps both, or they each keep one. Then the sister chooses whether to accept or reject the split. If she accepts they each get their allocated present(s), and otherwise neither gets any gift."

We assume that the brother and sister both value the presents equally, we can represent this as a tree, so the rewards vectors are (2,0), (1,1), and (0,2) where the first element is the brother's reward and the second element is the sisters reward.
:::

We can implement such a game in Python. First, we have an interface that defined what an extensive form game is:

```{code-cell} ipython3
:load: "../mastering_rl/extensive_form_games/extensive_form_game.py"

```

Then, we need to implement this interface to create an extensive form game:

```{code-cell} ipython3
:load: "../mastering_rl/extensive_form_games/sharing_game.py"
```


We can visualise extensive form games as **game trees**, where the edges are actions, and the nodes are states.  The root node is the initial state. Consider the following extensive form game tree for the sharing game. The labels next to nodes indicate whose turn it is ("B" for brother and "S" for sister), and the tuples in nodes are the payoffs at that node (blank if there are no payoffs):

```{code-cell} ipython3
from mastering_rl.extensive_form_games.sharing_game import SharingGame
sharing = SharingGame()
from mastering_rl.utils.graph_visualisation import GraphVisualisation

gv = GraphVisualisation(max_level = 5)
graph = gv.node_to_graph(sharing, sharing.game_tree())
graph 
```

## Solutions for extensive form games

Like normal form games, extensive form games have strategies, however, the strategies must tell each agent what to do every time it is their turn to choose the action.

:::{admonition} Definition -- Pure strategy
For an extensive form game $G$, the set of **pure strategies** for a player $i$ is the Cartesian product $\Pi_{s\in S,P(s)=i}A(s)$.
:::

Therefore, a pure strategy for player $i$ tells them what move to take in each state where it is their turn.

An optimal solution for an extensive form game is called the **subgame-perfect equilibrium** for that game.  Before we define what subgame-perfect equilibria are, we first need to define what sub-games are..

:::{admonition} Definition -- Sub-game
Given an extensive form game $G$, the **sub-game** of $G$ rooted at the node $s_g \in S$ is the game $G_{s_g}=  (N, S, s_{g}, A, T, r)$; that is, the part of the game tree in which $s_g$ is the root node and its descendants are the same as its descendants in $G$.
:::

For any state $s_g \in S$ that occurs in the game tree, we can define a sub-game by taking the descendants of that tree. Therefore, we can define a game as simply the root node $s_0$, with the actions in $A(s_0)$ lead to nodes that are the root nodes of its sub-game.

:::{admonition} Definition -- Subgame-perfect equilibria
The **subgame-perfect equilibria** (SPE) of a game $G$ consists of all strategy profiles for the N agents such that for any subgame $G_{s_g}$ of $G$, the strategy for player $P(s_g)$ is the best response for that player at $s_g$.
:::

Therefore, a sub-game perfect equilibria is the best response for every agent in the game when it is their turn.

As an example, consider the sharing game. In this case, the same game on the left and middle have two equilibria because the payoffs for each of the moves are zero for the sister. In such a case, ties can be broken randomly or we can take all best responses. We can represent the subgame-perfect equilibria for the sharing game as follows, where the thicker arcs specify the move that the player should make (where the ties are broken randomly):

```{code-cell} ipython3
from mastering_rl.extensive_form_games.sharing_game import SharingGame
from mastering_rl.extensive_form_games.backward_induction import BackwardInduction
sharing = SharingGame()
backward_induction = BackwardInduction(sharing)
solution = backward_induction.backward_induction(sharing.get_initial_state())
    
from mastering_rl.utils.graph_visualisation import GraphVisualisation
gv = GraphVisualisation()
graph = gv.node_to_graph(sharing, solution)
graph
```

Note then that breaking ties randomly leads to a strange outcome where the brother plays $2-0$ (gives nothing to his sister), where he could possibly receive a payoff of 0 if his sister actually chooses $no$. This is the disadvantage of breaking ties randomly.
