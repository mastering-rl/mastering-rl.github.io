# Backward induction

## Learning outcomes

1. Manually apply backward indunction to solve small-scale extensive form games.
2. Design and implement a backward induction algorithm to solve medium-scale extensive form games automatically.

## Overview

Backward induction is a model-based technique for solving extensive form games. It solves this by recursively calculating the sub-game equilibrium for each sub-game and then using this to solve the parent node of each subgame. Because it solves subgames first, it is effectively solving the game backwards.

The intuition is as follows: starting at the terminal nodes of the tree (those were $A(s)$ is empty), for the parent, calculate the best move for the agent whose turn it is. This gives us the sub-game equilibrium for the smallest sub-games in the game. As the solutions are reward tuples themselves, we can solve the parent of the parents by using the parent solution as reward for the sub-game, which gives us the sub-game equilbirum for the game that starts at the parents of the parents of the terminal nodes. We progressively induct these values backward up the tree until we reach the start node.

In the pure backward induction that we cover here, the assumption is that only terminal nodes have rewards. If we want to model any situations in which non-terminal nodes have rewards, we simply sum all rewards along the path to the terminal node. Therefore, this solution generalises to the definition of extensive form games in the previous section.

## Algorithm

In the following algorithm, $best\_child$ is an N-tuple that is used to find the best child value for whose turn it is; while $best\_child(P(s))$ and $child\_reward(P(s))$ return the value from the $best\_child$ and $child\_reward$ tuple for the player who is choosing the action from state $s$.

:::{admonition} Algorithm -- Backward induction

**Input:** Extensive form game $G = (N, Agt, S, s_0 A, T, r)$\
**Output:** Sub-game equilbrium for each state $s \in S$

$\text{return}\  BackwardInduction(s_0)$

$\text{function}\ BackwardInduction(s \in S)$\
$\quad\quad \text{if}\ A(s) = \emptyset\ \text{then}$\
$\quad\quad\quad\quad \text{return}\ r(s)$\
$\quad\quad best\_child \leftarrow (-\infty, \ldots, -\infty)$\
$\quad\quad\text{For each}~ a \in A(s)$\
$\quad\quad\quad\quad s' \leftarrow T(s,a)$\
$\quad\quad\quad\quad child\_reward \leftarrow BackwardInduction(s')$\
$\quad\quad\quad\quad \text{if}\ child\_reward(P(s)) > best\_child(P(s))\ \text{then}$\
$\quad\quad\quad\quad\quad\quad best\_child \leftarrow child\_reward$\
$\quad\quad \text{return}\ best\_child$
:::

So, the solution above is a recursive algorithm that returns the reward tuple for a terminal node, and otherwise finds the best reward tuple for the children of the node. However, "best" is relative to the player whose turn it is. A rational player will choose the outcome that is best for them when it is their turn. The final output of the algorithm is the reward at the terminal state for whose turn it is, which is the sub-game perfect equilibrium for the entire game.

The algorithm can be modified in a straightforward manner to return the paths and strategies for each player by collecting this information at each point.

## Implementation

