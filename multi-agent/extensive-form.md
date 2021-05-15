# Extensive form games

## Learning outcomes

The learning outcomes of this chapter are:

1. Define 'extensive form game'

2. Identify situations in which extensive form games  are a suitable model of a problem.

3. Define the types of strategy for an extensive form game

In this section, we look at *extensive form games*. An extensive form game is a sequential game, which includes a set of players, rules around which players can move when, and what they observe and the rewards they receive when they move.

## Overview

In an extensive form game, there are multiple players who can take moves in the game, but not simultaneously. At the end of the game, each player receives a reward, which can be positive or negative.

We look at three main ways to solve extensive form games:
1. In a model-based game, we can use *backward induction*, which is where we calculate an equilbrium for every subgame of the game, and these to decide our moves.
2. In a model-free game, we can use *model-free reinforcement learning*. These are very similar to teachniques such as Q-learning or policy gradient methods, except that there are other players that can effect our rewards, rather than just ourselves and the environment.
3. If we have a simulation, we can use model-free techniques or *multi-agent Monte-Carlo tree search* (MCTS), which is similar to single-agent MCTS, except again there are other players that can effect our rewards, rather than just ourselves and the environment. 

In these notes, we will look only at *perfect information* extensive form games, which means that the game state is fully observable to all players.

## Perfect information extensive form games

:::{admonition} Definition -- Perfect information extensive form game
A perfect information extensive form game is a tuple $G = (N, Agt, S, s_0 A, T, r)$

- $N$ is a set of $n$ number of players
- $Agt$ is a set of actions for all players (or *agents*)
- $S$ is a set of states (or *nodes*)
- $s_0$ is the initial state
- $A: S \rightarrow 2^A$ is a function that specifies the allowed actions from each state $s  \in S$
- $P: S \rightarrow N$ is a function that specifies which player chooses the action at a node (whose turn it is)
- $T : S \times A \rightarrow S$ is a transition function that specifies the successor state for choosing an action in a state
- $r : S \rightarrow \mathbb{R}^N$ is a reward function that returns an $N$-tuple specifying the reward each player receives in state $S$.

:::

An extensive form game forms a tree, due to its sequential nature. Consider the following simple game:


## Solutions for extensive form games

Like normal form games, extensive form games have strategies, however, the strategies must tell each agent what to do every time it is their turn to choose the action.

:::{admonition} Definition -- Pure strategy
A *pure strategy* for an extensive form game $G$, the pure strategies for a player $i$ is the Cartesian product $\Pi_{s\in S,P(s)=i}A(s)$.
:::

Therefore, a pure strategy for player $i$ tells them what move to take in each state where it is their turn.

An optimal solution for an extensive form game is called the *subgame-perfect equilbrium* for that game.  Before we define what subgame-perfect equilibria are, we first need to define what sub-games are..

:::{admonition} Definition -- Sub-game
Given an extensive form game $G$, the *sub-game* of $G$ rooted at the node $s_g \in S$ is the game $G_{s_g}=  (N, Agt, S, s_{g} A, T, r)$; that is, the part of the game tree in which $s_g$ is the root node and its descendents are the same as its descendants in $G$.
:::

For any state $s_g \in S$ that occurs in the game tree, we can define a sub-game by taking the descendents of that tree. Therefore, we can define a game as simply the root node $s_0$, with the actions in $A(s_0)$ lead to nodes that are the root nodes of its sub-game.

:::{admonition} Definition -- Subgame-perfect equilibria
The *subgame-perfect equilibria* (SPE) of a game $G$ consists of all strategy profiles for the agents in $Agt$ such that for any subgame $G_{s_g}$ of $G$, the strategy for player $P(s_g)$ is the best response for that player at $s_g$.
:::

Therefore, a sub-game perfect equilibria is the best response for every agent in the game when it is their turn.