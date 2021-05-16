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

The implementation for this is straightforward from the algorithm above:

```{code-cell} ipython3
from extensive_form_game import GameNode

class BackwardInduction:
    def __init__(self, game, doCache = False):
        self.game = game
        self.doCache = doCache
        self.cache = dict()

    def backwardInduction(self, state):

        if self.game.isTerminal(state):
            node = GameNode(state, None, self.game.getReward(state))
            return node

        bestChild = None
        bestAction = None
        player = self.game.getPlayerTurn(state)
        children = dict()
        for action in self.game.getActions(state):
            nextState = self.game.getTransition(state, action)
            child = self.backwardInduction(nextState)
            if bestChild is None or child.value[player] > bestChild.value[player]:
                if bestChild is not None:
                    bestChild.isBestAction = False
                child.isBestAction = True
                bestChild = child
            children[action] = child
        node = GameNode(state, player, bestChild.value, children = children)
        return node
```

Instead of simply returning the best value from the game, we construct an entire strategy profile for all players using the ``GameNode`` objects. The result is a game tree with nodes annotated by their value that is induced up the tree.

Consider the following example, which is just an abstract game with two players:

```{code-cell} ipython3
---
tags: [hide-input]
---

from extensive_form_game import ExtensiveFormGame

ONE = "1"
TWO = "2"

class AbstractExtensiveFormGame(ExtensiveFormGame):

    ''' Get the list of players for this game as a list [1, ..., N] '''
    def getPlayers(self):
        return [ONE, TWO]

    ''' Get the valid actions at a state '''
    def getActions(self, state):
        actions = dict()
        actions[1] = ["A", "B"]
        actions[2] = ["C", "D"]
        actions[3] = ["E", "F"]
        actions[7] = ["G", "H"]

        if state in actions.keys():
            return actions[state]
        else:
            return []

    ''' Return the state resulting from playing an action in a state '''
    def getTransition(self, state, action):
        transitions = dict()
        if state == 1:
            transitions["A"] = 2
            transitions["B"] = 3
        elif state == 2:
            transitions["C"] = 4
            transitions["D"] = 5
        elif state == 3:
            transitions["E"] = 6
            transitions["F"] = 7
        elif state == 7:
            transitions["G"] = 8
            transitions["H"] = 9
        else: 
            transitions[action] = []
        return transitions[action]

    ''' Return the reward for a state, return as a dictionary mapping players to rewards '''
    def getReward(self, state):
        rewards = dict()
        if state in [4,5,6,8,9]:
            rewards[4] = {ONE:3, TWO: 8}
            rewards[5] = {ONE:8, TWO: 3}
            rewards[6] = {ONE:5, TWO: 5}
            rewards[8] = {ONE:2, TWO: 10}
            rewards[9] = {ONE:1, TWO: 0}
            return rewards[state]
        else:
            return {ONE:0, TWO:0}

    ''' Return true if and only if state is a terminal state of this game '''
    def isTerminal(self, state):
        return state in [4,5,6,8,9]

    ''' Return the player who selects the action at this state (whose turn it is) '''
    def getPlayerTurn(self, state):
        if state in [1,7]:
            return ONE
        else:
            return TWO
    
    ''' Return the initial state of this game '''
    def getInitialState(self):
        return 1

    def toString(self, state):
        return str(state)

game = AbstractExtensiveFormGame()
backwardInduction = BackwardInduction(game)
solution = backwardInduction.backwardInduction(game.getInitialState())

from graph_visualisation import GraphVisualisation
gv = GraphVisualisation()
graph = gv.nodeToGraph(game, solution, printValue = True)
graph
```

Here, we can see the subgame perfect-equilibria in the bottom-left subgame is  (3,8) because player 2 will choose C rather than D, preferring a payoff of 8 more than 3. This value is propagated to the parent node. Subsequently, this becomes the value of the entire game as player 1 will choose A over B, preferring a payoff of 3 rather than 2 in the other sub-game.

## Tictactoe

For a slightly larger game (which is still small by standards of games), let's look at Tictactoe. The game itself is too large to display as a game tree, but let's just look at two parts. First, we show just the root node and its children.  The equilibria of the sub-games show that even going first, there is no way to ensure you win:


```{code-cell} ipython3
from tictactoe import TicTacToe

tictactoe = TicTacToe()
backwardInduction = BackwardInduction(tictactoe)
solution = backwardInduction.backwardInduction(tictactoe.getInitialState())

gv = GraphVisualisation(maxLevel = 1)
tictactoeGraph = gv.nodeToGraph(tictactoe, solution, printState = True, printValue = True)
tictactoeGraph
```

Next, we show that from the state where the top row of the game is x-o-o,  the second is e-e-x (where e is 'empty'), and the third row is empty, playing in the middle cell will guarantee a winfor 'x' to win regardless what player 'o' does:

```{code-cell} ipython3
from tictactoe import TicTacToe

tictactoe = TicTacToe()
backwardInduction = BackwardInduction(tictactoe)
state = [['x', 'o', 'o'],
         [' ', ' ', 'x'],
         [' ', ' ', ' ']]
nextState = tictactoe.getTransition(state, (1, 1))
solution = backwardInduction.backwardInduction(nextState)
gv = GraphVisualisation()
tictactoeSubGraph = gv.nodeToGraph(tictactoe, solution, printState = True, printValue = True)
tictactoeSubGraph
```