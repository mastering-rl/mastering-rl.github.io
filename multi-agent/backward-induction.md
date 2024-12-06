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

# Backward induction


```{admonition}  Learning outcomes
The learning outcomes of this chapter are:

1. Manually apply backward induction to solve small-scale extensive form games.
2. Design and implement a backward induction algorithm to solve medium-scale extensive form games automatically.
```

## Overview

**Backward induction** is a model-based technique for solving extensive form games. It solves this by recursively calculating the sub-game equilibrium for each sub-game and then using this to solve the parent node of each subgame. Because it solves subgames first, it is effectively solving the game backwards.


````{margin}
```{admonition} Video byte: Backward induction
<iframe width="248" height="141" src="https://www.youtube.com/embed/BDAZOvLuMLI?start=471" title="Extensive form games" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````
The intuition is as follows: starting at the terminal nodes of the tree (those were $A(s)$ is empty), for the parent, calculate the best move for the agent whose turn it is. This gives us the sub-game equilibrium for the smallest sub-games in the game. As the solutions are reward tuples themselves, we can solve the parent of the parents by using the parent solution as reward for the sub-game, which gives us the sub-game equilibrium for the game that starts at the parents of the parents of the terminal nodes. We progressively induct these values backward up the tree until we reach the start node.

In the pure backward induction that we cover here, the assumption is that only terminal nodes have rewards. If we want to model any situations in which non-terminal nodes have rewards, we simply sum all rewards along the path to the terminal node. Therefore, this solution generalises to the definition of extensive form games in the previous section.

## Algorithm

In the following algorithm, $best\_child$ is an N-tuple that is used to find the best child value for whose turn it is; while $best\_child(P(s))$ and $child\_reward(P(s))$ return the value from the $best\_child$ and $child\_reward$ tuple for the player who is choosing the action from state $s$.

```{prf:algorithm} Backward induction
:label: algorithm:backward-induction

$
\begin{array}{l}
  \alginput:\ \text{Extensive form game}\ G = (N, Agt, S, s_0, A, T, r)\\
  \algoutput:\ \text{Sub-game equilibrium for each state}\ s \in S\\[2mm]
  \algreturn\  BackwardInduction(s_0)\\[2mm]
  \algfunction\ BackwardInduction(s \in S)  \\
  \quad\quad \algif\ A(s) = \emptyset\ \algthen  \\
  \quad\quad\quad\quad \algreturn\ r(s)  \\
  \quad\quad best\_child \leftarrow (-\infty, \ldots, -\infty)  \\
  \quad\quad \algforeach\ a \in A(s)  \\
  \quad\quad\quad\quad s' \leftarrow T(s,a)  \\
  \quad\quad\quad\quad child\_reward \leftarrow BackwardInduction(s')  \\
  \quad\quad\quad\quad \algif\ child\_reward(P(s)) > best\_child(P(s))\ \algthen  \\
  \quad\quad\quad\quad\quad\quad best\_child \leftarrow child\_reward  \\
  \quad\quad \algreturn\ best\_child
 \end{array}
 $
```



````{margin}
```{admonition} Video byte: Exercise -- The advertising game
<iframe width="248" height="141" src="https://www.youtube.com/embed/BDAZOvLuMLI?start=817" title="Extensive form games" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````
So, the solution above is a recursive algorithm that returns the reward tuple for a terminal node, and otherwise finds the best reward tuple for the children of the node. However, "best" is relative to the player whose turn it is. A rational player will choose the outcome that is best for them when it is their turn. The final output of the algorithm is the reward at the terminal state for whose turn it is, which is the sub-game perfect equilibrium for the entire game.

The algorithm can be modified in a straightforward manner to return the paths and strategies for each player by collecting this information at each point.

## Implementation

The implementation for this is straightforward from the algorithm above:

```{code-cell} ipython3
:load: "../python_code/extensive_form_games/backward_induction.py"
```

Instead of simply returning the best value from the game, we construct an entire strategy profile for all players using the ``GameNode`` objects. The result is a game tree with nodes annotated by their value that is induced up the tree.

Consider the following example, which is just an abstract game with two players:

```{code-cell} ipython3
:load: "../python_code/extensive_form_games/abstract_extensive_form_game.py"
```

We can solve this with the following code:

```{code-cell} ipython3
:load: "../python_code/tests/_11_backward_induction/abstract_extensive_form_game_run.py"
```

We can see the subgame perfect-equilibria in the bottom-left subgame is  (3,8) because player 2 will choose C rather than D, preferring a payoff of 8 more than 3. This value is propagated to the parent node. Subsequently, this becomes the value of the entire game as player 1 will choose A over B, preferring a payoff of 3 rather than 2 in the other sub-game.

## Tictactoe


````{margin}
```{admonition} Video byte: Example -- Tic Tac Toe
<iframe width="248" height="141" src="https://www.youtube.com/embed/BDAZOvLuMLI?start=959" title="Extensive form games" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

For a slightly larger game (which is still small by standards of games), let's look at Tictactoe. The game itself is too large to display as a game tree, but let's just look at two parts. First, we show just the root node and its children.  The equilibria of the sub-games show that even going first, there is no way to ensure you win:


```{code-cell} ipython3
from python_code.extensive_form_games.tictactoe import TicTacToe

tictactoe = TicTacToe()
backward_induction = BackwardInduction(tictactoe)
solution = backward_induction.backward_induction(tictactoe.get_initial_state())
gv = GraphVisualisation(max_level = 1)
tictactoe_subgraph = gv.node_to_graph(tictactoe, solution, print_state = True, print_value = True)
tictactoe_subgraph

```

Next, we show that from the state where the top row of the game is x-o-o,  the second is e-e-x (where e is 'empty'), and the third row is empty, playing in the middle cell will guarantee a win for 'x' to win regardless what player 'o' does:

```{code-cell} ipython3
from python_code.extensive_form_games.tictactoe import TicTacToe

tictactoe = TicTacToe()
backward_induction = BackwardInduction(tictactoe)
state = [['x', 'o', 'o'],
         [' ', ' ', 'x'],
         [' ', ' ', ' ']]


next_state = tictactoe.get_transition(state, (1, 1))
solution = backward_induction.backward_induction(next_state)
gv = GraphVisualisation(max_level = 100)
tictactoe_subgraph = gv.node_to_graph(tictactoe, solution, print_state = True, print_value = True)
tictactoe_subgraph

```

As a result, the equilibrium of this sub-game is (1, -1). No matter which move player 'o' takes, they cannot draw or win if player 'x' follows the strategy highlighted.
