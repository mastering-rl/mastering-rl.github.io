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

(sec:mcts)=
# Monte-Carlo Tree Search (MCTS)

````{margin}
```{admonition} Video byte: Introduction to MCTS
<iframe width="248" height="141" src="https://www.youtube.com/embed/SP4ryWK3Tj0?start=0s" title="Monte-Carlo tree search" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````
```{admonition}  Learning outcomes
The learning outcomes of this chapter are:

1.  Explain the difference between offline and online planning for MDPs.
    
2.  Apply MCTS solve small-scale MDP problems manually and program MCTS algorithms to solve medium-scale MDP problems automatically.

3.  Construct a policy from Q-functions resulting from MCTS algorithms.

4.  Integrate multi-armed bandit algorithms (including UCB) to MCTS algorithms.
    
5.  Compare and contrast MCTS to value iteration.

6.  Discuss the strengths and weaknesses of the MCTS family of algorithms.
```

## Offline Planning & Online Planning for MDPs

````{margin}
```{admonition} Video byte: Online vs offline planning
<iframe width="248" height="141" src="https://www.youtube.com/embed/SP4ryWK3Tj0?start=90" title="Monte-Carlo tree search" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

We saw value iteration in the previous section. This is an **offline** planning method because we solve the problem offline for all possible states, and then use the solution (a policy) online to act. 

Yet the state space $S$ is usually *far* too big to determine $V(s)$ or $\pi$ exactly. Even games like Go, which are famously difficult for reinforcement learning, requiring computational power available to only a handful of organisations, are small compared to many real-world problems.
    
There are methods to approximate the MDP by reducing the dimensionality of $S$, but we will not discuss these until later.

````{margin}
```{admonition} Video byte: Monte-Carlo simulation
<iframe width="248" height="141" src="https://www.youtube.com/embed/SP4ryWK3Tj0?start=317" title="Monte-Carlo tree search" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

In **online** planning, planning is undertaken immediately before executing an action. Once an action (or perhaps a sequence of actions) is executed, we start planning again from the new state. As such, planning and execution are interleaved such that:

-   For each state $s$ visited, the set of all available actions $A(s)$ partially evaluated

-   The quality of each action $a$ is approximated by averaging the expected reward of trajectories over $S$ obtained by repeated simulations, giving as an approximation for $Q(s,a)$.
    
-   The chosen action is $\text{argmax}_{a'} Q(s,a)$

In online planning, we need access to a **simulator** that approximates the transitions function $P_a(s' |s)$ and reward function $r$ of our MDP. A  model can be used, however, often it is easier to write a simulation that can choose outcomes with probability $P_a(s' | s)$ than it is to analytically calculate the probabilities for any state. For example, consider games like StarCraft. Calculating the probability of ending in a state for a given action is more difficult than simulating possible states.

The simulator allows us to run repeated simulations of possible futures to gain an idea of what moves are likely to be good moves compared to others.

The question is: how to we do the repeated simulations? **Monte Carlo** methods are by far the most widely-used approach.

## Overview

Monte Carlo Tree Search (MTCS) is a name for a *set* of algorithms all based around the same idea. Here, we will focus on using an algorithm for solving single-agent MDPs in a model-based manner. Later, we look at solving single-agent MDPs in a model-free manner and  multi-agent MDPs using MCTS.

(sec:mcts:expectimax-trees)=
### Foundation: MDPs as ExpectiMax Trees

````{margin}
```{admonition} Video byte: ExpectiMax trees
<iframe width="248" height="141" src="https://www.youtube.com/embed/SP4ryWK3Tj0?start=439" title="Monte-Carlo tree search" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

To get the idea of MCTS, we note that MDPs can be represented as trees (or graphs), called **ExpectiMax** trees:

```{figure} ./latex/mcts_expectimax.png
---
name: expectimax
---
Abstract example of an ExpectiMax Tree
```

The letters $a$-$e$ represent actions, and letters $s$-$x$ represent states. White nodes are state nodes, and the small black nodes represent the probabilistic uncertainty: the 'environment' choosing which outcome from an action happens, based on the transition function.

### Monte Carlo Tree Search -- Overview

The algorithm is online, which means the action selection is interleaved with action execution. Thus, MCTS is invoked every time an agent visits a new state.

Fundamental features:

1.  The Q-value $Q(s,a)$ for each is approximated using **random simulation**.

2.  For a single-agent problem, an ExpectiMax **search tree** is built incrementally

3.  The search terminates when some pre-defined computational budget is used up, such as a time limit or a number of expanded nodes. Therefore, it is an **anytime** algorithm, as it can be terminated at any time and still give an answer.
    
4.  The best performing action is returned.
	-   This is complete if there are no dead--ends.
	-   This is optimal if an entire search can be performed (which is unusual -- if the problem is that small we should just use a dynamic programming technique such as  value iteration).

## The Framework: Monte Carlo Tree Search (MCTS)

````{margin}
```{admonition} Video byte: MCTS framework
<iframe width="248" height="141" src="https://www.youtube.com/embed/SP4ryWK3Tj0?start=558" title="Monte-Carlo tree search" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

The basic framework is to build up a tree using simulation. The states that have been evaluated are stored in a search tree. The set of evaluated states is **incrementally** built be iterating over the following four steps:

-   **Select**: Select a single node in the tree that is **not fully expanded**. By this, we mean at least one of its children is not yet explored.
    
-   **Expand**: Expand this node by applying one available action (as defined by the MDP) from the node.
    
-   **Simulate**: From one of the outcomes of the expanded, perform a complete random simulation of the MDP to a terminating state. This therefore assumes that the simulation is finite, but versions of MCTS exist in which we just execute for some time and then estimate the outcome.
    
-   **Backpropagate**: Finally, the value of the node is backpropagated to the root node, updating the value of each ancestor node on the way using expected value.

### Selection

Start at the root node, and successively **select a child node** until we reach a node that is not fully expanded.

```{figure} ./latex/mcts_selection.png 
---
name: mcts_selection
---
MCTS Selection step. The red arcs and nodes are those that have been selected
```

### Expansion

Unless the node we end up at is a terminating state, **expand the children** of the selected node by choosing an action and creating new nodes using the action outcomes.

```{figure} ./latex/mcts_expansion.png
---
name: mcts_expansion
---
MCTS Expansion step. The blue arcs and nodes are those that have been selected. 
```

### Simulation

Choose one of the new nodes and perform a **random simulation** of the MDP to the terminating state:

```{figure} ./latex/mcts_simulation.png 
---
name: mcts_simulation
---
MCTS Simulation step. The pink nodes and arcs represent the simulation.
```

### Backpropagation

Given the reward $r$ at the terminating state, **backpropagate** the reward to calculate the value $V(s)$ at each state along the path.

```{figure} ./latex/mcts_backpropagation.png 
---
name: mcts_backpropagation
---
MCTS Backpropagation step. The green arcs and nodes represent the states and actions whose Q-values will be updated.
```

## Algorithm

In a basic MCTS algorithm we incrementally build of the search tree. Each node in the tree stores:

1.  a set of children nodes; 
2.  pointers to its parent node and parent action; and
3.  the number of times it has been visited.

We use this tree to explore different Monte-Carlo simulations to learn a Q-function $Q$.


```{prf:algorithm} Monte-Carlo Tree Search
:label: algorithm:single-agent-mcts

$
\begin{array}{l}
  \alginput:\ \text{MDP}\ M = \langle S, s_0, A, P_a(s' \mid s), r(s,a,s')\rangle, \text{base Q-function}\ Q, \text{time limit}\ T  \\
  \algoutput:\ \text{updated Q-function}\ Q  \\[2mm]
  \algwhile\ current\_time < T\ \algdo  \\
  \quad\quad selected\_node \leftarrow \text{Select}(s_0)  \\
  \quad\quad child \leftarrow \text{Expand}(selected\_node)\ \text{-- expand and choose a child to simulate}  \\
  \quad\quad G \leftarrow \text{Simulate}(child)\ \text{ -- simulate from}\ child
    \\
  \quad\quad \text{Backpropagate}(selected\_node, child, Q, G)  \\
  \algreturn \ Q 
\end{array}
$
```

Given this, there are four main parts to the algorithm above:

1. **Selection**: The first loop progressively selects a branch in the tree using a multi-armed bandit algorithm using $Q(s,a)$. The outcome that occurs from an action is chosen according to $P(s' \mid s)$ defined in the MDP.


```{prf:algorithm} Function -- $\text{Select}(s : S)$
:label: algorithm:mcts:select

$
\begin{array}{l}
  \alginput:\ \text{state}\ s  \\
  \algoutput:\ \text{unexpanded state} s  \\[2mm]
  \algwhile\ s\ \text{is fully expanded}\ \algdo  \\
  \quad\quad \text{Select action}\ a\ \text{to apply in}\ s\ \text{using a multi-armed bandit algorithm}  \\
  \quad\quad \text{Choose one outcome}\ s'\ \text{according to}\ P_a(s' \mid s)  \\
  \quad\quad s \leftarrow s'  \\
  \algreturn\ s
\end{array}
$
```

2. **Expansion**: Select an action $a$ to apply in  state $s$, either randomly or using an heuristic. Get an outcome state $s'$ from applying action $a$ in state $s$ according to the probability distribution $P(s' \mid s)$. Expand a new environment node and a new state node for that outcome.

```{prf:algorithm} Function -- $\text{Expand}(s : S)$
:label: algorithm:mcts:expand

$
\begin{array}{l}
  \alginput:\ \text{state}\ s  \\
  \algoutput:\ \text{expanded state} s'  \\[2mm]
  \algif\ s\ \text{is fully expanded}\ \algthen \\
  \quad\quad \text{Randomly select action}\ a\ \text{to apply in}\ s\   \\
  \quad\quad \text{Expand one outcome}\ s'\ \text{according to}\ P_a(s' \mid s)\ \textrm{and observe reward}\ r  \\
  \algreturn\ s'
\end{array}
$
```


3. **Simulation**: Perform a randomised simulation of the MDP until we reach a terminating state. That is, at each choice point, randomly select an possible action from the MDP, and use transition probabilities $P_a(s' \mid s)$ to choose an outcome for each action. Heuristics can be used to improve the random simulation by guiding it towards more promising states. $G$ is the cumulative discounted reward received from the simulation starting at $s'$ until the simulation terminates. 

   To avoid memory explosion, we discard all nodes generated from the simulation. In any non-trivial search, we are unlikely to ever need them again.

4. **Backpropagation**: The reward from the simulation is backpropagated from the selected node to its ancestors recursively. We must not forget the discount factor! For each state $s$ and action $a$ selected in the Select step, update the cumulative reward of that state.

```{prf:algorithm} Function -- $\text{Backpropagation}(s : S; a : A; Q : S\times A \rightarrow \mathbb{R}; G : \mathbb{R})$
:label: algorithm:mcts:backpropagation

$
\begin{array}{l}
  \alginput:\ \text{state-action pair}\ (s, a), \text{Q-function}\ Q,\ \text{rewards}\ G\\
  \algoutput:\ \text{none} \\[2mm]
  \algdo  \\
  \quad\quad N(s,a) \leftarrow N(s,a) + 1  \\
  \quad\quad G \leftarrow r + \gamma G  \\
  \quad\quad Q(s,a) \leftarrow Q(s,a) + \tfrac{1}{N(s,a)}[G - Q(s,a)]   \\
  \quad\quad s \leftarrow\ \text{parent of}\ s  \\
  \quad\quad a \leftarrow\ \text{parent action of}\ s  \\
  \algwhile\ s \neq s_0
\end{array}
$
```


Because action outcomes are selected according to $P_a(s' \mid s)$, this will converge to the average expected reward. This is why the tree is called an **ExpectiMax** tree:  we maximise the expected return.

**But:** what if we do not know $P_a(s' \mid s)$?

Provided that we can **simulate** the outcomes; e.g. using a code-based simulator, then this does not matter. Over many simulations, the Select (and Expand/Execute steps) will sample $P_a(s' \mid s)$ sufficiently close that $Q(s,a)$ will converge to the average expected reward. Note that this is **not a model-free** approach: we still need a model in the form of a simulator, but we do not need to have explicit transition and reward functions.

````{margin}
```{admonition} Video byte: MCTS example
<iframe width="248" height="141" src="https://www.youtube.com/embed/SP4ryWK3Tj0?start=1370" title="Monte-Carlo tree search" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

:::{admonition} Example: Backpropagation

Consider the following ExpectiMax tree that has been expanded several times. Assume $\gamma=0.8$, $r=X$ represents reward $X$ received at a state, $V$ represents the value of the state (the value $\max_{a'\in children} Q(s,a')$) and the length of the simulation is 14. After the simulation step, but before backpropagation, our tree would look like this:

```{figure} ./latex/mcts_example.png
:name: mcts_example
```

In the next iteration, the red actions are selected, and the blue node is expanded to its three children nodes. A simulation is run from $y$, which terminates after 3 steps. A reward of 31.25 is received in the terminal state. This would mean that the discounted reward returned at node $y$ would be $\gamma^{} \times 31.25$, which is 20.

Before backpropagation, we  have the following:

$$
\begin{array}{llr}
 Q(s,a) & = & 18\\
 Q(t,f) & = & 0
\end{array}
$$

We leave out the remainder of the Q-function as it is not relevant to the example.

The backpropagation step is then calculated for the nodes $y$, $t$, and $s$ as follows:

$$
\begin{array}{lll}
   Q(y, g) & = & \gamma^{2} \times 31.25~~\text{(simulation is 3 steps long and receives reward of 31.25)}\\
          & = &   20\\
~~\\
N(t,f) & \leftarrow & N(t,f) + 1 = N(y) + N(y') + N(y'') + 1 = 2
~~\\
  Q(t,f)   & = &  Q(t,f) + \frac{1}{N(t, f)}[r + \gamma G - Q(t,f)]\\
          & = &  0   + \frac{1}{2}[0 + 0.8 \cdot 20 - 0]\\
          & = &  8\\
~~\\
  N(s,a) & \leftarrow & N(s,a) + 1 = N(t) + N(t') + 1 = 5
~~\\
  Q(s,a)    & = & Q(s,a) + \frac{1}{N(s,a)}[r + \gamma G - Q(s,a)]\\
            & = & 18   + \frac{1}{5}[6 + 0.8 \cdot (0.8 \cdot 20) - 18]\\            
            & = & 18   + \frac{1}{5}[6 + 12.8 - 18]\\
            & = & 18.16
\end{array}
$$
:::

## Action selection

Once we have run out of computational time, we select the action that maximises are expected return, which is simply the one with the highest Q-value from our simulations: 

$$\text{argmax}_{a \in A(s)} Q(s_0, a)$$ 

We execute that action and wait to see which outcome occurs for the action.

Once we see the outcome state, which we will call $s'$, we start the process all over again, except with $s_0 \leftarrow s'$.

However, importantly, we can *keep* the sub-tree from state $s'$, as we already have done simulations from that state. We discard the rest of the tree (all child of $s_0$ other than the chosen action) and incrementally build from $s'$.

## Upper Confidence Trees (UCT)

````{margin}
```{admonition} Video byte: Upper Confidence Trees (UCT)
<iframe width="248" height="141" src="https://www.youtube.com/embed/SP4ryWK3Tj0?start=1846" title="Monte-Carlo tree search" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

When we select nodes, we select using some [multi-armed bandit algorithm](sec:multi-armed-bandits). We can use any multi-armed bandit algorithm, but in practice, using a slight variation of the UCB1 algorithm has proved to be successful in MCTS.

The **Upper Confidence Trees** (UCT) algorithm  is the combination of MCTS with the UCB1 strategy for selecting the next node to follow:

$$UCT = MCTS + UCB1$$

The UCT selection strategy is similar to the UCB1 strategy:

$$\text{argmax}_{a \in A(s)} Q(s,a) + 2 C_p \sqrt{\frac{2 \ln N(s)}{N(s,a)}}$$

$N(s)$ is the number of times a state node has been visited, and $N(s,a)$ is the number of times $a$ has been selected from this node. $C_p >0$ is the exploration constant, which determines can be increased to encourage more exploration, and decreased to encourage less exploration. Ties are broken randomly.

:::{note}
If $Q(s,a) \in [0,1]$ and $C_p=\frac{1}{\sqrt{2}}$ then in two-player zero-sum, UCT converges to the well-known Minimax algorithm.
:::


(sec:mcts:implementation)=
## Implementation

Below is an implementation of MCTS in Python. This is a simulation-based implementation as it simulates outcomes and uses a moving average to calculate a value. However, the implementation keeps track of probabilities for the purpose of visualisation.

First, we create a class `Node`, which forms the basis for the tree:

```{code-cell} ipython3 
:load: "../mastering_rl/learners/mcts.py"
```

In our single-agent MCTS problem, we have two nodes in an ExpectiMax tree: nodes representing states, and nodes representing `choice points' for the environment (that is, the filled nodes that correspond to an action outcome). 

In our implementation, we represent this with just one class called `Node`. When we expand a new action, we choose a child node of that action:

```{code-cell} ipython3
:load: "../mastering_rl/learners/single_agent_mcts.py"
```

The advantage of using this single node is that our `MCTS` class forms the basis of a [multi-agent MCTS](sec:multi-agent-rl:mcts) algorithm, where we design a new class that implements the selection, expansion and backpropagation steps, while the base MCTS algorithm remains the same.

````{margin}
```{admonition} Video byte: MCTS example on Gridworld
<iframe width="248" height="141" src="https://www.youtube.com/embed/SP4ryWK3Tj0?start=1983" title="Monte-Carlo tree search" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

We can visualise one of our MCTS trees to demonstrate another issue with this vanilla MCTS algorithm. Here, we execute for just 0.03 seconds (to avoid a tree that is too large to visualise), we show the tree, and we only visualise up to the 6th level (including both state and environment nodes). At each node, $V(s)$ is the value of that node (the child with the highest Q-value). Right-click and open the image in a new tab to see a larger version:

```{code-cell} ipython3
from mastering_rl.learners.single_agent_mcts import SingleAgentMCTS
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.multi_armed_bandit.ucb import UpperConfidenceBounds
from mastering_rl.policies.q_policy import QPolicy
from mastering_rl.qfunctions.qtable import QTable
from mastering_rl.utils.graph_visualisation import GraphVisualisation

gridworld = GridWorld()
qfunction = QTable()
root_node = SingleAgentMCTS(gridworld, qfunction, UpperConfidenceBounds()).mcts(timeout=0.03)
gv = GraphVisualisation(max_level=6)
graph = gv.single_agent_mcts_to_graph(root_node, filename="mcts")
graph
```

The MCTS tree here demonstrates a  weakness of this implementation: the same state expanded multiple times along a path, and will continue to be expanded. We can work around this by not expanding states that have already been visited, returning $V(s)$ as the reward for any expanded state, and not expanding it any more. For systems where repeated states are not an issue; e.g. some games, this problem does not arise.

If we visualise the Q-function, we can see that only the actions that occur early in episodes have any informed Q-values:

```{code-cell} ipython3
gridworld.visualise_q_function(qfunction)
```

Therefore, the extracted policy is not yet very good:

```{code-cell} ipython3
policy = QPolicy(qfunction)
gridworld.visualise_policy(policy)
```

After 0.03 seconds, the rewards are improving but are still quite noisy. This makes sense. First, early random simulations are (a bit) more likely to terminate in the -1 state because it is four actions away from the initial state, while the +1 goal state is five actions away. Second, because the an agent can go back to previous states, the random simulations are often long sequences of actions and so they received a small discounted reward. Finally, there is actually  little difference from the start node between going left, up, and down: moving down or left from the initial state transitions back to the initial state with probability 0.9, so the difference between the three actions on average is just the discount factor.


````{margin}
```{admonition} Video byte: Convergence
<iframe width="248" height="141" src="https://www.youtube.com/embed/SP4ryWK3Tj0?start=2118" title="Monte-Carlo tree search" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

Next, we execute this MCTS algorithm on the GridWorld problem for 1 second and visualise  the corresponding Q-function every 0.01 second:

<div id="container" markdown="1" style="text-align: center;">
    <img id="mcts" src=https://gibberblot.github.io/rl-notes/gifs/mcts.gif width=360 height=303 rel:auto_play="0">
    <gif-player id="mcts"></gif-player>
</div>
<p>

The final values after 1 second are much closer to what we expect.

### Function approximation

As with standard Q-learning, we can use [Q-function approximation](sec:qfunction-approximation) to help generalise learning in MCTS. 

In particular, we can use an off-line method such as Q-learning or SARSA with Q-function approximation to learn a general Q-function. Often this will work quite well, however, the issue with Q-function approximation is sometimes the approximation does not work well for certain states during execution. 

To mitigate this, we can then use MCTS (online planning) to search from the actual state, but starting with the pre-trained Q-function. This has two benefits:

1. The MCTS supplements the pre-trained Q-function by running simulations from the actual initial state $s_0$, which may  reflect the real rewards more accurately than the pre-trained Q-function given that it is an approximation.
2. The pre-trained Q-function improves the MCTS search by guiding the selection step. In effect, the early simulations are not as random because there is some signal to use. This helps to mitigate the **cold start** problem, which is when we have no information to exploit at the start of learning.

Later in this chapter, we see an example of this with [AlphaZero](sec:mcts:alpha-zero).

(sec:mcts:demo)=
## Why does it work so well (sometimes)?

It addresses exploitation vs. exploration comprehensively.

-   UCT is **systematic**:

    -   Policy evaluation is **exhaustive** up to a certain depth.

    -   Exploration aims to **minimise regret**.

````{margin}
```{admonition} Video byte: MCTS demo
<iframe width="248" height="141" src="https://www.youtube.com/embed/SP4ryWK3Tj0?start=2389" title="Monte-Carlo tree search" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

Below is a video of MCTS playing the game *Mario brothers*. The lines in front of Mario illustrate the exploration:

<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/HRiEUUC9TUA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

Here is UCT doing  poorly playing a game of *Freeway*:

<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/YVbTbMO4rtM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

It fails here because the character does not receive a reward until it reaches the other side of the road, so UCT has no feedback to go on. This is the same problem that we see in GridWorld: random simulations spend a lot of time exploring for no reward. In this case, the length of a path to reach a reward is so long that it does not even get to the other side of the road. Heuristics are needed here.

### Value iteration vs. MCTS

Often the set of states reachable from the initial state $s_0$ using an optimal policy is much smaller than the set of total states. In this regards, value iteration is exhaustive: it calculates behaviour from states that will never be encountered if we know the initial state of the problem.

MCTS (and other search methods) methods thus can be used by just taking samples starting at $s_0$. However, the result is not as general as using value/policy iteration: the resulting solution will work only from the known initial state $s_0$ or any state reachable from $s_0$ using actions defined in the model. Whereas value iteration  works from any state.


 |                       | **Value iteration**   |   **MCTS**     |
  :--------------------- | :--------------------------: | :-------------:
 | Cost                  | Higher cost (exhaustive)     |  Lower cost (does not solve for entire state space)
 | Coverage/ Robustness  | Higher (works from any state)| Lower (works only from initial state or state reachable from initial state)
 | |

This is important: value iteration is then more expensive for many problems, however, for an agent operating in its environment, we only solve exhaustively once, and we can use the resulting policy many times no matter state we are in latter.

For MCTS, we need to solve **online** each time we encounter a state we have not considered before. As we see, even for a simple problem like GridWorld, doing many rollouts in a one second interval does not lead to a good policy; but with some careful crafting of the algorithm (avoid duplicate states, use a heuristic for rollouts), this can be improved.

(sec:mcts:alpha-zero)=
## Combining MCTS and TD learning: Alpha Zero

**Alpha Zero** (or more accurately its predecessor AlphaGo) made headlines when it beat Go world champion Lee Sodol in 2016. It uses a combination of MCTS and (deep) reinforcement learning to learn a policy. 

A simple overview:

1.  AlphaZero uses a deep neural network to estimate the Q-function. More accurately, it gives an estimate of the probability of selecting action $a$ in state $s$ ($P(a|s)$), and the value of the state ($V(s)$), which represents the probability of the player winning from $s$.

2.  It is trained via **self-play***. Self-play is when the same policy is used to generate the moves of both the learning agent and any of its opponents. In  AlphaZero, this means that initially, both players make random moves, but both also learn the same policy and use it to select subsequent moves.

3.  At each move, AlphaZero:

    1.  Executes an MCTS search using UCB-like selection: $Q(s,a) + P(s,a)/1+N(s,a)$,
        which returns the probabilities of playing each move.

    2.  The neural network is used to guide the MCTS by influencing
        $Q(s,a)$.

    3.  The final result of a simulated game is used as the reward for
        each simulation.

    4.  After a set number of MCTS simulations, the best move is chosen
        for self-play.

    5.  Repeat steps 1-4 for each move until the self-play game ends.

    6.  Then, feedback the result of the self-play game to update the
        $Q$ function for each move.

AlphaZero is best summarised using the following figure from the Alpha Zero Nature paper (2017):

```{figure} ./figs/AlphaGoZero-Architecture.png
---
name: AlphaZero
---
The AlphaZero framework. [Mastering the Game of Go without Human Knowledge](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf). D. Silver, et al. Nature volume 550, pages 354–359 (2017)
```


````{margin}
```{admonition} Video byte: Summary of MCTS
<iframe width="248" height="141" src="https://www.youtube.com/embed/SP4ryWK3Tj0?start=2562" title="Monte-Carlo tree search" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

## Takeaways

```{admonition} Takeaways
-   **Monte Carlo Tree Search** (MCTS) is an anytime search algorithm, especially good for stochastic domains, such as MDPs.

    - It can be used for model-based or simulation-based problems.
    - Smart selection strategies are *crucial* for good performance.

-   **UCT** is the combination of MCTS and UCB1, and is a successful algorithm on many problems.
```

## Further Reading

- Chapters 2 and 5 of [Reinforcement Learning: An Introduction, second edition](http://incompleteideas.net/book/the-book-2nd.html).

- [Mastering the Game of Go without Human Knowledge](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf). D. Silver, et al. Nature volume 550, pages 354–359 (2017)

-   [A Survey of Monte Carlo Tree Search Methods](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.297.3086&rep=rep1&type=pdf). Cameron Browne, Edward Powley, Daniel Whitehouse, Simon Lucas,  Peter I. Cowling, Philipp Rohlfshagen, Stephen Tavener, Diego Perez, Spyridon Samothrakis and Simon Colton. *IEEE Transactions on Computational Intelligence and AI in Games*, (4)1: 1-49, 2012

-   [Monte-Carlo Tree Search: A New Framework for Game AI](https://www.aaai.org/Papers/AIIDE/2008/AIIDE08-036.pdf).  Guillaume Chaslot, Sander Bakkes, Istvan Szita, and Pieter Spronck.  In *AIIDE*, 1-2, 2008.
