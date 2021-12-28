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
(sec:single-agent:reward-shaping)=
# Reward shaping

## Learning Outcomes

1.  Explain how reward shaping can be used to help model-free
    reinforcement learning methods to converge

2.  Manually apply reward shaping for a given potential function to
    solve small-scale MDP problems

3.  Design and implement potential functions to solve medium-scale MDP
    problems automatically

4.  Compare and contrast reward shaping with Q-function initialisation

## Overview

In the previous chapters, we looked at fundamental temporal difference (TD) methods for reinforcement learning. As noted, these methods have some weaknesses, including that rewards are sometimes *sparse*. This means that  there are few state/actions that lead to non-zero rewards. This is problematic because initially, reinforcement learning algorithms behave entirely randomly and will struggle to find good rewards. Remember the example of a [UCT algorithm playing Freeway](sec:monte-carlo-tree-search:demo).

In this section, we look at two simple approaches that can improve temporal difference methods:

1.  *Reward shaping*: If rewards are sparse, we can modify/augment our reward function to reward behaviour that we think moves us closer to the solution.
2.  *Q-Value Initialisation*:  We can "guess" a good Q-function at the start and initialise $Q(s,a)$ to be this at the start, which will guide our learning algorithm.

## Reward shaping

What is reward shaping? The basic idea is to give small intermediate rewards to the algorithm that help it converge more quickly.

In many applications, you will have some idea of what a good solution should look like. For example, in our simple navigation task, it is clear that moving towards the reward of +1 and away from the reward of -1 are likely to be good solutions. 

Can we then speed up learning and/or improve our final solution by nudging our
reinforcement learner towards this behaviour?

The answer is: Yes! We can modify our reinforcement learning algorithm
slightly to give the algorithm some information to help, while also guaranteeing optimality.

This information is known as *domain knowledge* --- that is, stuff about the domain
that the human modeller knows about while constructing the model to be
solved.

:::{admonition} Exercise: Freeway What would be a good heuristic for the Freeway game to
learn how to get the kangaroo across the freeway?

![image](./figs/freeway_screenshot.png)
:::

```{code-cell} ipython3
---
tags: [remove-cell]
---
from myst_nb import glue
from gridworld import GridWorld

mdp = GridWorld()
gridworld_image = mdp.visualise()
glue("gridworld_image", gridworld_image, display=False)
```

:::{admonition} Exercise: GridWorld What would be a good heuristic for GridWorld?

```{glue:} gridworld_image
```

:::

### Shaped Reward

In TD learning methods, we update a Q-function when a reward is received. E.g, for 1-step Q-learning:

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

The approach to reward shaping is not to modify the reward function or the received reward $r$, but to just give some additional *shaped reward* for some actions:

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \underbrace{F(s,s')}_{\text{additional reward}} + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

The purpose of the function is to give an additional reward $F(s,s')$ when any action transitions from state $s$ to state $s'$. The function $F : S \times S \to \mathbb{R}$ provides *heuristic* domain knowledge to the problem that is typically manually programmed. 

We say that $r + F(s,s')$ is the *shaped reward* for an action.

Further, we say that $G^{\Phi} = \sum_{i=0}^{\infty} \gamma^i (r_i + F(s_i,s_{i+1}))$ is the
shaped reward for the entire trace.

If we define $F(s,s') > 0$ for states $s$ and $s'$, then this provides a small positive reward for transitioning from $s$ to $s'$, thus encouraging actions that transition from $s$ to $s'$ in future exploitation. If we define $F(s,s') < 0$ for states $s$ and $s'$, then this provides a small *negative* reward for transitioning from $s$ to $s'$, thus discouraging actions that transition like this in future exploitation.

### Potential-based Reward Shaping 

*Potential-based* reward shaping is a particular type of reward shaping with nice theoretical guarantees. In potential-based reward shaping, $F$ is of the form:

$$F(s,s') = \gamma \Phi(s') - \Phi(s)$$

We call $\Phi$ the *potential function* and $\Phi(s)$ is the *potential* of state $s$.

So, instead of defining $F : S \times S \to \mathbb{R}$, we define $\Phi : S \to \mathbb{R}$, which is some heuristic measure of the value of each state $s \in S$.

**Theoretical guarantee**: this will still converge to the optimal policy under the assumption that all state-action pairs are sampled infinitely often.

This is quite straightforward to show as follows. Consider an episode with shaped reward $G^{\Phi}$:

$
\begin{array}{lll}
G^{\Phi} & = & \sum_{i=0}^{\infty} \gamma^i (r_i + F(s_i,s_{i+1}))\\
         & = & \sum_{i=0}^{\infty} \gamma^i (r_i + \gamma\Phi(s_{i+1}) - \Phi(s_i))\\
         & = & \sum_{i=0}^{\infty} \gamma^i r_i + \sum_{i=0}^{\infty}\gamma^{i+1}\Phi(s_{i+1}) - \sum_{i=0}^{\infty}\gamma^i\Phi(s_i)\\
         & = & G + \sum_{i=0}^{\infty}\gamma^{i}\Phi(s_{i}) - \Phi(s_0) - \sum_{i=0}^{\infty}\gamma^i\Phi(s_i) \\
         & = & G - \Phi(s_0)
\end{array}
$

where $G$ refers to the shaped reward for the episode, and $s_0$ is the starting state of the episode. What this says is that the shaped reward $G^{\Phi}$ is just the unshaped reward $G$ minus the potential of the initial state $s_0$. However, because $F$ does not depend on the actions and $G^{\Phi}$ does not depend on shaped rewards beyond the initial state, the *shaped* Q function, which we refer to as $Q^{\Phi}$, can be defined as just $Q^{\Phi}(s,a) = Q(s,a) + \Phi(s)$. Given this, any optimal policy extracted from $Q^{\Phi}$ will be equivalent to any optimal policy extracted from $Q$.

**However!** While it provides guarantees about the end result, potential-based reward shaping may either increase or decrease the time taken to learn. A well-designed potential function decrease the time to convergence.

###  Example -- Potential Reward Shaping for GridWorld 

For Grid World, we use the Manhattan distance to define the potential function, normalised by the size of the grid:

$$
\Phi(s) = 1 - \frac{|x(g) - x(s)| + |y(g) - y(s)|}{width + height - 2}
$$

in which $x(s)$ and $y(s)$ return the $x$ and $y$ coordinates of the agent respectively, $g$ is the goal state. and $width$ and $height$ are the width and height of the grid respectively. Note that the coordinates are indexed from 0, so we subtract 2 from the denominator.

Even on the very first iteration, a greedy policy such as $\epsilon$-greedy, will feedback those states closer to the +1 reward. From state (1,2) with $\gamma=0.9$ if we go Right, we get:

$$
\begin{array}{lll}
  F((1,2), (2,2)) & = & \gamma\Phi(2,2) - \Phi(1,2)\\
                    & = & 0.9 \cdot (1 - \frac{1}{5}) - (1 - \frac{2}{5})\\
                    & = & 0.12
\end{array}
$$

We can compare the Q-values for these states for the four different possible moves that could have been taken from (1,2), using and $\alpha=0.5$ and $\gamma=0.9$:

$$
\begin{array}{lllcc}
\hline
 \textbf{Action}  & r & F(s,s') & \gamma \max_{a'}Q(s',a') & \textrm{New}~ Q(s,a)\\
 \hline
 Up    & 0 & 0.9(1 - \frac{2}{5}) - (1 - \frac{2}{5}) = -0.06 & 0 & -0.06\\
 Down  & 0 & 0.9(1 - \frac{2}{5}) - (1 - \frac{2}{5}) = -0.06 & 0 & -0.06\\
 Right & 0 & 0.9(1 - \frac{1}{5}) - (1 - \frac{2}{5}) = \phantom{-}0.12 & 0 & \phantom{-}0.12\\
 Left  & 0 & 0.9(1 - \frac{3}{5}) - (1 - \frac{2}{5}) = -0.24 & 0 & -0.24\\
 \hline
 \end{array}
$$
Thus, we can see that our potential reward function rewards actions that go towards the goal and penalises actions that go away from the goal. Recall that state (1,2) is in the top row, so action Up just leaves us in state (1,2) and Down similarly because we cannot go through the walls.

But! It will not always work. Compare states (0,0) and (0,1). Our potential function will reward (0,1) because it is closer to the goal, but we know from from our value iteration example that (0,0) is a higher value state than (0,1). This is because our reward function does not consider the negative reward.

In practice, it is non-trivial to derive a perfect reward function -- it is the same problem as deriving the perfect search heuristic. If we could do this, we would not need to even use reinforcement learning -- we could just do a greedy search over the reward function.

### Implementation

To implement potential-based reward shaping, we need to first implement a potential function. We implement potential functions as subclasses of ``PotentialFunction``. For the GridWorld example, the potential function is 1 minus the normalised distance from the goal:

```{code-cell} ipython3
:load: "../python_code/gridworld_potential_function.py"
```

Reward shaping for Q-learning is then a simple extension of the ``QLearning`` class, overriding the ``update`` method:

```{code-cell} ipython3
:load: "../python_code/reward_shaped_qlearning.py"
```

We can run this on the simple GridWorld example:

```{code-cell} ipython3
from qtable import QTable
from qlearning import QLearning
from reward_shaped_qlearning import RewardShapedQLearning
from gridworld_potential_function import GridWorldPotentialFunction
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy


mdp = GridWorld(width = 15, height = 12, goals = [((14,11), 1), ((13,11), -1)])
qfunction = QTable()
potential = GridWorldPotentialFunction(mdp)
RewardShapedQLearning(mdp, EpsilonGreedy(), potential, qfunction).execute()
policy = qfunction.extract_policy(mdp)
mdp.visualise_q_function(qfunction)
mdp.visualise_policy(policy)
reward_shaped_rewards = mdp.get_rewards()
```

Now, we compare this with Q-learning without reward shaping:

```{code-cell} ipython3
mdp = GridWorld(width = 15, height = 12, goals = [((14,11), 1), ((13,11), -1)])
qfunction = QTable()
QLearning(mdp, EpsilonGreedy(), qfunction).execute()
policy = qfunction.extract_policy(mdp)
mdp.visualise_q_function(qfunction)
mdp.visualise_policy(policy)
q_learning_rewards = mdp.get_rewards()
```

If we plot the average episode length during training, we see that reward shaping reduces the length of the early episodes because it has knowledge nudging it towards the goal::

```{code-cell} ipython3
from plot import Plot

Plot.plot_episode_length(
    ["Tabular Q-learning", "Reward shaping"],
    [q_learning_rewards, reward_shaped_rewards],
)
```

## Q-function initialisation 

An approach related to reward shaping is *Q-function initialisation*. Recall that TD learning methods can start
at any arbitrary Q-function. The closer our Q-function is to the optimal Q-function, the quicker it will converge.

Imagine if we happened to initialise our Q-function to the optimal Q-function. It would converge in one step!

Q-function initialisation is similar to reward shaping: we use heuristics to assign higher values to 'better' states. If we just define $\Phi(s) = V_0(s)$, then they are equivalent. In fact, if our potential function is *static* (the definition does not change during learning), then Q-function initialisation and reward shaping are equivalent[^1].

### Example -- Q-function Initialisation in GridWorld 

Using the idea of  Manhattan distance for a potential function, we can define an initial Q-function as follows for state (1,2) using our potential function:

$$
\begin{array}{llllr}
 Q((1,2), Up) & = & 0.9(1 - \frac{2}{5}) - (1 - \frac{2}{5}) & = & -0.06\\
 Q((1,2), Down) & = & 0.9(1 - \frac{2}{5}) - (1 - \frac{2}{5}) & = & -0.06\\
 Q((1,2), Right)  & = & 0.9(1 - \frac{1}{5}) - (1 - \frac{2}{5}) & = & 0.12\\ 
 Q((1,2), Left)  & = & 0.9(1 - \frac{3}{5}) - (1 - \frac{2}{5}) & = & -0.24
\end{array}
$$

Once we start learning over episodes, we will select those actions with a higher heuristic value, and also we are already closer to the optimal Q-function, so will will converge faster. As with reward shaping though, this entirely depends on having a good potential funtion! A poor potential function will give an inaccurate initial Q-function, which may take longer to converge.

## Summary

- A weakness of model-free methods is that they spend a lot of time exploring at the start of the learning. It is not until they find some rewards that the learning begins. This is particularly problematic when rewards are sparse.
- Reward shaping takes in some domain knowledge that "nudges" the learning algorithm towards more positive actions.
- Q-function initialisation is a "guess" of the initial Q-function to guide early exploration
- Reward sharping and Q-function initialisation are equivalent if our potential function is static.

### Related Reading

-   Chapter 9 (Approximate Solution Methods) of *Introduction to
    Reinforcement Learning* \[*Sutton and Barto*\]: <https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html>


[^1]: Wiewiora: ?Potential-based shaping and Q-value initialization are equivalent.? (JAIR, 2003)
