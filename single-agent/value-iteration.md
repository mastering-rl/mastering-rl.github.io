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

(sec:value-iteration)=
# Value Iteration
## Learning outcomes

The learning outcomes of this chapter are:

1.  Apply value iteration to solve small-scale MDP problems manually and program value iteration algorithms to  solve medium-scale MDP problems automatically
    
2.  Construct a policy from a value function

3.  Discuss the strengths and weaknesses of value iteration

## Overview

*Value Iteration* is a method for finding the optimal value function $V^*$ by solving the
Bellman equations iteratively. It uses the concept of dynamic programming to maintain  a value function $V$ that approximates the optimal value function $V^*$, iteratively improving $V$ until it converges to $V^*$ (or close to it). 

````{margin}
```{admonition} Video byte: Introduction to value iteration
<iframe width="248" height="141" src="https://www.youtube.com/embed/UwjvpYrCUZ0?start=1992" title="Value iteration" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

## Algorithm


Once we understand the Bellman equation, the value iteration algorithm is straightforward: we just repeatedly calculate $V$ using the Bellman equation until we converge to the solution or we execute a pre-determined number of iterations.

:::{admonition} Algorithm -- Value Iteration

**Input:** MDP $M = \langle S, s_0, A, P_a(s' \mid s), r(s,a,s')\rangle$\
**Output:** Value function $V$

Set $V$ to arbitrary value function; e.g., $V(s)=0$ for all $s$

$\text{Repeat}$\
$\quad\quad \Delta \leftarrow 0$\
$\quad\quad \text{For each}~ s \in S$\
$\quad\quad\quad\quad \underbrace{V'(s) \leftarrow \max_{a \in A(s)} \sum_{s' \in S}  P_a(s' \mid s)\ [r(s,a,s') +  \gamma\ V(s') ]}_{\text{Bellman equation}}$\
$\quad\quad\quad\quad \Delta \leftarrow \max(\Delta, |V'(s) - V(s)|)$\
$\quad\quad V \leftarrow V'$\
$\text{Until}~ \Delta \leq \theta$
:::

````{margin}
```{admonition} Video byte: Example: Value iteration on GridWorld
<iframe width="248" height="141" src="https://www.youtube.com/embed/UwjvpYrCUZ0?start=2295" title="Value iteration on GridWorld" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

As we can see, this is just applying the Bellman equation iteratively until either the value function $V$ doesn't change anymore, or until it changes in by a very small amount ($\theta$).



````{margin}
```{admonition} Video byte: Quiz: Value iteration in GridWorld
<iframe width="248" height="141" src="https://www.youtube.com/embed/UwjvpYrCUZ0?start=2773" title="Quiz: Value iteration in GridWorld" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

We could also write the algorithm using the idea of Q-values, which is closer to a code-based implementation. For this, the loop is:

$\quad\quad \Delta \leftarrow 0$\
$\quad\quad \text{For each}~ s \in S$\
$\quad\quad\quad\quad \text{For each}~ a \in A(s)$\
$\quad\quad\quad\quad\quad\quad Q(s,a) \leftarrow \sum_{s' \in S}  P_a(s' \mid s)\ [r(s,a,s') +  \gamma\ V(s') ]$\
$\quad\quad\quad\quad \Delta \leftarrow \max(\Delta, |\max_{a \in A(s)} Q(s,a) - V(s)|)$\
$\quad\quad\quad\quad V(s) \leftarrow \max_{a \in A(s)} Q(s,a)$


Value iteration converges to the optimal policy as iterations continue: $V \mapsto V^*$ as $i \mapsto \infty$, where $i$ is the number of iterations. So, given an infinite amount of iterations, it will be optimal.



Value iteration converges to the optimal value function $V^*$ asymptotically, but in practice, the algorithm is stopped when the *residual*  $\Delta$ reaches some pre-determined threshold $\theta$ -- that is, when the largest change in the values between iterations is "small enough".

A policy can now be easily defined: in a state $s$, given $V$, choose the action with the highest expected reward using policy extraction. The loss of the result greedy policy is bound by $\frac{2 \gamma  \Delta}{1-\gamma}$.

Note that we do not need an optimal value function $V$ to obtain an optimal policy. A value function that is "close enough" can still give an optimal policy because the small values do not change the resulting policy. Of course, we would not *know* whether a policy is optimal unless we know the value function is optimal.



(sec:value-iteration:implementation)=
## Implementation

Below is a Python implementation for value iteration. In this implementation, the parameter `iterations` is the number of iterations around the loop, which will terminate before convergence is the maximum number of iterations is reach. The parameter `theta` is $\theta$ in the value iteration algorithm above. Once the difference ($\Delta$) is less than `theta` , the loop will terminate.

```{code-cell} ipython3
:load: ../python_code/value_iteration.py


```

Given this, we can create a GridWorld MDP, and solve using value iteration. The code below computes a value function using value iteration for 100 iterations:

```{code-cell} ipython3
from gridworld import GridWorld
from value_iteration import ValueIteration
from tabular_value_function import TabularValueFunction

gridworld = GridWorld()
values = TabularValueFunction()
ValueIteration(gridworld, values).value_iteration(max_iterations=100)
gridworld.visualise_value_function(values, "Value function after 100 iterations")
```

From the value function, we extract a policy:

```{code-cell} ipython3
policy = values.extract_policy(gridworld)
gridworld.visualise_policy(policy, "Policy after 100 iterations")
```

Using the visualisation below, stepping through the 100 iterations, we can see that using value iteration, the values converge within about 10 iterations (to two decimal places), with each iteration giving us diminishing returns:


````{margin}
```{admonition} Video byte: Convergence of value iteration
<iframe width="248" height="141" src="https://www.youtube.com/embed/UwjvpYrCUZ0?start=3134" title="Convergence of value iteration" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

<div id="container" markdown="1" style="text-align: center;">
    <img id="gridworld_value_function" src=https://gibberblot.github.io/rl-notes/gifs/value_iteration.gif width=360 height=303 rel:auto_play="0">
    <gif-player id="gridworld_value_function" width=500></gif-player>
</div>
<p>

+++

The process works in exactly the same way for the Contested Crossing problem. However, in this case, the state space is no longer represented only by the coordinates of the agent. We also have to take into account non-locational information - damage to the ship, damage to the enemy, and the previous direction of travel - since these features will also affect the likelihood of the ship reaching its goal.

```{code-cell} ipython3
from contested_crossing import ContestedCrossing
from value_iteration import ValueIteration
from tabular_value_function import TabularValueFunction

ccross = ContestedCrossing()
values = TabularValueFunction()
ValueIteration(ccross, values).value_iteration(max_iterations=100)
enemy_health = direction = 1
for x in [1,2]:
    for y in [1,2]:
        for ship_health in [1,2]:
            print("state: {0} - value: {1}".format((x,y, ship_health, enemy_health, direction),
                                                   round(values.value_table[(x, y, ship_health, enemy_health, direction)], 3)))
```

This can be difficult to visualise graphically. We can no longer simply assign one value to each physical location and so map the progress of the agent from each value to the largest adjacent one. We can, however, use other ways to get a general sense of how the state affects the behaviour of an agent at any one point. For instance, we can show the mean and standard deviation for all states at a particular location, or we can show tables of values at each location to express the differnece between different possible states located there.

```{code-cell} ipython3
ccross.visualise_value_function(values, "Value function after 100 iterations", mode=3, cell_size=1.6)
ccross.visualise_value_function(values, "Value function after 100 iterations, with sub-tables", mode=0, cell_size=1.6)
```

An agent simply traversing the map to each successive location with the highest mean value of all states would choose the safe path - heading west to the nearest no-danger location, traversing around the safe zones and only briefly cutting back in through the low-danger zones at the end. The visualisation which includes means for different values of ship health and enemy health (the two smaller columns) shows a more complicated picture. In states where the ship has full health the highest value first move is north-west and subsequent highest-value moves are all north-east, straight to the opposite shore. However, where the ship has sustained damage (health values 1 and 2) it is more likely that high-value states are found in the low-danger and no-danger areas, meaning that the ship will choose a safer path.

Another way to visualise this is by aggregate policy plot or path plot of actual uses of the policy. An aggregate policy plot (aggregating at each location over all states which include that location) shows which policies may be preferred at which locations, for different full state values. The opacity of each arrow (or of the starburst which represents the 'shoot' action) shows how many different states have the action as policy. This is different from a stochastic policy, because the choice of policy is not based on probabilities - it is simply a representation of the fact that multiple states exist at one location.

The path plot gives a number of traces of the actual movement of an agent following the policy, over multiple iterations. Since the outcome of 'shoot' actions is random (as is the outcome of being shot at by the enemy), different paths may be taken by agents following the same policy. In this plot, the path colour becomes more red for higher values of ship damage (lower values of ship health) and becomes more blue for higher values of enemy damage. In this way we can see how the results of following the policy change depending on random outcomes during the operation of the agent. Agents which have been damaged tend to follow the safer path round the outside of the map while agents which have damaged the enemy (to the extent of destroying it completely) head straight for the nearest shore. Occasionally an agent is sunk (represented by a black star) - although the policy is optimal on average, this does not guarantee success at every iteration.

```{code-cell} ipython3
policy = values.extract_policy(ccross)
ccross.visualise_as_image(policy=policy,title="Policy Plot",mode=0)
ccross.visualise_as_image(policy=policy,title="Path Plot",mode=1)
```


## Complexity

````{margin}
```{admonition} Video byte: Time complexity of value iteration
<iframe width="248" height="141" src="https://www.youtube.com/embed/UwjvpYrCUZ0?start=3354" title="Time complexity of value iteration" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

The complexity of each iteration is $O(|S|^2 |A|)$. On each iteration, we iterate in an outer loop over all states in $S$, and in each outer loop iteration, we need to iterate over all states ($\sum_{s' \in S}$), meaning $|S|^2$ iterations. But also within each outer loop iteration, we need to calculate the value for every action to find the maximum.

It is clear to see that the value iteration can be easily parallelised by updating the value of many states at once: the values of states at step $t + 1$ are dependent only on the value of other states at step $t$.

## Summary

````{margin}
```{admonition} Video byte: Summary: MDPs and value iteration
<iframe width="248" height="141" src="https://www.youtube.com/embed/UwjvpYrCUZ0?start=3794" title="Summary: MDPs and value iteration" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

- Value iteration is an algorithm for calculating a  value function $V$, from which a policy can be extracted using policy extraction.

- It produces an optimal policy  an infinite amount of time.

- For medium-scale problems, it works well, but as the state-space grows, it does not scale well.

