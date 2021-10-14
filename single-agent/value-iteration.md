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
(sec:value-iteration)=
# Value Iteration
## Learning outcomes

The learning outcomes of this chapter are:

1.  Apply value iteration to solve small-scale MDP problems manually and program value iteration algorithms to  solve medium-scale MDP problems automatically
    
2.  Construct a policy from a value function

4.  Discuss the strengths and weaknesses of value iteration

## Overview

*Value Iteration* is a method for finding the optimal value function $V^*$ by solving the
Bellman equations iteratively. It uses the concept of dynamic programming to maintain  a value function $V$ that approximates the optimal value function $V^*$, iteratively improving $V$ until it converges to $V^*$ (or close to it). 

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

As we can see, this is just applying the Bellman equation iteratively until either the value function $V$ doesn't change anymore, or until it changes in by a very small amount ($\theta$).

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

## Complexity

The complexity of each iteration is $O(|S|^2 |A|)$. On each iteration, we iterate in an outer loop over all states in $S$, and in each outer loop iteration, we need to iterate over all states ($\sum_{s' \in S}$), meaning $|S|^2$ iterations. But also within each outer loop iteration, we need to calculate the value for every action to find the maximum.

It is clear to see that the value iteration can be easily parallelised by updating the value of many states at once: the values of states at step $t + 1$ are dependent only on the value of other states at step $t$.

(sec:value-iteration:implementation)=
## Implementation

Below is a Python implementation for value iteration. In this implementation, the parameter `iterations` is the number of iterations around the loop, which will terminate before convergence is the maximum number of iterations is reach. The parameter `theta` is $\theta$ in the value iteration algorithm above. Once the difference ($\Delta$) is less than `theta` , the loop will terminate.

```{code-cell} ipython3
:load: "../code/value_iteration.py"
```

Given this, we can create a GridWorld MDP, and solve using value iteration. The code below prints the value function for value iteration after 1, 2, 3, 4, 5, 10, and 100 iterations:

```{code-cell} ipython3
mdp = GridWorld()

for iterations in [0, 1, 2, 3, 4, 5, 10, 1000]:
    values = TabularValueFunction()
    ValueIteration(mdp, values).value_iteration(iterations=iterations)
    mdp.visualise_value_function(values, "After %d iterations" % (iterations))
```

From the policy, we extract a policy:

```{code-cell} ipython3
values = TabularValueFunction()
ValueIteration(mdp, values).value_iteration(iterations=100)
policy = values.extract_policy(mdp)
mdp.visualise_policy(policy, "Policy after 100 iterations")
```


## Summary

- Value iteration is an algorithm for calculating a  value function $V$, from which a policy can be extracted using policy extraction.

- It produces an optimal policy  an infinite amount of time.

- For medium-scale problems, it works well, but as the state-space grows, it does not scale well.
