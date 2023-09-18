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
(sec:policy-iteration)=
# Policy iteration
## Learning outcomes

````{margin}
```{admonition} Video byte: Introduction to policy-based approaches and policy iteration
<iframe width="248" height="141" src="https://www.youtube.com/embed/h_zVV_hfPD8?start=0s" title="Policy iteration" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

The learning outcomes of this chapter are:

1. Apply policy iteration to solve small-scale MDP problems manually and program policy iteration algorithms to solve medium-scale MDP problems automatically
2. Discuss the strengths and weaknesses of policy iteration
3. Compare and contrast policy iteration to value iteration

## Overview

````{margin}
```{admonition} Video byte: Intuition of policy-based approaches 
<iframe width="248" height="141" src="https://www.youtube.com/embed/h_zVV_hfPD8?start=83" title="Policy iteration" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

The other common way that MDPs are solved is using *policy iteration* -- an approach that is similar to value iteration. While value iteration iterates over value functions, policy iteration iterates over policies themselves, creating a strictly improved policy in each iteration (except if the iterated policy is already optimal).

Policy iteration first starts with some (non-optimal) policy, such as a random policy, and then calculates the value of each state of the MDP given that policy --- this step is called the *policy evaluation*. It then updates the policy itself for every state by calculating the expected reward of each action applicable from that state.

The basic idea here is that policy evaluation is easier to computer than value iteration because the set of actions to consider is fixed by the policy that we have so far.

## Policy evaluation

````{margin}
```{admonition} Video byte: Model-based policy evaluation
<iframe width="248" height="141" src="https://www.youtube.com/embed/h_zVV_hfPD8?start=169" title="Policy iteration" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

An important concept in policy iteration is *policy evaluation*, which is an evaluation of the expected reward of a policy.

The *expected reward* of policy $\pi$ from $s$, $V^\pi(s)$, is the weighted average of reward of the possible state sequences defined by that policy times their probability given $\pi$.

:::{admonition} Definition
*Policy evaluation* can be characterised as $V^{\pi}(s)$ as defined by  the following equation:

$$V^\pi(s) =  \sum_{s' \in S} P_{\pi(s)} (s' \mid s)\ [r(s,a,s') +  \gamma\ V^\pi(s') ]$$

where $V^\pi(s)=0$ for terminal states.
:::

Note that this is very similar to the [Bellman equation](sec:mdps:bellman-equation), except $V^\pi(s)$ is not the value of the best action, but instead just as the value for $\pi(s)$, the action that would be chosen in $s$ by the policy $\pi$. Note the expression $P_{\pi(s)}(s' \mid s)$ instead of $P_a(s' \mid s)$, which means we only evaluate the action that the policy defines.

Once we understand the definition of policy evaluation, the implementation is straightforward. It is the same as value iteration except that we use the policy evaluation equation instead of the Bellman equation.

:::{admonition} Algorithm -- Policy evaluation

**Input:** $\pi$ the policy for evaluation, $V^\pi$ value function, and MDP $M = \langle S, s_0, A, P_a(s' \mid s), r(s,a,s')\rangle$\
**Output:** Value function $V^\pi$

$\text{Repeat}$\
$\quad\quad \Delta \leftarrow 0$\
$\quad\quad \text{For each}~ s \in S$\
$\quad\quad\quad\quad \underbrace{V'^{\pi}(s) \leftarrow \sum_{s' \in S}  P_{\pi(s)}(s' \mid s)\ [r(s,a,s') +  \gamma\ V^\pi(s') ]}_{\text{Policy evaluation equation}}$\
$\quad\quad\quad\quad \Delta \leftarrow \max(\Delta, |V'^\pi(s) - V^\pi(s)|)$\
$\quad\quad V^\pi \leftarrow V'^\pi$\
$\text{Until}~ \Delta \leq \theta$
:::

The *optimal expected reward* $V^*(s)$ is $\max_{\pi} V^\pi(s)$ and the *optimal policy* is the $\textrm{arg max}$

## Policy improvement

````{margin}
```{admonition} Video byte: Model-based policy improvement
<iframe width="248" height="141" src="https://www.youtube.com/embed/h_zVV_hfPD8?start=418" title="Policy iteration" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

If we have a policy and we want to improve it, we can change the policy (that is, change the actions recommended for states) by updating the actions it recommends based on $V(s)$ that we receive from the policy evaluation.

Let $Q^{\pi}(s,a)$ be the expected reward from $s$ when doing $a$ first and then following the policy $\pi$. Recall from the chapter on [Markov Decision Processes](sec:mdps) that we define define this as:

$$Q^{\pi}(s,a)  =  \sum_{s' \in S} P_a(s' \mid s)\ [r(s,a,s') \, + \,  \gamma\ V^{\pi}(s')]$$

In this case,  $V^{\pi}(s')$ is the value function from the policy evaluation.

If there is an action $a$ such that $Q^{\pi}(s,a) > Q^{\pi}(s,\pi(s))$, then the policy $\pi$ can be *strictly improved* by setting $\pi(s) \leftarrow a$. This will improve the overall policy.

## Policy iteration

````{margin}
```{admonition} Video byte: Intuition of policy-based approaches
<iframe width="248" height="141" src="https://www.youtube.com/embed/h_zVV_hfPD8?start=572" title="Policy iteration" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

Pulling together policy evaluation and policy improvement, we can define an *policy Iteration*, which computes an optimal $\pi$ by performing a sequence of interleaved policy evaluations and improvements:

````{margin}
```{admonition} Video byte: Example --- Policy iteration in Gridworld
<iframe width="248" height="141" src="https://www.youtube.com/embed/h_zVV_hfPD8?start=816" title="Policy iteration" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

:::{admonition} Algorithm -- Policy Iteration

**Input:** MDP $M = \langle S, s_0, A, P_a(s' \mid s), r(s,a,s')\rangle$\
**Output:** Policy $\pi$

Set $V^\pi$ to arbitrary value function; e.g., $V^\pi(s)=0$ for all $s$.

Set $\pi$ to arbitrary policy; e.g. $\pi(s) = a$ for all $s$, where $a \in A$ is an arbitrary action.

$\text{Repeat}$\
$\quad\quad$ Compute $V^\pi(s)$ for all $s$ using policy evaluation\
$\quad\quad$ $\text{For each}~ s \in S$\
$\quad\quad\quad\quad$ $\pi(s) \leftarrow \textrm{argmax}_{a \in A(s)}Q^{\pi}(s,a)$\
$\text{Until}~ \pi$ does not change
:::

````{margin}
```{admonition} Video byte: Convergence of policy iteration
<iframe width="248" height="141" src="https://www.youtube.com/embed/h_zVV_hfPD8?start=1085" title="Policy iteration" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

The policy iteration  algorithm finishes with an optimal $\pi$ after a finite number of iterations, because the number of policies is finite, bounded by $O(|A|^{|S|})$, unlike value iteration, which can theoretically require infinite iterations.

However, each iteration costs $O(|S|^2 |A| + |S|^3)$. Empirical evidence suggests that the most efficient is dependent on the particular MDP model being solved, but that surprisingly few iterations are often required for policy iteration.

## Implementation

Below is a Python implementation for policy iteration. In this implementation, the parameter `max_iterations` is the maximum number of iterations of the policy iteration, and the parameter `theta` the largest amount the value function corresponding to the current policy can change before the policy evaluation look terminates.

```{code-cell} ipython3
:load: "../python_code/policy_iteration.py"

```

From this, we can see that policy evaluation  looks very similar to value iteration. The main differences is in the inner loop: instead of finding the action with the maximum Q-value, we simply find the value of the action that is given the policy: ``policy.select_action(state)``.

We can execute this to get the policy:

```{code-cell} ipython3
from gridworld import GridWorld
from policy_iteration import PolicyIteration
from tabular_policy import TabularPolicy

gridworld = GridWorld()
policy = TabularPolicy(default_action=gridworld.LEFT)
PolicyIteration(gridworld, policy).policy_iteration(max_iterations=100)
gridworld.visualise_policy(policy)
```

We can see that this matches the optimal policy according to [value iteration](sec:value_iteration).

Let's look at the policies that are generated after each iteration, noting that the initial policy is defined by taking a random action (left) and using that for every state:

```{div} full-width
<div id="container" markdown="1" style="text-align: center;">
    <img id="policy_iteration" src=https://gibberblot.github.io/rl-notes/gifs/policy_iteration.gif width=360 height=303 rel:auto_play="0">
    <gif-player id="policy_iteration"></gif-player>
</div>
<p>
```

We can see that this converges in just four iterations. Let's try on a larger state space of a 20 x 15 grid:

```{code-cell} ipython3
gridworld = GridWorld(width=20, height=15)
policy = TabularPolicy(default_action=gridworld.LEFT)
iterations = PolicyIteration(gridworld, policy).policy_iteration(max_iterations=100)
print("Number of iterations until convergence: %d" % (iterations))
gridworld.visualise_policy(policy)
```

This terminates in 19 iterations.   We can see that the policy is optimal as it always directs the agent to terminating state at (3,2) with the positive reward. However, the number of iterations can change depending on the initial policy and the order in which actions are evaluated.

## Summary

````{margin}
```{admonition} Video byte: Summary of policy iteration
<iframe width="248" height="141" src="https://www.youtube.com/embed/h_zVV_hfPD8?start=1182" title="Policy iteration" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

- Policy iteration is a dynamic programming technique for calculating a policy directly, rather than calculating an optimal $V(s)$ and extracting a policy; but one that uses the concept of values.

- It produces an optimal policy in a finite number of steps.

- Similar to value iteration, for medium-scale problems, it works well, but as the state-space grows, it does not scale well.
