
# Policy iteration

The other common way that MDPs are solved is using *policy iteration* -- an approach that is similar to value iteration. While value iteration iterates over value functions, policy iteration iterates over policies themselves, creating a strictly improved policy in each iteration (except if the iterated policy is already optimal).

Policy iteration first starts with some (non-optimal) policy, such as a random policy, and then calculates the value of each state of the MDP given that policy --- this step is called the *policy evaluation*. It then updates the policy itself for every state by calculating the expected reward of each action applicable from that state.

The basic idea here is that policy evaluation is easier to computer than value iteration because the set of actions to consider is fixed by the policy that we have so far.

## Policy evaluation
An important concept in policy iteration is *policy evaluation*, which is an evaluation of the expected reward of a policy.

The *expected reward* of policy $\pi$ from $s$, $V^\pi(s)$, is the weighted average of reward of the possible state sequences defined by that policy times their probability given $\pi$.

:::{admonition} Definition
*Policy evaluation* can be characterised as $V(s)$ as defined by  the following equation:

$$V^\pi(s) =  \sum_{s' \in S} P_{\pi(s)} (s'|s)\ [r(s,a,s') +  \gamma\ V^\pi(s') ]$$

where $V^\pi(s)=0$ for terminal states.
:::

Note that this is very similar to the [Bellman equation](sec:mdps:bellman-equation), except that we do not value $V(s)$ as the value of the best action, but instead just as the value for $\pi(s)$, the action that would be chosen in $s$ by the policy $\pi$. Note the expression $P_{\pi(s)}(s' \mid s)$ instead of $P_a(s' \mid s)$, which means we only evaluate the action that the policy defines.

Once we understand the definition of policy evaluation, the implementation is straightforward. It is the same as value iteration except that we use the policy evaluation equation instead of the Bellman equation.

:::{admonition} Algorithm -- Policy evaluation

**Input:** $\pi$ the policy for evaluation, and MDP $M = \langle S, s_0, A, P_a(s' \mid s), r(s,a,s')\rangle$\
**Output:** Value function $V$

Set $V$ to arbitrary value function; e.g., $V(s)=0$ for all $s$

$\text{Repeat}$\
$\quad\quad \Delta \leftarrow 0$\
$\quad\quad \text{For each}~ s \in S$\
$\quad\quad\quad\quad \underbrace{V'(s) \leftarrow \sum_{s' \in S}  P_{\pi(s)}(s' \mid s)\ [r(s,a,s') +  \gamma\ V(s') ]}_{\text{Policy evaluation equation}}$\
$\quad\quad\quad\quad \Delta \leftarrow \max(\Delta, |V'(s) - V(s)|)$\
$\quad\quad V \leftarrow V'$\
$\text{Until}~ \Delta \leq \theta$
:::

This set of linear equations can be solved analytically using MATLAB, a procedure like value iteration, or whatever -- we do not care for this subject.

The *optimal expected reward* $V^*(s)$ is $\max_{\pi} V^\pi(s)$ and the *optimal policy* is the $\textrm{arg max}$

## Policy improvement

If we have a policy and we want to improve it, we can change the policy (that is, change the actions recommended for states) by updating the actions it recommends based on $V(s)$ that we receive from the policy evaluation.

Let $Q(a,s)$ be the expected reward from $s$ when doing $a$ first and then following the policy $\pi$. Recall from the chapter on [Markov Decision Processes](sec:mdps) that we define define this as:

$$Q(s,a)  =  \sum_{s' \in S} P_a(s'|s)\ [r(s,a,s') \, + \,  \gamma\ V(s')]$$

In this case,  $V(s')$ is the value function from the policy evaluation.

If there is an action $a$ such that $Q(s,a) > Q(s,\pi(s))$, then the policy $\pi$ can be *strictly improved* by setting $\pi(s) \leftarrow a$. This will improve the overall policy.

## Policy iteration
Pulling together policy evaluation and policy improvement, we can define an *policy Iteration*, which computes an optimal $\pi$ by performing a sequence of interleaved policy evaluations and improvements:

:::{admonition} Algorithm -- Policy Iteration

**Input:** MDP $M = \langle S, s_0, A, P_a(s' \mid s), r(s,a,s')\rangle$\
**Output:** Policy $\pi$

Set $\pi$ to arbitrary policy; e.g. $\pi(s) = a$ for all $s$, where $a \in A$ is an arbitrary action.

$\text{Repeat}$\
$\quad\quad$ Compute $V(s)$ for all $s$ using policy evaluation\
$\quad\quad \text{For each}~ s \in S$\
$\quad\quad\quad\quad$ $\pi(s) \leftarrow \textrm{argmax}_{a \in A(s)}Q(s,a)$\
$\text{Until}~ \pi$ does not change
:::

The policy iteration  algorithm finishes with an optimal $\pi$ after a finite number of iterations, because the number of policies is finite, bounded by $O(|A|^{|S|})$, unlike value iteration, which can theoretically require infinite iterations.

However, each iteration costs $O(|S|^2 |A| + |S|^3)$. Empirical evidence suggests that the most efficient is dependent on the particular MDP model being solved, but that surprisingly few iterations are often required for policy iteration.

## Summary

- Policy iteration is a dynamic programming technique for calculating a policy directly, rather than calculating an optimal $V(s)$ and extracting a policy; but one that uses the concept of values.

- It produces an optimal policy in a finite number of steps.

- Similar to value iteration, for medium-scale problems, it works well, but as the state-space grows, it does not scale well.