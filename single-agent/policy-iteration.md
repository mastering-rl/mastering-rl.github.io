
Policy Iteration
----------------

The other common way that MDPs are solved is using *policy iteration* --
an approach that is similar to value iteration. While value iteration
iterates over value functions, policy iteration iterates over policies
themselves, creating a strictly improved policy in each iteration
(except if the iterated policy is already optimal).

Policy iteration first starts with some (non-optimal) policy, such as a
random policy, and then calculates the value of each state of the MDP
given that policy --- this step is called the *policy evaluation*. It
then updates the policy itself for every state by calculating the
expected reward of each action applicable from that state.

The basic idea here is that policy evaluation is easier to computer than
value iteration because the set of actions to consider is fixed by the
policy that we have so far.

Policy evaluation
-----------------

The *expected reward* of policy $\pi$ from $s$ to goal, $V^\pi(s)$, is
weighted avg of reward of the possible state sequences defined by that
policy times their probability given $\pi$.

However, the expected reward $V^\pi(s)$ can also be characterised as a
solution to the equation
$$V^\pi(s) =  \sum_{s' \in S} P_a(s'|s)\ [r(s,a,s') +  \gamma\ V^\pi(s') ]$$
where $a = \pi(s)$, and $V^\pi(s)=0$ for goal states

This set of linear equations can be solved analytically using MATLAB, by
a VI-like procedure, or whatever -- we do not care for this subject.

The *optimal expected reward* $V^*(s)$ is $\max_{\pi} V^\pi(s)$ and the
*optimal policy* is the $\textrm{arg max}$

Policy Iteration
----------------

Let $Q^{\pi}(a,s)$ be the expected reward from $s$ when doing $a$ first
and then following the policy $\pi$:
$$Q^\pi(a,s) \, = \, \sum_{s' \in S} P_a(s'|s)\ [r(s,a,s') \, + \,  \gamma\ V^\pi(s')]$$
When $Q^\pi(a,s) < Q^\pi(\pi(s),s)$, $\pi$ *strictly improved* by
changing $\pi(s)$ to $a$\
*Policy Iteration* computes $\pi^*$ by a sequence of policy evaluations
and improvements:

1.  Starting with arbitrary policy $\pi$

2.  Compute $V^\pi(s)$ for all $s$ (policy evaluation)

3.  Improve $\pi$ by setting $\pi(s):=\argmax_{a \in A(s)} Q^\pi(a,s)$
    (improvement)

4.  If $\pi$ changed in 3, go back to 2, else *finish*

This algorithm finishes with an optimal $\pi^*$ after a finite number of
iterations, because the number of policies is finite, bounded by
$O(|A|^{|S|})$, unlike value iteration, which can theoretically require
infinite iterations.

However, each iteration costs $O(|S|^2 |A| + |S|^3)$. Empirical evidence
suggests that the most efficient is dependent on the particular MDP
model being solved.

## Applications of Policy Gradients

A great application of using off-policy updates in deep Q-learning for robotic arms to learn how to grasp unknown objects. The only input for the problem is the camera data:

<iframe width="560" height="315" src="https://www.youtube.com/embed/cXaic_k80uM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

This is using policy iteration (policy gradient descent) rather than standard Q-learning.


Summary: MDPs
-------------

We covered Markov Decision Processes (MDPs). They differ from classical
planning in that actions can have more than one possible outcome. Each
outcome has an associated probability.

There are two solutions for exhaustively calculating the optimal policy:
value iteration and policy iteration. These are both based on dynamic
programming -- specifically, they use the Bellman equations to
iteratively improve on a non-optimal solution.

Heuristic search can also be used, but does not produce solutions that
are as general -- the work only for states that are reachable from the
initial state of the search.

**What's next?** How to *learn* the probabilities over the action
outcomes using *reinforcement learning*.