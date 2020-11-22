# Model-based

## Learning Outcomes



1.  Apply value iteration to solve small-scale MDP problems manually and
    program value iteration algorithms to solve medium-scale MDP
    problems automatically

2.  Construct a policy from a value function

3.  Compare and contrast value iteration to policy iteration

4.  Discuss the strengths and weaknesses of value iterapdflatextion and
    policy iteration algorithms

## Relevant Reading

-   *Any* introduction to probability theory --- see the related reading
    on the LMS if you are unfamiliar.

-   Chapter 17 of *Artificial Intelligence --- A Modern Approach* by
    Russell and Norvig. Available in the university library and online
    in PDF format.

-   Chapter 4 of *Reinforcement Learning: An Introduction, second
    edition*. Freely downloadable at
    <http://www.incompleteideas.net/book/RLbook2020.pdf>



Using the Bellman Equations: Value Iteration
--------------------------------------------

**Value Iteration** finds the optimal value function $V^*$ solving the
Bellman equations iteratively, using the following algorithm:

-   Set $V_0$ to arbitrary value function; e.g., $V_0(s)=0$ for all $s$.

-   Set $V_{i+1}$ to result of Bellman's **right hand side** using $V_i$
    in place of $V$:
    $$V_{i+1}(s) := \max_{a \in A(s)} \sum_{s' \in S}  P_a(s'|s)\ [r(s,a,s') +  \gamma\ V_i(s') ]$$

This converges exponentially fast to the optimal policy as iterations
continue.

$V_i \mapsto V^*$ as $i \mapsto \infty$. That is, given an infinite
amount of iterations, it will be optimal.

The complexity of each iteration is $O(|S|^2 |A|)$. On each iteration,
we iterate in an outer loop over all states in $S$, and in each outer
loop iteration, we need to iterate over all states ($\sum_{s' \in S}$),
meaning $|S|^2$ iterations. But also within each outer loop iteration,
we need to calculate the value for every action to find the maximum.

Value Iteration in Practice
---------------------------

Value Iteration converges to the optimal value function $V^*$
asymptotically, but in practice, the algorithm is stopped when the
**residual** $R = \max_s|V_{i+1}(s)-V_i(s)|$ reaches some pre-determined
threshold $\epsilon$ -- that is, when the largest change in the values
between iterations is "small enough".

The resulting greedy policy $\pi_V$ has it's **loss** bounded by
$2 \gamma  R / 1-\gamma$.

It is clear to see that the value iteration can be easily parallelised
by updating the value of many states at once: the values of states at
step $t + 1$ are dependent only on the value of other states at step
$t$.

A policy can now be easily defined: in a state $s$, given $V$, choose
the action with the highest expected reward.

Value iteration example: Grid World.
------------------------------------

Assuming $\gamma = 0.9$.

$$\begin{array}{|c|c|c|c|}
\multicolumn{4}{c}{\textrm{After 1 iteration}}\\
\hline
0.00 & 0.00 & 0.00 & +1\\
\hline
0.00 & \cellcolor{gray!25}  & 0.00  & -1\\
\hline
0.00 & 0.00 & 0.00  & 0.00\\
\hline
\end{array}$$

$$\begin{array}{|c|c|c|c|}
\multicolumn{4}{c}{\textrm{After 2 iterations}}\\
\hline
0.00 & 0.00 & 0.72 & +1\\
\hline
0.00 & \cellcolor{gray!25}  & 0.00  & -1\\
\hline
0.00 & 0.00 & 0.00  & 0.00\\
\hline
\end{array}$$

$$\begin{array}{|c|c|c|c|}
\multicolumn{4}{c}{\textrm{After 3 iterations}}\\
\hline
0.00 & 0.52 & 0.78 & +1\\
\hline
0.00 & \cellcolor{gray!25}  & 0.43  & -1\\
\hline
0.00 & 0.00 & 0.00  & 0.00\\
\hline
\end{array}$$

$$\begin{array}{|c|c|c|c|}
\multicolumn{4}{c}{\textrm{After 4 iterations}}\\
\hline
0.37 & 0.66 & 0.83 & +1\\
\hline
0.00 & \cellcolor{gray!25}  & 0.51  & -1\\
\hline
0.00 & 0.00 & 0.31  & 0.00\\
\hline
\end{array}$$

$$\begin{array}{|c|c|c|c|}
\multicolumn{4}{c}{\textrm{After 5 iterations}}\\
\hline
0.51 & 0.72 & 0.84 & +1\\
\hline
0.27 & \cellcolor{gray!25}  & 0.55  & -1\\
\hline
0.00 & 0.22 & 0.37  & 0.13\\
\hline
\end{array}$$

$$\begin{array}{|c|c|c|c|}
\multicolumn{4}{c}{\textrm{After 100 iterations}}\\
\hline
0.64 & 0.74 & 0.85 & +1\\
\hline
0.57 & \cellcolor{gray!25}  & 0.57  & -1\\
\hline
0.49 & 0.43 & 0.48  & 0.28\\
\hline
\end{array}$$

$$\begin{array}{|c|c|c|c|}
\multicolumn{4}{c}{\textrm{After 1000 iterations}}\\
\hline
0.64 & 0.74 & 0.85 & +1\\
\hline
0.57 & \cellcolor{gray!25}  & 0.57  & -1\\
\hline
0.49 & 0.43 & 0.48  & 0.28\\
\hline
\end{array}$$

$$\begin{array}{|c|c|c|c|}
\multicolumn{4}{c}{\textrm{Policy after 1000 iterations}}\\
\hline
\rightarrow & \rightarrow & \rightarrow & +1\\
\hline
\uparrow & \cellcolor{gray!25}  & \uparrow  & -1\\
\hline
\uparrow & \leftarrow & \uparrow  & \leftarrow\\
\hline
\end{array}$$

Value iteration demo
--------------------

<http://www.cs.ubc.ca/~poole/demos/mdp/vi.html>


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

The Curse of Dimensionality
---------------------------

Thus, solving MDPs using these algorithms is polynomial in the size of
the state space, but exponential in the number of variables if we use a
PDDL-like language to represent our problem. That is, if there are $N$
number of variables, each a Boolean, there are $2^N$ number of states.
Value iteration and policy iteration require us to keep a vector of size
$|2^N|$.

**Question: Can we do better?**

Answer: Yes! Using function approximation, which we will see in a couple
of weeks

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
