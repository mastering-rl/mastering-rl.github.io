## Value Iteration

**Learning outcomes**

1.  Apply value iteration to solve small-scale MDP problems manually and program value iteration algorithms to 
    solve medium-scale MDP problems automatically

2.  Construct a policy from a value function

3.  Compare and contrast value iteration to policy iteration

4.  Discuss the strengths and weaknesses of value iteration and policy iteration algorithms

### Overview

*Value Iteration* is a method for finding the optimal value function $V^*$ by solving the
Bellman equations iteratively. It uses the concept of dynamic programming to keep a value function $V$ that approximates the optimal value function $V^**$, iteratvely improving $V$ until it converges to $V^*$ (or close to it). 

### Algorithm

Once we understand the Bellman equation, the value iteration algorithm is straightforward.

:::{admonition} Algorithm -- Value Iteration

**Input:** MDP $M = \langle S, s_0, A, P_a(s' \mid s), r(s,a,s')\rangle$\
**Output:** Value function $V$

Set $V_0$ to arbitrary value function; e.g., $V_0(s)=0$ for all $s$

$\text{Repeat}$\
$\quad\quad \Delta \leftarrow 0$\
$\quad\quad \text{For each}~ s \in S$\
$\quad\quad\quad\quad v \leftarrow V(s)$\
$\quad\quad\quad\quad \underbrace{V(s) \leftarrow \max_{a \in A(s)} \sum_{s' \in S}  P_a(s' \mid s)\ [r(s,a,s') +  \gamma\ V(s') ]}_{\text{Bellman equation}}$\
$\quad\quad\quad\quad \Delta \leftarrow \max(\Delta, |v - V(S)|)$\
$\text{Until}~ \Delta \leq \theta$
:::

As we can see, this is just applying the Bellman equation iteratively until either the value function $V$ doesn't change anymore, or until it changes in by a very small amount ($\theta$).

Value iteration converges to the optimal policy as iterations continue. $V \mapsto V^*$ as $i \mapsto \infty$. That is, given an infinite amount of iterations, it will be optimal.


Value iteration converges to the optimal value function $V^*$ asymptotically, but in practice, the algorithm is stopped when the *residual*  $\Delta$ reaches some pre-determined threshold $\theta$ -- that is, when the largest change in the values between iterations is "small enough".

A policy can now be easily defined: in a state $s$, given $V$, choose the action with the highest expected reward using policy extraction. The resulting greedy policy $\pi_V$ has it's *loss* bounded by $2 \gamma  \Delta / 1-\gamma$.

### Complexity

The complexity of each iteration is $O(|S|^2 |A|)$. On each iteration,
we iterate in an outer loop over all states in $S$, and in each outer
loop iteration, we need to iterate over all states ($\sum_{s' \in S}$),
meaning $|S|^2$ iterations. But also within each outer loop iteration,
we need to calculate the value for every action to find the maximum.

**The Curse of Dimensionality** Solving MDPs using value iteration is polynomial 
in the size of the state space, but exponential in the number of variables if we use a
factored representation such as a PDDL-like language to represent our problem. If there are $N$
number of variables, each a Boolean, there are $2^N$ number of states.
Value iteration requires us to keep a vector of size
$|2^N|$.

**Question:** Can we do better?

**Answer:** Yes! Using function approximation, which we will see later.

It is clear to see that the value iteration can be easily parallelised
by updating the value of many states at once: the values of states at
step $t + 1$ are dependent only on the value of other states at step
$t$.

### Value iteration example: Grid World.

Assuming $\gamma = 0.9$.



$$
\text{After 1 iteration}

\begin{array}{|c|c|c|c|}
\hline
0.00 & 0.00 & 0.00 & +1\\
\hline
0.00 & --  & 0.00  & -1\\
\hline
0.00 & 0.00 & 0.00  & 0.00\\
\hline
\end{array}$$
$$
\text{After 2 iterations}

\begin{array}{|c|c|c|c|}
\hline
0.00 & 0.00 & 0.72 & +1\\
\hline
0.00 & --  & 0.00  & -1\\
\hline
0.00 & 0.00 & 0.00  & 0.00\\
\hline
\end{array}$$

$$
\text{After 3 iterations}

\begin{array}{|c|c|c|c|}
\hline
0.00 & 0.52 & 0.78 & +1\\
\hline
0.00 & --  & 0.43  & -1\\
\hline
0.00 & 0.00 & 0.00  & 0.00\\
\hline
\end{array}$$
$$
\text{After 4 iterations}

\begin{array}{|c|c|c|c|}
\hline
0.37 & 0.66 & 0.83 & +1\\
\hline
0.00 & --  & 0.51  & -1\\
\hline
0.00 & 0.00 & 0.31  & 0.00\\
\hline
\end{array}$$

$$
\text{After 5 iterations}

\begin{array}{|c|c|c|c|}
\hline
0.51 & 0.72 & 0.84 & +1\\
\hline
0.27 & --  & 0.55  & -1\\
\hline
0.00 & 0.22 & 0.37  & 0.13\\
\hline
\end{array}$$
$$
\text{After 10 iterations}

\begin{array}{|c|c|c|c|}
\hline
0.64 & 0.74 & 0.85 & +1\\
\hline
0.57 & --  & 0.57  & -1\\
\hline
0.49 & 0.43 & 0.48  & 0.28\\
\hline
\end{array}$$

$$
\text{After 1000 iterations}

\begin{array}{|c|c|c|c|}
\hline
0.64 & 0.74 & 0.85 & +1\\
\hline
0.57 & --  & 0.57  & -1\\
\hline
0.49 & 0.43 & 0.48  & 0.28\\
\hline
\end{array}$$
$$
\text{Policy after 3 iterations}

\begin{array}{|c|c|c|c|}
\hline
\rightarrow & \rightarrow & \rightarrow & +1\\
\hline
\uparrow & --  & \uparrow  & -1\\
\hline
\uparrow & \leftarrow & \uparrow  & \leftarrow\\
\hline
\end{array}$$



### Strengths and Limitations

*TODO*