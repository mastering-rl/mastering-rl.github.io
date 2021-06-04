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
# n-step reinforcement learning

## Learning outcomes

1.  Manually apply n-step reinforcement learning approximation to solve small-scale MDP problems

2.  Design and implement n-step reinforcement learning to solve
    medium-scale MDP problems automatically

3.  Argue the strengths and weaknesses of n-step reinforcement learning

## Overview

In the previous sections on of this chapter, we looked at two fundamental temporal
difference (TD) methods for reinforcement learning: Q-learning and
SARSA.

These two methods have some weaknesses in this basic format:

1.  Unlike Monte-Carlo methods, which reach a reward and then
    backpropagate this reward, TD methods use bootstrapping (they
    estimate the future discounted reward using $Q(s,a)$), which means
    that for problems with sparse rewards, it can take a long time to
    for rewards to propagate throughout a Q-function.
2.  Rewards can be sparse, meaning that there are few state/actions that
    lead to non-zero rewards. This is problematic because initially,
    reinforcement learning algorithms behave entirely randomly and will
    struggle to find good rewards. 
3.  Both methods estimate a Q-function $Q(s,a)$, and the simplest way to
    model this is via a Q-table. However, this requires us to maintain a
    table of size $|A| \times |S|$, which is prohibitively large for any
    non-trivial problem.
4.  Using a Q-table requires that we visit every reachable state many
    times and apply every action many times to get a good estimate of
    $Q(s,a)$. Thus, if we never visit a state $s$, we have no estimate
    of $Q(s,a)$, even if we have visited states that are very similar to
    $s$.

To get around limitations 1 and 2, we are going to look at n-step temporal difference learning: 'Monte Carlo' techniques execute entire traces and then backpropagate the reward, while basic TD methods only look at the reward in the next step, estimating the future wards. n-step methods instead look $n$ steps ahead for the reward before updating the reward, and then estimate the remainder. In future parts of these notes, we'll look at techniques for mitigating limitations 3 and 4.

n-step TD learning comes from the idea used in the image below, from Sutton and Barto (2020). Monte Carlo methods uses 'deep backups', where entire traces are executed and the reward backpropagated. Methods such as Q-learning and SARSA use 'shallow backups', only using the reward from the 1-step ahead. n-step learning finds the middle ground: only update the Q-function after having explored ahead $n$ steps.

```{figure} ./figs/RL_approaches.png
:name: RL_approaches

Reinforcement learning approaches (from Sutton and Barto (2020))
```

## n-step TD learning

We will look at n-step reinforcement learning, in which $n$ is the parameter that
determines the number of steps that we want to look ahead before updating the Q-function. So for $n=1$, this is just "normal" TD learning such as Q-learning or SARSA.
When $n=2$, the algorithm looks one step beyond the immediate reward, $n=3$ it looks two steps beyond, etc.

Both Q-learning and SARSA have an n-step version. We will look at n-step learning more generally, and then show an algorithm for n-step
SARSA. The version for Q-learning is similar.

### Discounted Future Rewards (again)

When calculating a discounted reward over a trace, we simply sum up the rewards over the trace:

 $$ G_t   =   r_1 + \gamma r_2 + \gamma^2 r_3 + \gamma^3 r_4 + \ldots $$


We can re-write this as:

 $$ G_t =   r_1 + \gamma(r_2 + \gamma(r_3 + \gamma(r_4 + \ldots))) $$

If $G_t$ is the value received at time-step $t$, then 

 $$ G_t = r_t + \gamma G_{t+1} $$

In TD(0) methods such as Q-learning and SARSA, we do not know $G_{t+1}$
when updating $Q(s,a)$, so we estimate using bootstrapping:

 $$ G_t = r_t + \gamma \cdot V(s_{t+1}) $$ 

That is, the reward of the
entire future from step $t$ is estimated as the reward at $t$ plus the
estimated (discounted) future reward from $t+1$. $V(s_{t+1})$ is
estimated using the maximum expected return (Q-learning) or the
estimated value of the next action (SARSA).

This is a *one-step return*.

### Truncated Discounted Rewards

However, we can estimate a two-step return:

$$ G^2_t = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) $$ 

a three-step return:

$$ G^3_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} +  \gamma^3 V(s_{t+3}) $$

or n-step returns:

$$ G^n_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots  \gamma^n V(s_{t+n}) $$

In this above expression $G^n_t$ is the full reward, *truncated* at $n$
steps, at time $t$. 

The basic idea of n-step reinforcement learning is that we do not update the Q-value
immediately after executing an action: we wait $n$ steps and update it
based on the n-step return.

If $T$ is the termination step and $t+n>T$, then we just use the full
reward.

In Monte-Carlo methods, we go all the way to the end of an episode.
Monte-Carlo Tree Search is one such Monte-Carlo method, but there are
others that we do not cover.

### Updating the Q-function

The update rule is then different. First, we need to
calculate the truncated reward for $n$ steps, in which $\tau$ is the
time step that we are updating for (that is, $\tau$ is the action taken $n$ steps ago):

$$G \leftarrow \sum^{\min(\tau+n, T)}_{i=\tau+1}\gamma^{i-\tau-1}r_i$$

This just sums the discounted rewards from time step $\tau+1$ until either $n$
steps ($\tau+n$) or termination of the episode ($T$), whichever comes
first. 

Then calculate the n-step expected reward:

   $$\text{If } \tau+n < T \text{ then } G \leftarrow G + \gamma^n Q(s_{\tau+n}, a_{\tau+n}).$$

This adds the future expect reward if we are not at the end of the episode (if $\tau+n < T$).

Finally, we update the Q-value:

   $$Q(s_{\tau}, a_{\tau}) \leftarrow  Q(s_{\tau}, a_{\tau}) + \alpha[G - Q(s_{\tau}, a_{\tau}) ]$$

In the update rule above, we are using a SARSA update, but a Q-learning update is similar.


### n-step SARSA

While conceptually this is not so difficult, an algorithm for doing n-step learning needs to store the rewards and observed states for $n$ steps, as well as keep track of which step to update. An algorithm for n-step SARSA is shown below.

:::{admonition} Algorithm -- n-step SARSA

**Input:** MDP $M = \langle S, s_0, A, P_a(s' \mid s), r(s,a,s')\rangle$\, number of steps $n$\
**Output:** Q-function $Q$

Initialise $Q$ arbitrary; e.g., $Q(s,a)=0$ for all $s$ and $a$

Repeat (for each episode)\
$\quad\quad$ $T \leftarrow \infty$\
$\quad\quad$ $t \leftarrow 0$  ($t$ is the current time step of this episode)\
$\quad\quad$ $s \leftarrow$ the first state in episode $e$\
$\quad\quad$ Select action $a$ to apply in $s$ using Q-values in $Q$ and a multi-armed bandit algorithm such as $\epsilon$-greedy\
$\quad\quad$ Repeat (for each step in episode $e)$\
$\quad\quad\quad\quad$ If $t < T$ then:\
$\quad\quad\quad\quad\quad\quad$ Execute action $a_t$ in state $s_t$\
$\quad\quad\quad\quad\quad\quad$ Observe and store reward $r_{t+1}$ and new state $s_{t+1}$\
$\quad\quad\quad\quad\quad\quad$ If $s_{t+1}$ is a terminal state then:\
$\quad\quad\quad\quad\quad\quad\quad\quad$ $T \leftarrow t + 1$\
$\quad\quad\quad\quad\quad\quad$ Else:\
$\quad\quad\quad\quad\quad\quad\quad\quad$ Select & store action $a_{t+1}$ to apply in $s_{t+1}$ using $Q$ and a multi-armed bandit algorithm\
$\quad\quad\quad\quad$ $\tau \leftarrow t - n + 1$  (calculate the index of the action to update)\
$\quad\quad\quad\quad$ If $\tau \geq 0$ then:\
$\quad\quad\quad\quad\quad\quad$ $G \leftarrow \sum^{\min(\tau+n, T)}_{i=\tau+1}\gamma^{i-\tau-1}r_i$\
$\quad\quad\quad\quad\quad\quad$ If $\tau+n < T$ then: $G \leftarrow G + \gamma^n Q(s_{\tau+n}, a_{\tau+n})$\
$\quad\quad\quad\quad\quad\quad$ $Q(s_{\tau}, a_{\tau}) \leftarrow  Q(s_{\tau}, a_{\tau}) + \alpha[G - Q(s_{\tau}, a_{\tau})]$\
$\quad\quad\quad\quad$ $t \leftarrow t + 1$\
$\quad\quad$ Until $\tau = T - 1$
:::

For the first $n-1$ steps of the any episode, we do not update $Q$ at all; that is, if $\tau < 0$.

Also, we have to continue updating $n-1$ steps after the end of the episode, but not selecting actions; that is, if $t \geq T$.

Computationally, this is not much worse than 1-step learning. We need to store the last $n$ states, but the per-step computation is small and uniform for n-step, just as for 1-step.

### Example -- $n$-step SARSA update

Consider our simple 2D navigation task, in which we do not know the probability transitions nor the rewards. Initially, the reinforcement learning algorithm will be required to search randomly until it finds a reward. Propagating this reward back n-steps will be helpful.

Imagine the first episode consisting of the following (very lucky!) trace:

```
  --------------- --------------- --------------- --------------- 
 |               |               |               |               |
 |               2               3               6               |
 |           --------►      --------►        --------► +1.00     |
 |       ▲       |               |   |      ▲    |               |
 |       |       |               | 4 |      |    |               |
  -------|------- --------------- ---|------|---- --------------- 
 |       | 1     | ############# |   |      | 5  |               |
 |       |       | ############# |   ▼      |    |               |
 |               | ############# |               |     -1.00     |
 |       ▲       | ############# |               |               |
 |       |       | ############# |               |               |
  -------|------- --------------- --------------- --------------- 
 |       | 0     |               |               |               |
 |       |       |               |               |               |
 |               |               |               |               |
 |               |               |               |               |
 |               |               |               |               |
  --------------- --------------- --------------- --------------- 
```

Assuming $Q(s,a)=0$ for all $s$ and $a$, if we traverse the episode the labelled episode, what will our Q-function look like for a 5-step update with $\alpha=0.5$ and $\gamma=0.9$?

For the first $n-1$ steps of the episode, no update is made to the Q-values, but rewards and states are stored for future processing. Let's look at those first $n-1 = 4$ steps if we apply n-step SARSA to them:

$$
\begin{array}{llll}
\text{For } n = 5\\[2mm]
 t = 0 & s_0 = (0,0),\ a_0 = Up & r_1 = 0 & s_1 = (0,1)\\
       & T = \infty\\
       & \tau = t - n + 1 = -4\\[1mm]
 t = 1 & s_1 = (0,1),\ a_1 = Up & r_2 = 0 & s_2 = (0,2)\\
       & T = \infty\\
       & \tau = t - n + 1 = -3\\[1mm]
 t = 2 & s_2 = (0,2),\ a_2 = Right & r_3 = 0 & s_3 = (1,2)\\
       & T = \infty\\
       & \tau = t - n + 1 = -2\\[1mm]
 t = 3 & s_3 = (1,2),\ a_3 = Right & r_4 = 0 & s_4 = (2,2)\\
       & T = \infty\\
       & \tau = t - n + 1 = -1
\end{array}
$$

On the next step $t=4$, we reach the end of our n-step window, and we start to update values. At $t=4$, we see that $\tau \geq 0$ becomes true, so we calculate $G$ for $a_0$ and update $Q(s_0, a_0)$, and then similarly for $t=5$:

$$
\begin{array}{llll}
 t = 4 & s_4 = (2,2),\ a_4 = Down & r_5 = 0 & s_5 = (2,1)\\
\phantom{\text{For }$n = 5$} %for consistent spacy
       & T = \infty\\
       & \tau = t - n + 1 = 0\\
       & G = \gamma^0 r_1 + \ldots + \gamma^4 r_5 = 0\\
       & Q(s_0, a_0) = 0\\[1mm]
 t = 5 & s_5 = (2,1),\ a_5 = Up & r_6 = 0 & s_6 = (2,2)\\
       & T = \infty\\
       & \tau = t - n + 1 = 1\\
       & G = \gamma^0 r_2 + \ldots + \gamma^4 r_6 = 0\\
       & Q(s_1, a_1) = 0\\[1mm]
\end{array}
$$

We can see that there are no rewards in the first five steps, so $Q(s_0, a_0)$ and $Q(s_1, a_1)$ both remain 0, their original values.

When we reach $t=6$, however, we reach both a terminal state and we receive a reward. We can then update $Q(s_2, a_2)$:

$$
\begin{array}{llll}
 t = 6 & s_6 = (2,2),\ a_6 = Right \quad\quad r_7 = 1 \quad s_7 = (3,2)\\
 \phantom{\text{For }$n = 5$}
       & T = t + 1 = 7\\
       & \tau = t - n - 1 = 2\\
       & G = \gamma^0 r_3 + \ldots + \gamma^4 r_7 = 0.9^4 \cdot 1 = 0.6561\\
       & Q(s_2, a_2) = 0 + 0.5[0.9^4 \cdot 1 - 0] = 0.32805
       \end{array}
$$

From this point, $t \geq T$, so we no longer select and execute actions, nor store the rewards, actions, and states. However, we continue to update the steps in the episode:

$$
\begin{array}{llll}
 t = 7 & T = 7\\
\phantom{\text{For }$n = 5$}
       & \tau = t - n - 1 = 3\\
       & G = \gamma^0 r_4 + \ldots + \gamma^3 r_7 = 0.9^3 \cdot 1 = 0.729\\
       & Q(s_3, a_3) = 0 + 0.5[0.9^3 \cdot 1 - 0] = 0.3645\\[1mm]
 t = 8 & T = 7\\
       & \tau = t - n - 1 = 4\\
       & G = \gamma^0 r_5 + \ldots + \gamma^2 r_7 = 0.9^2 \cdot 1 = 0.81\\
       & Q(s_4, a_4) = 0 + 0.5[0.9^2 \cdot 1 - 0] = 0.405\\[1mm]
 t = 9 & T = 7\\
       & \tau = t - n - 1 = 5\\
       & G = \gamma^0 r_6 + \gamma^1 r_7 = 0.9^1 \cdot 1 = 0.9\\
       & Q(s_5, a_5) = 0 + 0.5[0.9^1  \cdot 1 - 0] = 0.45\\[1mm]
 t = 10 & T = 7\\
       & \tau = t - n - 1 = 6\\
       & G = \gamma^0 r_7 = 0.9^0 \cdot 1 = 1\\
       & Q(s_6, a_6) = 0 + 0.5[1  \cdot 1 - 0] = 0.5\\[1mm]
\end{array}
$$

At this point, $\tau = 6$ and $T=7$, so the inner loop terminates, and we start a new episode.

The tables below compares 1-step vs. 5-step SARSA for the trace above. In 1-step SARSA, reaching the reward only informs the state from which it is reached. Whereas for 5-step, it informs the previous five steps. Then, in the next episode, there is more chance of encountering a non-zero state, so which will again inform the five steps instead of just one. The rewards 'spread' throughout the Q-table faster.


```{code-cell} ipython3
---
tags: [remove-input]
---
from tabulate import tabulate
import math

alpha = 0.5
gamma = 0.9
headers=["State", "Up", "Down", "Right", "Left"]
data = [[(0,0), 0, 0, 0, 0],
        [(0,1), 0, 0, 0, 0],
        [(0,2), 0, 0, 0, 0],
        ["..."],
        [(1,2), 0, 0, 0, 0],
        [(2,1), 0, 0, 0, 0],
        [(2,2), 0, 0, alpha * math.pow(gamma, 0), 0]]
print ("For n = 1")
print (tabulate(data, headers))
```

```{code-cell} ipython3
---
tags: [remove-input]
---
data = [[(0,0), 0, 0, 0, 0],
        [(0,1), 0, 0, 0, 0],
        [(0,2), 0, 0, alpha * math.pow(gamma, 4), 0],
        ["..."],
        [(1,2), 0, 0, alpha * math.pow(gamma, 3), 0],
        [(2,1), 0, alpha * math.pow(gamma, 1), 0, 0],
        [(2,2), 0, alpha * math.pow(gamma, 2), alpha * math.pow(gamma, 0), 0]]
print ("For n = 5")
print (tabulate(data, headers))
```

## Further Reading

-   Chapter 7 of *Introduction to Reinforcement Learning* \[*Sutton and
    Barto*\]  https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html

