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

### Intuition

The details and algorithm for n-step reinforcement learning making it seem more complicated than it really is. 
At an intuitive level, it is quite straightforward: at each step, instead of updating our Q-function or policy based on the reward received from the previous action, plus the discounted future rewards, we update it based on the last $n$ rewards received, plus the discounted future rewards from $n$ states ahead.

Consider the following interative gif, which shows the update over an episode of five actions. The bracket represents a window size of $n=3$:

```{div} full-width
<div id="container" markdown="1" style="text-align: center;">
    <img id="n_step_window" src="https://gibberblot.github.io/rl-notes/gifs/n-step-window.gif" width="900" height="400" rel:auto_play="0">
    <gif-player id="n_step_window" width="900"></gif-player>
</div>
<p>
```

At time $t=0$, no update can be made because there is no action.

At time $t=1$, after action $a_0$, no update is done yet. In standard TD learning, we would update $Q(s_0, s_0)$ here. However, because we have only one reward instead of three ($n=3$), we delay the update.

At time $t=2$, after action $a_1$, again we do not update because we have just two rewards, instead of three.

At time $t=3$, after action $a_2$, we have three rewards, so we do our first update. But note: we update $Q(s_0, a_0)$ -- the first state-action pair in the sequence, rather than the most recent state action pair $(s_2,a_2)$. Notice that we consider the rewards $r_1$, $r_2$, and $r_3$ (appropriately discounted), and use the discounted future reward $V(s_3)$.  Effectively, the update of $Q(s_0, a_0)$ "looks forward" three actions into the future instead one, because $n=3$.

At time $t=4$, after action $a_3$, the update is similar: we now update $Q(s_1, a_1)$.

At time $t=5$, it is similar again, except that state $s_5$ is a terminal state, so we do not include the discounted future reward -- there is no state $s_6$ such that we can estimate $V(s_6)$, so it is omitted.

At time $t=6$, we can see that the window slides beyond the length of the episode. However, even though we have reached the terminal state, we continue updating --- this time updating $Q(s_3, s_3)$. We need to do this because we have still not updated the Q-values for all state-action pairs that we have executed.

At time $t=7$, we do the final update --- this time for $Q(s_4,a_4)$; and the episode is complete.

So, we can see that, intuitively, $n$-step reinforcement learning is quite straightfoward. However, to implement this, we need data structures to keep track of the last $n$ states, actions, and rewards, and modifications to the standard TD learning algorithm to both delay Q-value updates in the first $n$ steps of an episode, and to continue updating beyond the end of the episode for ensure the last $n$ state-action pairs are updated. This "book-keeping" code can be confusing at first, unless we already have an intuitive understanding of what it achieves.

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

If $T$ is the termination step and $t + n \geq T$, then we just use the full
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

<!--
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
-->

:::{admonition} Algorithm -- n-step SARSA

**Input:** MDP $M = \langle S, s_0, A, P_a(s' \mid s), r(s,a,s')\rangle$\, number of steps $n$\
**Output:** Q-function $Q$

Initialise $Q$ arbitrary; e.g., $Q(s,a)=0$ for all $s$ and $a$

Repeat (for each episode)\
$\quad\quad$ Select action $a$ to apply in $s$ using Q-values in $Q$ and a multi-armed bandit algorithm such as $\epsilon$-greedy\
$\quad\quad$ $ss = \langle s\rangle$\
$\quad\quad$ $as = \langle a\rangle$\
$\quad\quad$ $rs = \langle \rangle$\
$\quad\quad$ While $ss$ is not empty\
$\quad\quad\quad\quad$ If $s$ is not a terminal state then:\
$\quad\quad\quad\quad\quad\quad$ Execute action $a$ in state $s$\
$\quad\quad\quad\quad\quad\quad$ Observe reward $r$ and new state $s'$\
$\quad\quad\quad\quad\quad\quad$ $rs \leftarrow rs + \langle r\rangle$\
$\quad\quad\quad\quad\quad\quad$ If $s'$ is not a terminal state then:\
$\quad\quad\quad\quad\quad\quad\quad\quad$ Select action $a'$ to apply in $s'$ using $Q$ and a multi-armed bandit algorithm\
$\quad\quad\quad\quad\quad\quad\quad\quad$ $ss \leftarrow ss + \langle s' \rangle$\
$\quad\quad\quad\quad\quad\quad\quad\quad$ $as \leftarrow ss + \langle a' \rangle$\
$\quad\quad\quad\quad$ If $|rs| = n$ or $s$ is a terminal state then:\
$\quad\quad\quad\quad\quad\quad$ $G \leftarrow \sum^{|rs| - 1}_{i=0}\gamma^{i}rs_i$\
$\quad\quad\quad\quad\quad\quad$ If $s$ is not a terminal state then: $G \leftarrow G + \gamma^n Q(s', a')$\
$\quad\quad\quad\quad\quad\quad$ $Q(ss_0, as_0) \leftarrow  Q(ss_0, as_0) + \alpha[G - Q(ss_0, as_0)]$\
$\quad\quad\quad\quad\quad\quad$ $rs \leftarrow rs_{[1 : n + 1]}$\
$\quad\quad\quad\quad\quad\quad$ $ss \leftarrow ss_{[1 : n + 1]}$\
$\quad\quad\quad\quad\quad\quad$ $as \leftarrow as_{[1 : n + 1]}$\
$\quad\quad\quad\quad$ $s \leftarrow s'$\
$\quad\quad\quad\quad$ $a \leftarrow a'$
:::

This is similar to standard SARSA, except that we are storing the last $n$ states, actions, and rewards; and also calculating the rewards on the last five rewards rather than just one. The variables $ss$, $as$, and $rs$ as the list of the last $n$ states, actions, and rewards respectively. We use the syntax $ss_i$ to get the $i^{th}$ element of the list, and the Python-like syntax $ss_{[1:n+1]}$ to get the elements between indices 1 and $n+1$ (remove the first element).

As with SARSA and Q-learning, we iterate over each step in the episode. The first branch simply executes the selected action, selects a new action to apply, and stores the state, action, and reward.

It is the second branch where the actual learning happens. Instead of just updating with the 1-step reward $r$,  we use  the $n$-step reward $G$. This requires a bit of "book-keeping". The first thing we do is calculate $G$. This simply sums up the elements in the reward sequence $rs$, but remembering that they must be discounted based on their position in $rs$. The next line  adds the TD-estimate $y^n Q(s',a')$ to $G$, but only is the most recent state is not a terminal state. If we have already reached the end of the episode, then we must exclude the TD-estimate of the future reward, because there will be no such future reward. Of importance, also note that we multiple this by $\gamma^n$ instead of $\gamma$. Why? This because the future estimated reward is $n$ steps from state $ss_0$. The $n-step$ reward in $G$ somes first. Then we do the actualy update, which updates the state-action pair $(ss_0, as_0)$ that is $n$-steps back.

The final part of this branch removes the first element from the list of states, actions, and rewards, and moves on to the next state.

What is the effect of all of the computation? This algorithm differs from standard SARSA as follows: it only updates a state-action pair after it has seen the next $n$ rewards that are returned, rather than just the single next reward. This means that there are no updates until the $n^{th}$ steps of  the episode; and no TD-estimate of the future reward in the last $n$ steps of the episode.

Computationally, this is not much worse than 1-step learning. We need to store the last $n$ states, but the per-step computation is small and uniform for n-step, just as for 1-step.


### Example -- $n$-step SARSA update

Consider our simple 2D navigation task, in which we do not know the probability transitions nor the rewards. Initially, the reinforcement learning algorithm will be required to search randomly until it finds a reward. Propagating this reward back n-steps will be helpful.

Imagine the first episode consisting of the following (very lucky!) trace:

```{code-cell} ipython3
---
tags: [remove-input]
---
from gridworld import GridWorld
from tests.print_gridworld_sequence import draw_action_sequence


draw_action_sequence((0,0), [GridWorld.UP, GridWorld.UP, GridWorld.RIGHT, GridWorld.RIGHT, GridWorld.DOWN, GridWorld.UP, GridWorld.RIGHT])
```

Assuming $Q(s,a)=0$ for all $s$ and $a$, if we traverse the episode the labelled episode, what will our Q-function look like for a 5-step update with $\alpha=0.5$ and $\gamma=0.9$?

For the first $n-1$ steps of the episode, no update is made to the Q-values, but rewards and states are stored for future processing.

On step 5, we reach the end of our n-step window, and we start to update values becasue $|rs|=n$. We calculate $G$ from $rs$ and update $Q(ss_0, as_0)$, and then similarly for the 6$^{th}$ step:


$
\begin{array}{llll}
& \text{Step 5} & rs = \langle 0, 0, 0, 0, 0\rangle\\
&       & G = \gamma^0 rs_0 + \ldots + \gamma^4 rs_4 + \gamma^5 Q((2,1), Up) = 0\\
&       & Q((0,0), Up) = 0\\[1mm]
& \text{Step 6} & rs = \langle 0, 0, 0, 0, 0\rangle\\
&           & G = \gamma^0 rs_0 + \ldots + \gamma^4 rs_4 + \gamma^5 Q((2,2), Right) = 0\\
&           & Q((0,1), Up) = 0 + 0.5[0 - 0] = 0
\end{array}
$

We can see that there are no rewards in the first five steps, so the Q-values first two state-action pairs in the sequence $Q((0,0), Up)$ and $Q((0,1), Up)$ both remain 0, their original values.

When we reach step 7, however,  we receive a reward. We can then update the Q-value $Q((0,2), Right)$ of the state-action pair at step 3:

$
\begin{array}{llll}
& \text{Step 7} & rs = \langle 0, 0, 0, 0, 1\rangle\\
&      & G = \gamma^0 rs_0 + \ldots + \gamma^4 rs_4 + \gamma^5 Q((3,2), Terminate) = 0.9^4 \cdot 1 = 0.6561\\
&      & Q((0,2), Right) = 0 + 0.5[0.6561 - 0] = 0.32805
\end{array}
$

From this point, $s$ is a terminal state,, so we no longer select and execute actions, nor store the rewards, actions, and states. However, we continue to update the steps in the episode, leaving off $Q(ss_5, as_5)$ because there is no future discount reward beyond the end of the episode:

$
\begin{array}{llll}
& \text{Step 8} & rs = \langle 0, 0, 0, 1\rangle\\
&       & G = \gamma^0 rs_0 + \ldots + \gamma^4 rs_3 = 0.9^3 \cdot 1 = 0.729\\
&       & Q((1,2), Right) = 0 + 0.5[0.729 - 0] = 0.3645\\[1mm]
& \text{Step 9} & rs = \langle 0, 0, 1\rangle\\
&       & G = \gamma^0 rs_0 + \ldots + \gamma^3 rs_2 = 0.9^2 \cdot 1 = 0.81\\
&       & Q((2,2), Down) = 0 + 0.5[0.81 - 0] = 0.405\\[1mm]
& \text{Step 10} & rs = \langle 0, 1\rangle\\
&       & G = \gamma^0 rs_0 + \ldots + \gamma^2 rs_1 = 0.9^1 \cdot 1 = 0.9 = 0.9\\
&       & Q((2,1), Up) = 0 + 0.5[0.9 - 0] = 0.45\\[1mm]
& \text{Step 11} & rs = \langle 1\rangle\\
&      & G = \gamma^0 rs_0 = 0.9^0 \cdot 1 = 1\\
&      & Q((2,2, Right) = 0 + 0.5[1  \cdot 1 - 0] = 0.5
\end{array}
$

At this point, there are no further states left to update, so the inner loop terminates, and we start a new episode.

## Implementation

Below is a Python implementation of n-step temporal difference learning. We first implement an abstract superclass ```NStepReinforcementLearner```, which contains most of the code we need, except the part that determines the value of state $s'$ in the Q-function update, which is left to the subclass so we can support both n-step Q-learning and n-step SARSA:

```{code-cell} ipython3
:load: ../python_code/n_step_reinforcement_learner.py
```

We inherit from this class to implement the n-step Q-learning algorithm:

```{code-cell} ipython3
:load: ../python_code/n_step_qlearning.py
```



### Example -- 1-step Q-learning vs 5-step Q-learning

Using the interactive graphic below, we compare 1-step vs. 5-step Q-learning over the first 20 episodes of learning in the GridWorld task. Using 1-step Q-learning, reaching the reward only informs the state from which it is reached in the first episode; whereas for 5-step Q-learning, it informs the previous five steps. Then, in the 2nd episode, if any action reaches a state that has been visited, it can access the TD-estimate for that state. There are five such states in 5-step Q-learning; and just one in 1-step Q-learning. On all subsequent iterations, there is more chance of encountering a state with a TD estimate and those estimates are better informed. The end result is that the estimates 'spread' throughout the Q-table more quickly:

```{div} full-width
<div id="container" markdown="1" style="text-align: center;">
    <img id="1_step_vs_5_step_qlearning" src="https://gibberblot.github.io/rl-notes/gifs/1_step_vs_5_step_qlearning.gif" width="900" height="400" rel:auto_play="0">
    <gif-player id="1_step_vs_5_step_qlearning" width="900"></gif-player>
</div>
<p>
```


## Values of *n*

**Can we  just increase $n$ to be infinity so that we get the reward for the entire trace?** Doing this is the same as [Monte-Carlo reinforcement learning](sec:model-free:monte-carlo-learning), as we would no longer use TD estimates in the update rule. As we have seen, this leads to more variance in the learning.

**What is the best value for $n$ then?** Unfortunately, there is no theoretically best value for $n$. It depends on the particular application and reward function that is being trained. In practice, it seems that values of $n$ around 4-8 give good updates because we can easily assign credit to each of the 4-8 actions; that is, we can tell whether the 4-8 actions in the lookahead contributed to the score, because we use the TD estimates. 

## Summary

- n-step reinforcement learning propagates rewards back $n$ steps to help with learning.

- It is conceptually quite simple, but the implementation requires a lot of 'book-keeping'.

- Choosing a value of $n$ for a domain requires experimentation and intuition.

## Further Reading

-   Chapter 7 of [Introduction to Reinforcement Learning, Sutton and
    Barto](https://incompleteideas.net/book/the-book-2nd.html)

