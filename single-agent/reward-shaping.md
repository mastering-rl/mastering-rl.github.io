
## Reward shaping

**Learning Outcomes**

1.  Explain how reward shaping can be used to help model-free
    reinforcement learning methods to converge

2.  Manually apply reward shaping for a given potential function to
    solve small-scale MDP problems

3.  Design and implement potential functions to solve medium-scale MDP
    problems automatically

4.  Compare and contrast reward shaping with Q-function initialisation

Reinforcement Learning -- Some Weaknesses

In the previous lectures, we looked at fundamental temporal difference
(TD) methods for reinforcement learning. As noted, these two methods
have some weaknesses in this basic format:

1.  Unlike Monte-Carlo methods, which reach a reward and then
    backpropagate this reward, TD methods use bootstrapping (they
    estimate the future discounted reward using $Q(s,a)$), which means
    that for problems with spare rewards, it can take a long time to for
    rewards to propagate throughout a Q-function.

2.  Both methods estimate a Q-function $Q(s,a)$, and the simplest way to
    model this is via a Q-table. However, this requires us to maintain a
    table of size $|A| \times |S|$, which is prohibitively large for any
    non-trivial problem.

3.  Using a Q-table requires that we visit every reachable state many
    times and apply every action many times to get a good estimate of
    $Q(s,a)$. Thus, if we never visit a state $s$, we have no estimate
    of $Q(s,a)$, even if we have visited states that are very similar to
    $s$.

4.  Rewards can be sparse, meaning that there are few state/actions that
    lead to non-zero rewards. This is problematic because initially,
    reinforcement learning algorithms behave entirely randomly and will
    struggle to find good rewards. Remember the Freeway demo from the
    previous lecture?

Reinforcement Learning -- Some Improvements

To get around these limitations, we are going to look at three simple
approaches that can improve temporal difference methods:

1.  *$n$-step temporal difference learning*: Monte Carlo techniques
    execute entire traces and then backpropagate the reward, while basic
    TD methods only look at the reward in the next step, estimating the
    future wards. $n$-step methods instead look $n$ steps ahead for the
    reward before updating the reward, and then estimate the remainder.
    *Last lecture!*

2.  *Approximate methods*: Instead of calculating an exact Q-function,
    we approximate it using simple methods that both eliminate the need
    for a large Q-table (therefore the methods scale better), and also
    allowing use to provide reasonable estimates of $Q(s,a)$ *even if we
    have not applied action $a$ in state $s$ previously*. *This
    lecture!*

3.  *Reward shaping and Q-Value Initialisation*: If rewards are sparse,
    we can modify/augment our reward function to reward behaviour that
    we think moves us closer to the solution, or we can guess the
    optimal Q-function and initial $Q(s,a)$ to be this. *This lecture!*

### Overview

What is reward shaping? In many applications, you will have some idea of
what a good solution should look like. For example, in our simple
navigation task, it is clear that moving towards the reward of +1 and
away from the reward of -1 are likely to be good solutions. 

Can we then speed up learning and/or improve our final solution by nudging our
reinforcement learner towards this behaviour?

The answer is: Yes! We can modify our reinforcement learning algorithm
slightly to give the algorithm some information to help, while also guaranteeing optimality.

This information is known as *domain knowledge* --- that is, stuff about the domain
that the human modeller knows about while constructing the model to be
solved.

Exercise: Freeway What would be a good reward for the Freeway game to
learn how to get the chicken across the freeway?

![image](../images/freeway_screenshot)

Exercise: Gridworld What would be a good reward for the GridWorld
example?

![image](../images/MDP-GridWorld)

Reward Shaping In TD learning methods, we update a Q-function when a
reward is received. E.g, for 1-step Q-learning:

$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$

The approach to reward shaping is not to modify the reward function, but
to just give additional reward for some moves:

$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \underbrace{F(s,s')}_{\text{additional reward}} + \gamma \max_{a'} Q(s',a') - Q(s,a)]$

The function $F$ provides *heuristic* domain knowledge to the problem.

$r + F(s,s')$ is the *shaped reward* for an action.

$G^{\Phi} = \sum_{i=0}^{\infty} \gamma^i (r_i + F(s_i,s_{i+1}))$ is the
shaped reward for the entire trace.

Potential-based Reward Shaping *Potential-based* reward shaping is a
particular type of reward shaping with nice theoretical guarantees. In
potential-based reward shaping, $F$ is of the form:
$$F(s,s') = \gamma \Phi(s') - \Phi(s)$$ We call $\Phi$ the *potential
function* and $\Phi(s)$ is the potential of state $s$.

**Theoretical guarantee**: this will still converge to the optimal
policy.

**But!** It may either increase or decrease the time taken to learn. A
well-designed potential function will descrease the time.

Example: GridWorld For Grid World, we can use 1 over Manhattan distance
to define the potential function:
$$\Phi(s) = \frac{1}{|x(g) - x(s)| + |y(g) - y(s)|}$$ in which $x(s$)
and $y(s)$ return the $x$ and $y$ coordinates of the agent respectively,
and $g$ is the goal state.

Even on the very first iteration, a greedy policy, such as
$\epsilon$-greedy, will feedback those states closer to the +1 reward.
From state (1,2) with $\gamma=0.9$ if we go East, we get:

  ------------------- ----- ---------------------------------------
  $F((1,2), (2,2))$   $=$   $\gamma\Phi(2,2) - \Phi(1,2)$
                      $=$   $0.9 \cdot \frac{1}{1} - \frac{1}{2}$
                      $=$   $0.4$
  ------------------- ----- ---------------------------------------

Reward Shaping Example: GridWorld (continued)

We can compare the Q-values for these states for the four different
possible moves that could have been taken from (1,2), using and
$\alpha=0.5$ and $\gamma=0.9$:

  **Action**    $r$                 $F(s,s')$                 $\gamma \max_{a'}Q(s',a')$   New $Q(s,a)$
  ------------ ----- --------------------------------------- ---------------------------- --------------
  North          0     $0.9\frac{1}{2} - \frac{1}{2} = 0$                 0                     0
  South          0     $0.9\frac{1}{2} - \frac{1}{2} = 0$                 0                     0
  East           0    $0.9\frac{1}{1} - \frac{1}{2} = 0.4$                0                    0.4
  West           0    $0.9\frac{1}{3} - \frac{1}{2} = -0.2$               0                   $-0.1$

Thus, we can see that our potential reward function rewards actions that
go towards the goal and penalises actions that go away from the goal.
Recall that state (1,2) is in the top row, so action North just leaves
us in state (1,2) and South similarly because we cannot go into the
wall.

But! It will not always work. Compare states (0,0) and (0,1). Our
potential function will reward (0,1) because it is closer to the goal,
but we know from the lecture on MDPs that (0,0) is a higher value state
than (0,1). This is because our reward function does not consider the
negative reward.

In practice, it is non-trivial to derive a perfect reward function. If
we could, we would not need to even use reinforcement learning -- we
could just do a greedy search over the reward function.

Q-function initialisation An approach related to reward shaping is
*Q-function initialisation*. Recall that TD learning methods can start
at any arbitrary Q-function. The closer our Q-function is to the optimal
Q-function, the quicker it will converge.

Imagine if we happened to initialise our Q-function to the optimal
Q-function. It would converge in one step!

Q-function initialisation is similar to reward shaping: we use
heuristics to assign higher values to 'better' states. If we just define
$\Phi(s) = V(s)$, then they are equivalent. In fact, if our potential
function is *static* (the definition does not change during learning),
then Q-function initialisation and reward shaping are equivalent[^1].

Q-function Initialisation Example : GridWorld Using the idea of inverse
Manhattan distance, we can define an initial Q-function as follows for
state (1,2):

  ------------------- ----- ----------------------------- ----- -----------
  $Q((1,2), North)$   $=$   $\frac{1}{2} - \frac{1}{2}$   $=$   $0$
  $Q((1,2), South)$   $=$   $\frac{1}{2} - \frac{1}{2}$   $=$   $0$
  $Q((1,2), East)$    $=$   $\frac{1}{1} - \frac{1}{2}$   $=$   $0.5$
  $Q((1,2), West)$    $=$   $\frac{1}{3} - \frac{1}{2}$   $=$   $-0.16^*$
  ------------------- ----- ----------------------------- ----- -----------

Once we start learning over episodes, we will select those actions with
a higher heuristic value, and also we are already closer to the optimal
Q-function, so will will converge faster.

Conclusion
==========

 {#section-3 .unnumbered}

Summary

1.  We can scale reinforcement learning by approximating Q-functions,
    rather than storing complete Q-tables.

2.  Using simple linear methods in which we select features and learn
    weights are effective and guarantee convergence.

3.  Using neural networks offer alternatives in which we do not need to
    select features, but require a lot of training data and have not
    convergence guarantees.

Some tips for reinforcement learning in the group project

1.  Linear Q-function approximation should work well if the features are
    not strongly dependent on the random initial state.

2.  I would be surprised if using deep Q networks for function
    approximation was effective because deep neural nets are data hungry
    and generating enough training data could require weeks or months of
    computation.

3.  Design good reward shaping structures will help, but keep them
    simple.

4.  For some advanced techniques, read Sutton and Barto; in particular
    stuff on: experience replay, approximate methods (even for value
    iteration!), and generalised policy iteration.

Reading

-   Chapter 9 (Approximate Solution Methods) of *Introduction to
    Reinforcement Learning* \[*Sutton and Barto*\]

    Available at:

    <https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html>

-   *Playing Atari with Deep Reinforcement Learning* from DeepMind.

    Available at:

    <https://arxiv.org/pdf/1312.5602v1.pdf>

-   Before AlphaGo there was TD-gammon, which was the first paper to
    combine reinforcement learning and neural networks:

    <http://www.aaai.org/Papers/Symposia/Fall/1993/FS-93-02/FS93-02-003.pdf>

[^1]: Wiewiora: ?Potential-based shaping and Q-value initialization are
    equivalent.? (JAIR, 2003)
