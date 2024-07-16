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

(sec:policy-based:actor-critic)=
# Actor-critic methods

```{contents}
:local:
:depth: 2
```

```{admonition}  Learning outcomes
The learning outcomes of this chapter are:

1. Apply actor-critic methods to solve small-scale MDP problems manually and program actor critic algorithms to solve medium-scale MDP problems automatically.
2. Compare and contrast actor-critic methods with policy gradient methods like REINFORCE and value-based reinforcement learning.
```

````{margin}
```{admonition} Video byte: Actor-critic methods
<iframe width="248" height="141" src="https://www.youtube.com/embed/51DmzTJrEgk?start=1053" title="Policy gradients" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

The sample efficiency problem in REINFORCE leads to issues with policy convergence. As with Monte-Carlo simulation, the high variance in the cumulative rewards $G$ over episodes leads to instability.

**Actor critic** methods aim to mitigate this problem. The idea is that instead of learning a value function or a policy, we learn both. The policy is called the **actor** and the  value function is called the **critic**. The primary idea is that the actor produces actions, and as in [temporal difference learning](sec:model-free:td-learning), the  value function (the critic) provides feedback or "criticism" about these actions as a way of bootstrapping.

## Q Actor-Critic

The **Q Actor Critic** algorithm uses a Q-function as the critic.

````{margin}
```{admonition} Video byte: Q actor critic
<iframe width="248" height="141" src="https://www.youtube.com/embed/51DmzTJrEgk?start=1313" title="Policy gradients" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````


```{prf:algorithm} Q Actor Critic
:label: algorithm:q-actor-critic

$
\begin{array}{l}
\alginput:\ \text{MDP}\ M = \langle S, s_0, A, P_a(s' \mid s), r(s,a,s')\rangle\\
\alginput:\ \text{A differentiable actor policy}\ \pi_{\theta}(s,a)\\
\alginput:\ \text{A differentiable critic Q-function}\ Q_w(s,a)\\
\algoutput:\ \text{Policy}\ \pi_{\theta}(s,a) \\[2mm]
\text{Initialise actor}\ \pi\ \text{parameters}\ \theta\ \text{and critic parameters}\ w\ \text{arbitrarily}\\[2mm]
\algrepeat\ \text{(for each episode}\ e\ \text{)}\\
\quad\quad s \leftarrow\ \text{the first state in episode}\ e\\
\quad\quad \text{Select action}\ a \sim \pi_\theta(s)\\
\quad\quad \algrepeat\ \text{(for each step in episode e)}\\
\quad\quad\quad\quad \text{Execute action}\ a\ \text{in state}\ s\\
\quad\quad\quad\quad \text{Observe reward}\ r\ \text{and new state}\ s'\\
\quad\quad\quad\quad \text{Select action}\ a' \sim \pi_\theta(s')\\
\quad\quad\quad\quad \delta \leftarrow r + \gamma \cdot  Q_w(s',a') - Q_w(s,a)\\
\quad\quad\quad\quad w  \leftarrow w + \alpha_w \cdot \delta \cdot \nabla Q_w(s,a)\\
\quad\quad\quad\quad \theta \leftarrow \theta + \alpha_{\theta} \cdot \delta \cdot \nabla \textrm{ln}\ \pi_{\theta}(s,a)\\
\quad\quad\quad\quad s \leftarrow s'; a \leftarrow a'\\
\quad\quad \alguntil\ s\ \text{is the last state of episode}\ e\ \text{(a terminal state)}\\
\alguntil\ \pi_{\theta}\ \text{converges}
\end{array}
$
```

Note that we have two different learning rates $\alpha_w$ and $\alpha_{\theta}$ for the Q-function and policy respectively.

Let's analyse the key parts in more detail. The line that updates $\delta$ is the same as the $\delta$ calculation in SARSA:  it is temporal difference value for executing action $a$ in state $s$, with the estimate of the future discount reward being $Q_w(s',a')$.

Once the $\delta$ value is calculated, we update both the actor and the critic.  The weights of the critic $Q_w$ are updated by following the gradient $\nabla Q_w(s,a)$ of the critic Q-function at $s,a$, and then the parameters of the actor $\theta$ are updated the same way as in REINFORCE, except that the value of $\delta$ uses the temporal difference estimate based on $Q_w(s,a)$ instead of using $G$.

So, this simulataneously learns the policy (actor) $\pi_{\theta}$ and a critic (Q-function) $Q_w$, but the critic is learnt only to provide the temporal difference update, not to extract the policy.

**But wait!** Didn't we say early that the weakness of value-based methods was that they could not extend to continuous action spaces? Haven't we now gone backwards by including a Q-function? Why not just use the Q-function directly?

````{margin}
```{admonition} Video byte: Summary of policy gradients
<iframe width="248" height="141" src="https://www.youtube.com/embed/51DmzTJrEgk?start=1610" title="Policy gradients" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
````

The reason the actor critic methods still work like this is because the actor policy $\pi_{\theta}$ selects actions for us, while the critic $Q_w(s,a)$ is only ever used to calculate the temporal difference estimate for an already selected action. We do not have to iterate over the critic Q-function to select actions, so we do not have to iterate over the set of actions -- we just use the policy. As such, this will still extend to continuous and large state spaces and be more efficient for large action space.

### Implementation

To implement the Q Actor Critic framework, we first create a new base class called `ActorCritic`, which can be used as a base for other types of actor critic methods, such as **advantage actor critics**, which we will not discuss here.

The `ActorCritic` class is an abstract class that looks similar to that of `QLearning`, except that we update both the actor and the critic:

```{code-cell} ipython3
:load: "../python_code/actor_critic.py"

```

Note from the code above that we use the actor (the policy) to choose an action, and then update both the critic and the actor. In this particular implementation, we batch update the actor policy at the end of the episode.

Next, we have to instantite the  `ActorCritic` class as a `QActorCritic` class to implement the `update_actor` and `update_critic` classes:

```{code-cell} ipython3
:load: "../python_code/q_actor_critic.py"

```

Now, we can create a policy and Q-function using any differentiable policy and any Q-function implementation. We choose `DeepNeuralNetworkPolicy` and `DeepQFunction` with `QLearning` updates. 

```{code-cell} ipython3
:load: "../python_code/tests/deep_q_actor_critic_run.py"
```

We can see that the actor critic agent has learnt both a policy that is very good, but also a Q-function critic that could be used as a policy (because our action space is finite and very small). In a continuous state space, we would be able to learn the critic, but not use it as a policy because we cannot iterate over the possible actions.


## Takeaways

```{admonition} Takeaways

- Like REINFORCE, **actor-critic** methods are policy-gradient based, so directly learn a policy instead of first learning a value function or Q-function.

- Actor-critic methods also learn a **value function or Q-function** to reduce the variance in the cumulative rewards.
```
