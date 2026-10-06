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

```{code-cell}
:tags: [remove-input]

import random
random.seed(1028)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
```

(sec:experience-replay)=
# Experience replay

```{admonition}  Learning outcomes
The learning outcomes of this chapter are:

1. Explain the advantages and disadvantages of experience replay approaches.

2. Design and implement experience replay algorithms to solve medium-scale MDP problems.
    
3. Identify and explain the assumptions behind experience replay approaches.
```

Experience replay is a reinforcement learning technique in which, instead of executing a transition and then updating a Q-function immediately, such as in Q-learning and SARSA, instead, the **experiences** (state, reward, next state) are stored in a buffer of transitions. Then, intermittently, a set of experiences from the buffer are randomly sampled and **replayed** to update the Q-function.

## Why use experience replay?

**Why re-use old traces?** This seems like a strange idea. The reason is really because of the **Independent and Identically Distributed (i.i.d.)** assumption of machine learning. 

```{admonition} Definition -- Independent and Identical Distribution (i.i.d.)
Machine learning algorithms typically assume that training data is:

1. **Independent**: Each data point is independent of the others; for example, in a set of images of skin lesions, the same lesion is not present more than once, even if taken at a different time.
2. **Identically distributed**: All data points are from the same probability distribution.
```

In algorithms such as [Q-learning](sec:model-free:q-learning), the assumption of independence is broken due to the temporal correlation between experiences/transitions when using [function approximation](sec:qfunction-approximation). That is, if we execute a transition, update, execute another transition, update, then the two transitions are not independent. Both transitions come from the same episode that will receive the same reward. By storing experiences and randomly sampling, we can **approximate** the i.i.d. assumption.

Further to this, experience replay can **improve data efficiency** by sampling past experiences multiple times.

## Intuition

There are two parts to the way the experience replay is usually implemented: the **target network** and the **replay buffer**.

### Target network

The technique that is often paired with experience replay, is the use of a **target Q-function** (often called a target network), which is a separate Q-function just used for calculating the amount to update the Q-function. We keep both a **policy Q-function** $Q_{policy}$, which is the one used to generate actions for the experiences, and the target Q-function $Q_{target}$, using the target Q-function only to estimate future rewards.

With this in mind, the Q-function update becomes:

$$\delta \leftarrow r + \gamma \cdot \overbrace{\max_{a'} Q_{target}(s',a')}^{\text{target estimate}}- \overbrace{Q_{policy}(s,a)}^{\text{policy estimate}}$$

**Why do this?** Doesn't this just basically copy values over so now both functions estimate the same thing? 

Yes! But the trick is that we only update the target Q-function, $Q_{target}$ intermittently at fixed intervals, or update it slowly compared to the policy. For example, we update the target function every 1000 steps by copying over the parameters to the target Q-function, or we update it every step, but with a smoothing factor.

The issue with using [Q-function approximation techniques](sec:qfunction-approximation) such as [deep Q-learning](sec:function-approximation:deep-Q-learning) is that when a Q-function is updated, this actually updates many state-action values at the same time, because of the approximation techniques that help this scale. So, this means that if we do an update of the form $\delta \leftarrow r + \gamma \cdot \max_{a'} Q_{policy}(s',a') - Q_{policy}(s,a)$, where $Q_{policy}$ is both used to estimate future reward using $\max_{a'} Q_{policy}(s',a')$, then the actual value of $\max_{a'} Q_{policy}(s',a')$ may in fact be a **different value after the update**, so the estimate can be quite wrong. This leads to instability in the learning.

An analogy to this would be if you were learning how to play a new game, and your friend with experience was giving you feedback. You can see when you are getting points (the reward), but your friend's feedback tells you whether each move would be good in the future (the $\max_{a'} Q(s',a')$). Now imagine that your friend changes their mind about how to play the game each time you make a move! It would be difficult to determine how well you were playing. By updating the target network periodically, effectively, your friend gives you consistent feedback for a while, but then improves the feedback periodically, leading to more stable learning.
### Replay buffer

The intuition behind experience replay is quite straightforward: as we execute transitions, we store them in a **replay buffer** for later use. Once we have enough transitions (a pre-defined limit), we randomly sample the buffer and update our Q-function based on that sample. The size of the buffer is typically bounded, and uses a first-in, first-out (FIFO) queue, where the oldest experiences are removed to make way for new ones.

But doesn't replaying old samples mean that we just learn the same behaviour multiple times, both not learning anything new, and breaking the i.i.d. assumption? 

No! Even when we replay a sample that we have used before, we reuse the transition (state, reward, action, next state, done), but we do not reuse the estimates for the state and next state values. 

$$\delta \leftarrow \underbrace{r}_{\text{reused}} + \gamma \cdot \overbrace{\max_{a'} Q_{target}(\underbrace{s',a'}_{\text{reused}})}^{\text{new estimate}}- \overbrace{Q_{policy}(\underbrace{s,a}_{\text{reused}})}^{\text{new estimate}}$$

Given that we **know** the reward, state, and next state from the transitions, and we can provide new estimates for the values of the state and next state, replaying an old transition is **as if** we had only just sampled it.


## Experience replay

The change from standard Q-learning to experience replay is reasonably straightforward. First, we keep a replay buffer of transitions, $\mathcal{D}$,  sample these and replay them back for the update. Second, we update using target network $Q_{target}$ to estimate the future discounted rewards; that is, $\max_{a'} Q_{target}(s',a')$.

:::{prf:algorithm} Experience Replay
:label: algorithm:experience-replay
$
\begin{array}{l}
\alginput:\  \text{MDP}\ M = \langle S, s_0, A, P_a(s' \mid s), r(s,a,s')\rangle, \\
\quad\quad\quad \text{Replay Buffer}\ \mathcal{D}, \text{Batch Size}\ B, \text{Update Period}\ U, \text{smoothing factor}\ \tau\\
\algoutput:\ \text{Q-functions}\ Q_{target}, Q_{policy}\\[2mm]
\text{Initialise}\ Q_{policy}\ \text{arbitrarily}\\
\text{Initialise}\ Q_{target}\ \text{with}\ Q_{policy}\\
\text{Initialise Replay Buffer}\ \mathcal{D}\ \text{with capacity}\ N\\[2mm]
\algrepeat\ \text{(for each episode}\ e \text{)}\\
\quad\quad s \leftarrow\ \text{initial state of episode}\ e\\
\quad\quad \algrepeat\ \text{(for each step in episode}\ e \text{)}\\
\quad\quad\quad\quad \text{Select action}\ a\ \text{using policy derived from}\ Q_{policy}\ \text{(e.g., }\epsilon\text{-greedy)}\\
\quad\quad\quad\quad \text{Execute action}\ a\ \text{in state}\ s\\
\quad\quad\quad\quad \text{Observe reward}\ r\ \text{and new state}\ s'\\
\quad\quad\quad\quad \text{Store transition}\ (s, a, s', r)\ \text{in Replay Buffer}\ \mathcal{D}\\
\quad\quad\quad\quad \text{If}\ \text{Replay Buffer}\ \mathcal{D}\ \text{contains enough samples:}\\
\quad\quad\quad\quad\quad\quad \text{Sample random batch of transitions of size}\ B\ \text{from}\ \mathcal{D}\\
\quad\quad\quad\quad\quad\quad \text{For each transition}\ (s, a, s', r)\ \text{in batch:}\\
\quad\quad\quad\quad\quad\quad\quad\quad \delta \leftarrow r + \gamma \cdot \max_{a'} Q_{target}(s',a') - Q_{policy}(s,a)\\
\quad\quad\quad\quad\quad\quad\quad\quad Q_{policy}(s,a) \leftarrow Q_{policy}(s,a) + \alpha \cdot \delta\\
\quad\quad\quad\quad \text{Every}\ U\ \text{steps:}\\
\quad\quad\quad\quad\quad\quad \text{Update}\ Q_{target}\ \text{with}\ Q_{policy}\ \text{with smoothing factor}\ \tau\\
\quad\quad \alguntil\ s\ \text{is a terminal state}\\
\alguntil\ Q\ \text{converges}
\end{array}
$
:::


In the algorithm, note that we intermittently (every $U$ steps) update $Q_{target}$ with $Q_{policy}$, but we can do so with a so-called **soft update**. A **hard update** would simply copy the parameters of $Q_{policy}$ and put them in $Q_{target}$, while a soft update only "nudges" the parameters of $Q_{target}$ towards $Q_{policy}$. 

Soft updates help to provide a more stable learning process. Just like using a single Q-function leads to instability, abrupt changes in $Q_{target}$ can make learning unstable. Updating more gradually can help with this.

This can be implemented by taking a weighted average of each parameter. For example:

$$\theta_{target} \leftarrow \tau \theta_{policy} + (1 - \tau) \theta_{target}$$

where $\theta_{target}$ are the parameters for e.g. a [deep Q-function](sec:function-approximation:deep-Q-learning) of the target Q-function, $\theta_{policy}$ are the parameters for the policy deep Q-function, and $\tau \in [0,1]$ is the update rate. A value of $\tau=1.0$ gives us a hard update, while $\tau < 1.0$ is a soft update.

Typically, we either update every $U > 1$ steps (e.g. $U=100$) using a hard update ($\tau=1.0$), or we update every step ($U=1$) using a value of $\tau$ in the range $[0.001,0.05]$. If we do both a soft update and every $U >> 1$ steps, the target network becomes out of date quickly.

## Implementation

Below is a Python implementation for experience replay. First, we need to set up the replay buffer, which is defined in the class ``ReplayBuffer``. This keeps experiences of the form ``("state", "action", "next_state", "reward", "done")``, and allows us to sample random batches.

The ``ExperienceReplayLearner`` class looks quite similar to standard Q-learning, except for: (1) storing and then sampling experiences for replay; and (2) doing a soft update every time ``step % self.update_period == 0``.


```{code-cell} ipython3
:load: ../mastering_rl/learners/experience_replay_learner.py

```

In our framework, the soft update is encapsulated in the implementation of the Q-function. For example, this is the update for a [deep Q-function](sec:function-approximation:deep-Q-learning) with parameters:

```{code-cell} ipython3
:tags: [remove-input]

    def soft_update(self, policy_qfunction, tau=0.1):
        target_dict = self.q_network.state_dict()
        policy_dict = policy_qfunction.q_network.state_dict()
        for parameter in policy_dict:
            target_dict[parameter] = policy_dict[parameter] * tau + target_dict[parameter] * (1 - tau)
        self.q_network.load_state_dict(target_dict)
```

Here is Python code to run this on the ``GridWorld`` example, with an update period of 1 step:

```{code-cell} ipython3
:load: ../mastering_rl/tests/_12_experience_replay/experience_replay_run.py

```



