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

(sec:policy-based:policy-gradients)=
# Policy gradient methods

As noted earlier, policy-based methods search for a policy directly, rather than searching for a value function and extracting a policy. In this section, we look at a model-free method that optimises a policy directly. It is similar to Q-learning and SARSA, but instead of updating a Q-function, it updates the parameters $\theta$ of a policy directly using gradient ascent.

## Overview

In policy gradient methods, we approximate the policy from the rewards and actions received in our episodes, similar to the way we do it with Q-learning. We can do this provided that the policy has two properties:

1. The policy is represented using some function that is *differentiable* with respect to its parameters. For a non-differentiable policy, we cannot calculate the gradient.
2. Typically, we want the policy to be *stochastic*. Recall from the section on [policies](sec:mdps:policies) that a stochastic policy specifies a *probability distribution* over actions, defining the probability with which each action should be chosen.

The goal of a policy gradient is to approximate the optimal policy $\pi_{\theta}(s, a)$ via gradient ascent on the expected return. Gradient ascent will find the best parameters $\theta$ for the particular MDP.

## Policy improvement

The goal of gradient ascent is to find weights of a policy function that maximises the expected return. This is done in an iterative by calculating the gradient from some data and updating the weights of the policy

The expected value of a policy $\pi_{\theta}$ with parameters $\theta$ is defined as:

$$J(\theta) = V_{\pi_{\theta}}(s_0)$$

where $V(\pi_{\theta})$ is the policy evaluation  using the policy $\pi_{\theta}$ and $s_0$ is the intial state. This expression is computationally expensive to calculate, so we use policy gradient algorithms to approximate it. These search for a local maximum in $J(\theta)$ by *ascending* the gradient of the policy with respect to the parameters $\theta$, using episodic samples.

:::{admonition} Definition -- Policy gradient
Given a policy objective $J(\theta)$, the *policy gradient* of $J$ with respect to $\theta$, written $\nabla_{\theta}J(\theta)$ is defined as:

$$
\nabla_{\theta}J(\theta) = \begin{pmatrix} \frac{\partial J(\theta)}{\partial \theta_1} \\ \vdots \\ \frac{\partial J(\theta)}{\partial \theta_n} \end{pmatrix}
$$
:::

If we want to follow the gradient towards the optimal $J(\theta)$ for our problem, we need to calculate the gradient and update the weights:

$$\theta \leftarrow \theta + \alpha \nabla\ J(\theta)$$

where $\alpha$ is a learning rate parameter that dictates how big the step in the direction of the gradient should be.

The question is: what is $\nabla J(\theta)$? The *policy gradient theorem* (see Sutton and Barto, Section 13.2) says that for any differentiable policy $\pi_{\theta}(s,a)$ that $\nabla J(\theta)$ is:

$$\nabla J(\theta) = \mathbb{E}[\nabla\ \textrm{ln} \pi_{\theta}(s, a) Q(s,a)]$$

The expression $\textrm{ln} \pi_{\theta}(s, a)$ tells us how to change the weights $\theta$ is we want to increase the log probability of selecting action $a$ in state $s$. If the quality of selecting action $a$ in $s$ is positive, we would increase that probability; otherwise it will descrease
Thus, this is the expected return of of taking action $\pi_{\theta}(s,a)$ multiplied by the gradient.

In these notes, we will not go into details about gradients or algorithms for solving them -- this is itself a large topic that is relevant outside of reinforcement learning. Instead, we will just give the intuition.

## REINFORCE

The REINFORCE algorithm is one algorithm for policy gradients.  We cannot calculate the gradient optimally because this is too computationally expensive -- we would need to solve for all possible trajectories in our model. In REINFORCE, we sample trajectories, similar to the way done in TD learning.

:::{admonition} Algorithm -- REINFORCE
**Input:** A differentiable policy $\pi_{\theta}(s,a)$, an MDP $M = \langle S, s_0, A, P_a(s' \mid s), r(s,a,s')\rangle$\
**Output:** Policy $\pi_{\theta}(s,a)$          

Repeat\
$\quad\quad$ Generate episode $(s_0, a_0, r_1, \ldots s_{T-1}, a_{T-1}, r_{T})$ by following $\pi_{\theta}$\
$\quad\quad$ For each $(s_t, a_t)$ in the episode\
$\quad\quad\quad\quad G \leftarrow \sum_{k=t+1}^{T} \gamma^{k-t-1} r_k$\
$\quad\quad\quad\quad \theta \leftarrow \theta + \alpha \gamma^{t} G\ \nabla\ \textrm{ln}\ \pi(s,a;\theta)$\
Until some time limit or until $\pi$ converges
:::

REINFORCE  generates an entire episode using Monte-Carlo simulation by following the policy so far; therefore, it generates better and better policies as $\pi$ is improved. It then steps through each action in the episode, a calculates $G$, the total future discounted reward of the trajectory. Using this reward, it calculates the gradient $\pi$ and multiples this in the direction of $G$.

Comparing to value-based techniques, we can see that REINFORCE (and other policy-based approaches)  do not evaluate each action in the policy improvement process. In policy improvement, we update the policy $\pi(s) \leftarrow \textrm{argmax}_{a \in A(s)} Q(s,a)$. Instead, actions are selected from the policy directly. 

This has the advantage that policy-based approaches can be when the action space or state space are  continuous; e.g. there are one or more actions with a parameter that takes a continuous value. This is because it uses the gradient instead of doing the policy improvement explicitly. For the same reason, policy-based approaches are often more efficient than value-based approaches when there are a large number of actions.

### Implementation

The implementation of REINFORCE takes a policy, but this policy must be differentiable. As such, we cannot use the tabular policy used in policy iteration. As we can see, unlike Q-learning and SARSA, the update happens only at the end of each episode.

```{code-cell} ipython3
:load: "../python_code/policy_gradient.py"
```

We need a differentiable policy so that we can calculate the gradient and update. To illustrate this from first principles, let's look at an implementation of a policy that uses logistic regression. This basic implementation supports environments with only two actions, but it is sufficient to illustrate the basics of policy gradient methods from first principles.

The policy inherits from [`StochasticPolicy`](sec:mdp:deterministic-vs-stochastic-policies), which means that the policy $\pi_{\theta}(s,a)$ returns the probability of action $a$ being executed in state $s$. 

```{code-cell} ipython3
:load: "../python_code/logistic_regression_policy.py"
```

```{code-cell} ipython3
:load: "../python_code/tests/logistic_regression_policy_gradient.py"
```

## Deep policy gradients



## Applications of policy gradient ascent

A great application of using policy gradient descent is learning how to control robotic arms to grasp unknown objects -- that is, objects that have not been seen before. The only input for the problem is the camera data:

<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/cXaic_k80uM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

What is interesting is how the policy is used to adjust the position and the gripper continuously as it tries to grasp objects in the testing phase.