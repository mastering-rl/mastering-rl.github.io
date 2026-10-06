# What is reinforcement learning?


Reinforcement learning is **learning from experience**.

Reinforcement learning is a branch of machine learning in which **agents** learn to make **sequential decisions** in an environment, guided by a set of rewards and penalties.

It differs from traditional supervised machine learning in a few ways, primarily:

1. **Experience instead of labels**: In traditional supervised machine learning, we need to provide examples that contain inputs and labelled outputs. In reinforcement learning, the "labels" are provided by the environment when the agent interacts with the environment. These "labels" are called **rewards**. Rewards can be **positive**, which means our agent will try to learn behaviour that leads to positive rewards, or rewards can be **negative**, which means our agent will try to learn behaviour that leads away from negative rewards. The **aim** is to learn behaviour that increases the cumulative reward over a number of decisions.

2. **Decisions are sequential**: Reinforcement learning is **sequential decision making**. This means that our agent needs to make a series of decisions over time, which each decision affecting future outcomes. Feedback, in the form of positive/negative rewards, is received at each step (although the reward may be zero at most steps), but in most situations, simply maximising feedback at each step is not optimal --- our agent needs to consider how each action affects the future.


## Part II: Value-based methods

In Part II of these notes, we introduce  *Markov Decision Processes* (MDPs). MDPs provide a flexible framework that allows us to model many different reinforcement learning problems. 

We look at *value-based* techniques for solving MDPs, which solve MDPs by estimating the value of different states of the environment, and then moving towards high-value states. There are two broad types of value-based method: 1. *model-based* techniques, where the entire MDP model is known to us; and 2. *model-free* techniques, which are flexible enough that when some information is not provide explicitly, but can be sampled enough times, we can still learn good behaviour. We will look at foundational techniques, some more advanced techniques, and will pay particular attention to how to scale reinforcement learning so we can use it solve real problems.

## Part III: Policy-based methods

In Part III of these notes, we introduce *policy-based methods*, which are techniques that learn a policy directly without trying to calculate the value of states. Such techniques are important for e.g. applications with an infinite state space.

## Part IV: Scaling

In Part IV of these notes, we explore techniques for scaling reinforcement learning to large and complex problems, including function approximation, reward shaping, n-step methods, experience replay, and abstraction.

## Part V: Multi-agent reinforcement learning

In Part V of these notes, we look at *multi-agent MDPs* (sometimes called *games*), in which there are multiple (possibly adversarial) agents in a problem, and we need to plan our decisions while also considering what the other actors in the environment will do. Again, we look at both model-based and model-free techniques.

## Part VI: Appendix

In the appendix, we provide foundational background material, such as an introduction to basic probability theory.

