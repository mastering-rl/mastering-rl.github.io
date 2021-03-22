COMP90054: Reinforcement Learning
=====

Learning outcomes and intro

In the University of Melbourne subject COMP90054, the first part of the subject detals with *classical planning and search*. Classical planning tools can produce solutions quickly in large search spaces, but they make the following assumptions about the problem:

1.  Actions are all deterministic
2.   Environments change only as the result of an action
3.   Perfect knowledge (omniscience)
4.  Single actor (omnipotence).

In this book, we look at methods to relax a few of these assumptions, in particular, assumptions 1 and 4. 

In Part I of this book, we introduce  *Markov Decision Processes* (MDPs). MDPs allow us to model problems  in which the outcomes of actions are probabilistic; that is, we do not know the outcome beforehand, but we know there is some probability distribution over a set of possible outcomes. We look at *model-based* techniques, where these probabilistic outcomes are given to use, and *model-free* techniques, which are flexible enough the probabilitics are unknown, but we can sample enough times that we can still learn good behaviour.

In Part II of this book, we look at *game theoretical models*, in which there are multiple (possibly adversarial) actors in a problem, and we need to plan our actions while also considering what the other actors in the environment will do. Again, we look at both model-based and model-free techniques.
