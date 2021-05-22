(sec:single-agent:modelling-and-abstraction)=
# Modelling and abstraction for MDPs
## Learning outcomes

The learning outcomes of this chapter are:

1.  Describe modelling and abstraction strategies to scale MDP algorithms to problems.
    
2.  Apply modelling and abstraction strategies to non-trivial examples.


## Overview

As discussed through Part I of this book, often our reinforcement learning algorithms struggle with *scale*. That is, as the state and action spaces of problems get larger, the time taken to arrive at good policies becomes increases exponentially.

Two strategies that are useful if you can apply them are to : (1) apply more computational power to the problem so that you can run more episodes or assess more states; or (2) increase the efficiency of a simulator (if you are using a simulation-based technique).

However, we often already be using all the computational power we have access to, or maybe we are not working in a simulated environment at all. In these cases, one way is to improve the model of the problem that you are using by using a simplified version of the problem to explore. 

In this section, I give some tips that I have found useful when applying reinforcement learning techniques. Implementing these tips requires us to understand the domain that we are working in. An such implementation will only work on that domain or others that are very similar and will not generalise to other domains. However, these tips/strategies are general, and would work on many domains if we take the time to apply them.

## Abstraction

One solution is to define an abstracted version of the MDP and solve that, translating the policy back tot he original problem. Using abstraction, we take the original MDP model $M$ that we have, and define a new MDP model $M_A$ that is smaller than the original: it has fewer states and/or fewer actions than the original model. 

The process here is to answer the following:

1. What are the important decisions that I need to make in this problem to reach achieve some objective?
2. What information do I need to do this?
3. Can I translate the original MDP action and state spaces into smaller action and state spaces so that I can access this information and discard some information that is less important?

We then define an abstract model $M' = (S', s'_0, A', P', r', \gamma')$ that is an abstraction of $M$, such that there is some mapping $f_s(s) = s'$ where $s \in S$ and $s' \in S$, and similarly mappings for $A$, $P$, $r'$, and even $\gamma$ (the length of plans will be different if the action space is smaller). We solve for the new model $M'$ to receive a policy in $\pi' : S' \rightarrow A'$. We then need to define a mapping $g : A' \rightarrow A$ to map abstract actions to actions in the original model. Given this, we can then define the policy for $M$ as:

$$
\pi(s) = g(\pi'(f_s(s)))
$$

This means that $\pi(s)$ translates $s$ to $s' \in S'$ using $f_s$, then applies abstract policy $\pi'$ to get an abstraction actions $a' \in A'$, and then maps this to concrete action $a \in A$.

By "model", I don't just mean for model-based reinforcement learning: even in a model-free environment we can abstract the actions and states that we see, and modify the rewards, by putting a "layer" in between the environment and the reinforcement learning algorithm. 

For example, in these notes we use the GridWorld problem to illustrate various techniques because it is simple and intuitive. This is already an abstraction of a real navigation problem. In a real navigation problem, a robot would have to move in a continuous environment, with actions such as rotating and accelerating. However, our objective in the problem is to find the best route, not to move the wheels. For the purposes of finding the best route, we model the problem differently by dividing the state space into grids and mapping the actions into directions, as this is all the information that we need to solve the route-finding problem. If we wanted to actually move a robot, we would need a layer in-between that: (1) translates the real states into our abstracted grid coordinators and identifies when we are in a reward state, which is function $f_s$; and (2) translates our abstracted actions back into actions in the environment, which is function $g$. For example, the action `Right` may translate into rotating 90 degrees and then moving forward 2 metres.

Sometimes a problem may already be as abstract as it can be. For example, if we are learning a controller for a robotic arm in a manufacturing facility, the information set from the sensors and to the actuators may strongly influence the effect of higher-level actions, so cannot be abstracted away. However, in practice, there are usually abstractions we can use.

In reality, most of the techniques in this section are abstraction techniques of some sort, but some other tricks can help.

(sec:single-agent:modelling-and-abstraction:sub-goals)=
## Rewarding sub-goals

In some problems, there are final goals that need to be achieved, and we receive rewards for transitioning to states where those goals are achieved. One problem we saw in these notes, such as in the Freeway example, is when those rewards are a long way from the start state. As we often start with a random policy or value/Q-function, we spend a lot of time doing random simulation before we receive a reward by achieving the goal.

However, often these problems have *sub-goals* that we need to achieve on the way. If we do not achieve the sub-goals, we cannot achieve the primary goal. For example, in the Freeway game, we need to reach row 1, then row 2, etc., until we reach the other side. 

If we can identify these sub-goals, we can give "partial" rewards to the sub-goals on the way to the goal. In this case, we identify key states (or more accurately, key transitions) that are like sub-goals of our problem. In Freeway, assuming the reward for reaching the other side is 100 points, we could give 1 point for reaching row 1, 2 for teaching row 2, etc., and then 100 - (1 + 2 + ... + n) for reaching the other side, where $n$ is the 2nd-last row.

For an MDP $(S, s_0, A, P, r, \gamma)$, the process is:

1. Identify sub-goals that need to be achieved to achieve the final goal.
2. Create a new reward function $r'$ where the rewards are re-distributed to sub-goals, and where cumulative reward of using the original reward $r$ of any episode $e$ that reaches a real goal is roughly equal to the cumulative reward for $e$ using $r'$.
3. Solve a new MDP $(S, s_0, A, P, r', \gamma)$ to obtain policy $\pi$ (the MDP is the same but just with a different reward function).
4. Evaluate $\pi$ on the original MDP.

For example, consider the single-agent card game *Solitaire*. The aim of the game is to win by having all cards placed in four piles. Each pile must be in order from ace to king, alternating between red and black cards. If we were given a simulator for Solitaire, it may be that there is just one type of reward: a "point" for winning the game by arriving in one of the terminal states where the cards are in the correct piles. This reward would only be given in that terminal state.  

Simulations of a game like this would have to be long because there are no rewards until the end of the game. By re-distributed sub-goals that give a reward whenever a correct card is place on one of the piles, we feed rewards forward much earlier.

This idea is similar to [reward shaping](sec:single-agent:reward-shaping). In these notes, we focused on potential-based reward shaping, which gives small "fake" rewards. This effectively asks that you define a heuristic for states that guide the agent towards promising actions early during learning when there is not much information to exploit. Using sub-goals and new reward functions is similar, and the idea can be implemented as reward shaping, but explicitly changing the reward function can often be conceptually simpler to implement.

## Partially-observable state variables

Often we have situations where some variables are not observable for at least part of the game. For example, in Solitaire, the draw cards are not visible until we take each one. In this case, the model is a  [partially-observable Markov Decision Process](sec:mdps:pomdps) (POMDP). We can translate this to an MDP where the states are not represented as states in the real-world, but are *beliefs* over states in the real world. However, the computational complexity of this is high.

Another option is to simplify by *ignoring the partially observable variables.* In Solitaire, it is simple enough to just not track the cards in the deck and only consider actions for the cards that we can already see. In Solitaire, this will work well. If you have ever played the game, you will know that you spend very little time wondering about the cards in the deck. 

Of course, this will not always be a good idea: sometimes it is important to keep track of the partially-observable variables; for example, by mapping the probability that certain cards could be played in the future.

## State abstraction

In some cases, the size of the state space can be abstracted by combining some states that are different, but are equivalent from a semantic view. Solitaire is a good example of this. In the end, we need to have pile with alternative black and red cards. However, it does not matter whether a red card is a heart or a diamond, nor whether a black card is a club or a spade. In this case, we will still record there are two of each black card, but we could abstract the state to merely note that "black King" is present, and ignore whether there are one or two. This definitely *loses* information but may be suitable in some games and probably works well for a game like Solitaire.

## Heuristics

Using heuristics is a good way to speed up learning. There are three places where we could use heuristics:

1. First, if we have a reasonable heuristc, we can choose to "prune" states that are not promising, and explore only those that appear to be promising. For example, consider using [MCTS](sec:monte-carlo-tree-search) for Solitaire. If we are using a model or a simulator, we may be able to peek at the states resulting from actions. If one action results in us being able to move a block of cards to the goal pile, we may prioritise doing simulations using this action. Given we expand the first node fully before expanding any other nodes, we could look at the heuristic value of each node and prune it from the tree if it is too low. This would cut down our search space. Of course, a poor heuristic could do more damage than good! But it is not just MCTS that benefits. For example, using Q-learning or SARSA, we could terminate episodes when we reach unpromising states, either start new episodes or backtracking to an earlier promising node is using a simulator.
2. Second, heuristics can be good for helping to guide episodes towards good solutions. In any model-free or simulation-based technique, early episodes effectively choose each action with a uniform probability, as there is no good information to exploit. Like reward shaping, we can use heuristics to choose the most promising actions more often, meaning that simulations are still *randomised*, but informed. In MCTS, when we expand a node and then simulate, this is particularly valuable, as if we can find reasonable simulations, we get a better idea of the value of the expanded node. The trick here is to balance the heuristic against the information that we learn into our Q-function. Either we have a weighted measure, or we stop using the heurstic after we gain sufficient information in our Q-functions.
3. We can use heuristics to terminate episodes/simulations early. As we move towards terminal states, our heuristic values are likely to be more accurate, because it is "easier" to estimate how good a state is as we gain more information. As such, we can use heuristics to terminate episodes early. For example, in MCTS, instead of simulating until a terminal state, we can simulate for $X$ number of actions and then return the heuristic value of the state. This is an estimate, but often after a long episode, the value obtained is quite low as it probably involves a lot of random actions. In fact, this is precisely what temporal different methods like Q-learning and SARSA do! Q-learning uses $\max_a Q(s',a')$ as an heuristic to estimate the value of state $s'$ so that we can assign 'credit' to the previously executed action without having to do simulations. In that case, we simultaneously learn the heuristic too!

## Summary

- Techniques for solving MDPs face scalability issues. Using modelling tricks, we can find problems that are easier to solve, and apply them back
- Sometimes the smaller problem is enough to solve our problem; other times, it is not.
