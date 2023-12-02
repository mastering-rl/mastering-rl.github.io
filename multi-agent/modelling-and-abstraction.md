(sec:multi-agent:modelling-and-abstraction)=

# Modelling and abstraction for multi-agent games

```{admonition}  Learning outcomes
The learning outcomes of this chapter are:

1.  Describe modelling and abstraction strategies to scale algorithms for multi-agent games.
    
2.  Apply modelling and abstraction strategies to non-trivial multi-agent games.
```

## Overview

Just as we can use [modelling and abstraction to help us scale algorithms for MDP problems](sec:single-agent:modelling-and-abstraction), we can use it for multi-agent games. All of the strategies described in the [MDP modelling and abstraction section](sec:single-agent:modelling-and-abstraction) can also be applied to multi-agent games. However, when we consider opponents, there are a few other modelling and abstraction strategies we can use.


## Multi-agent to single-agent

A simple and sometimes effective way is to simply ignore our opponent, and only search or simulate over our own actions. This may sound like a bad strategy, and in games like Chess and Go where players actively take each others' pieces or surround each others' pieces, this will not be a good abstraction. However, in other games, it can be very useful. Given that we cannot control our opponents' actions directly, in many multi-agent games, it is more important to focus on what we can control: our own actions.

Consider multi-player board games like Cluedo or Scrabble. While our opponents' actions actively influence what we can do and the rewards we receive, quite often most of our strategy is formed by just focusing on our own pieces and what we want to do, because we have very little ability to influence each others' strategy. For example, in Scrabble, we do not spend much time considering what possible tiles our opponents could put down, even though it is a competitive, multi-agent game, and what they do can influence our future actions. 

By ignoring our opponents actions and simply solving the problem as a single-player game, we can search/simulate much deeper and broader to gain information about our own actions. 

## Limited reasoning of opponent actions

Similar to ignoring the opponent is to limit our own reasoning about our opponents actions.

**Focus on known feasible opponent actions.** Quite often in multi-agent games, we may have very limited knowledge about what our opponent can do; for example, they have some private information such as cards. So, it is often infeasible to consider all of their possible actions. Instead, we may be better focusing only on the moves that we know our opponent can make, as these are more likely to be made because we know they can be made. Therefore, it is more valuable to spend time analysing these.

**Focus on opponents' most promising actions only.** Another way is to use [heuristic techniques](sec:single-agent:modelling-and-abstraction:heuristics) over our opponents' moves. Sometimes we may not need to consider ALL possible actions our opponents can make in a state, but just to consider a small handful to get an idea of the value of the state. Of most use will be those that the opponents are more likely to make. We could calculate how likely they are using an heuristic or even a hand-coded opponent. In the former, this is the same as [state/action pruning for single-agent problems](sec:single-agent:modelling-and-abstraction:heuristics), except that we are pruning our opponents moves. Often, we can prune more aggressively for our opponents because we cannot control their actions.  For the latter, we can prune all but the most promising actions. If we have a reasonable hand-coded agent but we want to learn to play better, we can use the hand-coded opponent to give us its best action for our opponent, rather than exploring multiple options. These techniques again allow us to search deeper and broader to gain information about our own actions. 


## Takeaways

```{admonition} Takeaways
- The addition of other agents often create further **scalability** issues in stochastic games. 
- Using modelling tricks, we can find problems that are easier to solve, and apply them back to the original problem.
- Ignoring or approximating opponents can be good in some stochastic games, but not always.
```