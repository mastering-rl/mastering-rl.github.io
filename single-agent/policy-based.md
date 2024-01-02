# Policy-based methods

In this chapter, we cover **policy-based** methods for reinforcement learning. Policy-based methods learn a policy directly, rather than learning the value of states and actions. 

As noted earlier, learning a policy directly has advantages, particularly for applications where the state space or the action space are massive or infinite. If the action space is infinite, then we cannot do [policy extraction](sec:mdps:policy-extraction) because that requires us to iterate over all actions and extract the one that maximises the reward. Learning the policy directly mitigates this.

In this chapter, we will look introduce just two policy-based methods: 

1. [**Policy iteration*]*(sec:policy-iteration): Like value iteration, this is a dynamic programming-based method for model-based MDPs
2. [**Policy gradients**](sec:policy-based:policy-gradients): This is a model-free technique that performs gradient ascent, which is the same as the well-known gradient descent technique, but which maximises rewards instead of minimising error.
