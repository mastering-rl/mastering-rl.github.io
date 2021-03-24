# Value-based Methods

Techniques for solving MDPs (and POMDPs) can be separated into three categories:

1. *Value-based* techniques aim to learn the value of states (or learn an estimate for value of states) and actions: that is, they learn value functions or Q functions. We then use policy extraction to get a policy for deciding actions.
2. *Policy-based* techniques learn a policy directly, which completely by-passes learning values of states or actions all together. This is important if for example, the state space or the action space are massive or infinite. If the action space is infinite, then using policy extraction as defined in Part I is not possible because we must iterate over all actions to find the optimal one. If we learn the policy directly, we do not need this.
3. *Hybrid* techniques that combine value- and policy-based techniques.

In this chapter, we investigate some of the foundational techniques for value-based reinforcement learning, and follow up with policy-based and hybrid techniques in the follow chapters.