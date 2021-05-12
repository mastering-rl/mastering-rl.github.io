In contrast to value-based methods, Policy-based methods are targeted to optimize the policy function $\pi$ that maps states to actions directly instead of optimizing the value function. This is done by updating the parameters $\theta$ of the policy $\pi(a|s; \theta)$ via gradient ascent on the expectation of $R_t$, $\mathbb{E}\left[R_{t}\right]$ (i.e. to increase the expected Return). One of the well known example of this is the REINFORCE algorithm \cite{williams1992simple}. In this algorithm the policy parameters $\theta$ are updated in the direction of gradient $\nabla_{\theta} \mathbb{E}\left[R_{t}\right]$, which is estimated by the score function of the log likelihood of actions; $\nabla_{\theta} \log \pi\left(a_{t} | s_{t} ; \theta\right) R_{t}$._


This approach is unbiased, however it does have a high variance due to the way the $R{_t}$ is estimated, from sampling, and the possible wide changes in the results. As per \citeauthor{williams1992simple}, this variance can be reduced by introducing a baseline function $b_t(s_t)$ that depends on the state regardless of the action. This function is deducted from the return, hence, the gradient is updated to $\nabla_{\theta} \log \pi\left(a_{t} | s_{t} ; \theta\right) (R_{t} - b_t(s_t))$.

The most commonly used function as a baseline is the state-value function $V^\pi(s_t)$. The state value can be estimated by parameterising $w$ where $w$ is a parameter vector that is learned by some methods such as Monte Carlo. 

$R_t - b_t$ can be considered as the estimation of the Advantage of on an action $A(s, a) = Q^\pi(s, a) - V^\pi(s)$ as $R_t$ is an estimation of $Q^\pi(a_t,s_t)$. In general, the Advantage function provides a relative measure of the importance of the action and leads to faster identification of the right actions in policy evaluation. From this approach, the actor-critic architecture is devised, where the policy $\pi$ is the actor and the baseline $b_t$ is the critic \citet{degris2012model}.

Overall, the policy-based approach is more efficient in high dimensional action space and converges faster than value-based methods as the action space typically is more limited than the possible rewards, especially when considering discrete action spaces. Via gradient methods, the policy updates are smoother and will eventually converge to either local or global optimal.

## Applications of Policy Gradients

A great application of using off-policy updates in deep Q-learning for robotic arms to learn how to grasp unknown objects. The only input for the problem is the camera data:

<iframe width="560" height="315" src="https://www.youtube.com/embed/cXaic_k80uM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

This is using policy iteration (policy gradient descent) rather than standard Q-learning.