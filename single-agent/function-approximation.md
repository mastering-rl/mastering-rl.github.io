
# Q-Function Approximation

## Learning Outcomes

1.  Manually apply linear Q-function approximation to solve small-scale
    MDP problems given some known features

2.  Select suitable features and design & implement Q-function
    approximation for model-free reinforcement learning techniques to
    solve medium-scale MDP problems automatically

3.  Argue the strengths and weaknesses of function approximation
    approaches

4.  Compare and contrast linear Q-learning with deep Q-learning


## Overview
Using a Q-table has two main limitations:

1. It requires that we visit every reachable state many times and apply every action many times to get a good estimate of $Q(s,a)$. Thus, if we never visit a state $s$, we have no estimate of $Q(s,a)$, even if we have visited states that are very similar to $s$.

2. It requires us to maintain a table of size $|A| \times |S|$, which is prohibitively large for any non-trivial problem.


To get around these we will look at how to use machine learning to approximate Q-functions. In particular, we will look at *linear function approximation* and approximation using *deep learning* (deep Q-learning). Instead of calculating an exact Q-function, we approximate it using simple methods that both eliminate the need for a large Q-table (therefore the methods scale better), and also allowing use to provide reasonable estimates of $Q(s,a)$ *even if we have not applied action $a$ in state $s$ previously*. 


:::{admonition} Example --- Freeway
Conside the game *Freeway*, in which a chicken needs to cross several lanes on a freeway without being run over by a car. A screenshot of the game is shown below:

![image](./figs/freeway_screenshot.png)

Let us assume that there are 12 rows and about 40 columns. This grossly underestimates the actual number of rows and columns because the cars move a few pixels at a time, not in columns. This means there are 480 different positions that a chicken can be in, and there are two chickens. We also need to record whether there is a car in each location. 

This leads to:

$$480^2 +  2^{480} = 3.12\times 10^{144} \text{ states}$$ 

There are four actions: left, right, up, down.

A Q-table would need to store $12.48\times 10^{144}$ entries. This is a huge Q-table for what is a trivial example compared to many other problems.
:::

## Linear Q-learning (Linear Function Approximation) 

The key idea is to *approximate* the Q-function using a linear combination of *features* and their weights.
Instead of recording everything in detail, we think about what is most important to know, and model that.

What are some features that are relevant to the Freeway example?

The overall process is:

1.  For the states, consider what are the features that determine its representation.

2.  During learning, perform updates based on the *weights* of
    *features* instead of states.

3.  Estimate $Q(s,a)$ by summing the features and their weights.


:::{admonition} Example --- Features for *Freeway*

Instead of recording the position of both chickens and whether there is a car in every position, we just record the following features:

- the number of columns each chicken is away from the other side of the road in (two features -- one for each chicken); and
- how far away the *closest* car is in the row above and below each chicken (four features --- two for each chicken).

This requires just six features. 

:::

### Approximate Q-function Representation 
In linear Q-learning, we store features and weights, not states. What we need to learn is how important each feature is (its *weight*) for each action.

To represent this, we have two vectors:

1.  A *feature vector*, $f(s,a)$, which is a vector of $n \cdot |A|$
    different functions, where $n$ is the number of state features and
    $|A|$ the number of actions. Each function extracts the value of a
    feature for state-action pair $(s,a)$. We say $f_i(s,a)$ extracts
    the $i$th feature from the state-action pair $(s,a)$:

    $$f(s,a) = \begin{pmatrix} 
        f_1(s,a) \\
        f_2(s,a) \\
        \ldots\\
        f_n(s,a) \\
        \end{pmatrix}$$

    In the Freeway example, we have a vector with six state features
    times four actions. The function $f_1(s,Up)$ returns value of the feature that represents the distance chicken 1 is away from the goal. The function $f_{3}(s, Up)$
     returns the distance to the nearest car in the row above the first chicken. 

2.  A *weight* vector $w$ of size $n \times |A|$: one weight for each
    feature-action pair. $w^a_i$ defines the weight of a feature $i$ for
    action $a$.

### Defining State-Action Features 

Often it is easier to just define features for states, rather than state-action pairs. The features are just a vector of $n$ functions of the form $f_i(s)$.

However, for most applications, the weight of a feature is related to the action. The weight of being one step away from the end in Freeway is different if we go Up to if we go Right.

It is straightforward to construct $n \times |A|$ state-pair features from just $n$ state features:

$$
f_{ik}(s,a) = \Bigg \{
\begin{array}{ll}
 f_i(s) & \text{if } a=a_k\\
 0      & \text{otherwise}
 ~~~ 1 \leq i \leq n, 1 \leq k \leq |A|
 \end{array}
$$

This effectively results in $|A|$ different weight vectors:

$$
 f(s,a_1) = \begin{pmatrix} 
f_{1,a_1}(s,a) \\
f_{2,a_1}(s,a) \\
0\\
0\\
0\\
0\\
\ldots
\end{pmatrix}~~
f(s,a_2) = \begin{pmatrix} 
0\\
0\\
f_{1,a_2}(s,a) \\
f_{2,a_2}(s,a) \\
0\\
0\\
\ldots
\end{pmatrix}~~
f(s,a_3) = \begin{pmatrix} 
0\\
0\\
0\\
0\\
f_{1,a_3}(s,a) \\
f_{2,a_3}(s,a) \\
\ldots
\end{pmatrix}~~\ldots
$$

### Approximate Q-function Computation 
Give a feature vector $f$ and a weight vector $w$, the Q-value of a state is a simple linear combination of features and weights:

$$
\begin{array}{lll}
  Q(s,a) & = & f_1(s,a) \cdot w^a_1 + f_2(s,a)\cdot w^a_2 + \ldots  + f_n(s,a) \cdot w^a_n\\
         & = & \sum_{i=0}^{n} f_i(s,a) w^a_i
\end{array}
$$

:::{admonition} Example --- Approximate Q-function computation for Freeway

For the Freeway example, we would assume that moving up would give a better score than moving down, all else equal (that is, if the closest car in the next row up is the same distance away than the closest in the next row down). So, for state $s$ where the chicken is in row 1:

$$
\begin{array}{lll}
  Q(s,Up)   & = &  f_1(s,Up)\cdot 0.31  + \ldots + f_{14}(s,Up) \cdot 0.04
\end{array}
$$

Note that to be effective, our feature values should be *normalised* using e.g. min-max normalisation or mean normalisation.

### Approximate Q-function Update 

To use approximate Q-functions in reinforcement learning, there are two steps we need to change from the standard algorithsm: (1) initialisation; and (2) update.

For initialisation, initialise all weights to 0. Alternatively, you can
try Q-function initialisation and assign weights that you think will be
`good' weights.

For update, we now need to update the weights instead of the actions.
For Q-learning, the update rule is now:

$$w^a_i \leftarrow w^a_i + \alpha [r + \gamma max_a' Q(s',a') - Q(s,a)]\ f_i(s,a)$$

For SARSA:

$$w^a_i \leftarrow w^a_i + \alpha [r + \gamma Q(s',a') - Q(s,a)]\ f_i(s,a)$$

Note: we need to update for each feature $i$ for the last executed action $a$.

As this is linear, it is therefore convex, so the weights will converge!!

:::{note} Q-value Propagation

Note that this has the effect of updating Q-values to states that have never been visited! 

In Freeway, for example, if we receive our first reward by crossing the road (going Up from the final row), this will update the weight all features for Up, and now we have a Q-value for going Up from *any* position on the final row.
:::

:::{admonition} Example --- Q-value update for Freeway

Assume that all weights are 0, therefore, $Q(s,a) = 0$ for every state and action. Now, we receive the reward of 10 for getting to the other side of the road. If feature 14 is has the value $\frac{r}{D}$, where $r$ is the current row and $D$ is the distance to the other side, then
we have:

$$
\begin{array}{lll}
  w^a_i & \leftarrow & w^a_i + \alpha[r + \gamma \max_a Q(s',a') - Q(s,a)] f_i(s,a)\\
  w^{Up}_{14} & \leftarrow & 0 + 0.5[10 + 0.9 \times 0] \frac{10}{10}\\
              & = &5
\end{array}
$$

From this, we now can get an estimate of $Q(s,Up)$ from any state because we have some weights in our linear function. Those that are closer to the other size of the road will get a higher Q-value than those further away (all other things being equal).

## Deep Q-learning

The latest hype in reinforcement learning is all about the use of deep neural networks to approximate value and Q-functions. Instead of selecting features and training weights, we learn the parameters $\theta$ to a neural network. The Q-function is $Q(s,a; \theta)$, so takes the parameters as an argument.

The TD update for Q-learning is just:

$$\theta \leftarrow \theta + \alpha[r + \gamma \max_{a'} Q(s',a'; \theta) - Q(s,a ;\theta)]
\nabla_{\theta} Q(s,a; \theta)$$

where $\nabla_{\theta} Q(s,a; \theta)$ is the *gradient* of the Q-function.

**Advantages (compared to linear Q functions):** 
- We do not need to select features -- the 'features' will be learnt as part of the hidden layers of the neural network. 
-   The state $s$ can be less structured, such as images or sequences of images (video).

**Disadvantages:**
- There are no convergence guarantees.
- Deep neural networks are very data hungry because they need to learn features as well as Q-function, so compared to a linear approximation with good features, learning good Q-functions can be very difficult.

Despite this, deep Q-learning  works remarkably well in some areas, especially for tasks that require vision (see the robotic arm grasping unknown objects).

### Strengths and Limitations of Q-function Approximation

Approximating Q-functions using machine learning techniques such as linear functions or deep learning  has advantages and disadvantages.

**Advantages:**

-   Memory: More efficient representation compared with Q-tables because we only store weights/parameters for the Q-function, rather than the the $|A| \times |S|$ entries for a Q-table.

-   Q-value propagation: we do not need to apply action $a$ in state $s$ to get a value for $Q(s,a)$ because the Q-function generalises.

**Disadvantages:**

-   The Q-function is now only an approximation of the real Q-function: states that share feature values may have different actual values. 

## Applications of Function Approximation 

A great application of using off-policy updates in deep Q-learning for robotic arms to learn how to grasp unknown objects. The only input for the problem is the camera data:

<iframe width="560" height="315" src="https://www.youtube.com/embed/cXaic_k80uM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

This is using policy iteration (policy gradient descent) rather than standard Q-learning.

## Summary

1.  We can scale reinforcement learning by approximating Q-functions, rather than storing complete Q-tables.

2.  Using simple linear methods in which we select features and learn weights are effective and guarantee convergence.

3.  Deep Q-learning offers alternatives in which we do not need to select features, but require a lot of training data (more episodes) and has no convergence guarantees.




## Further Reading

- Chapter 9 (Approximate Solution Methods) of *Introduction to Reinforcement Learning* \[*Sutton and Barto*\]

  <https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html>

- Deep Q-learning for Atari. This uses Convolutional Neural Networks (NN) to estimate $\mathcal{Q}(s,a)$. The input for the NN is the state, and the output is the esimated reward for each action. There are two papers worth reading on this:

  - *Human-level control through deep reinforcement
    learning*: <http://www.davidqiu.com:8888/research/nature14236.pdf>
  - *Playing Atari with Deep Reinforcement Learning* from DeepMind. <https://arxiv.org/pdf/1312.5602v1.pdf>

-   Before AlphaGo there was TD-gammon, which was the first paper to
    combine reinforcement learning and neural networks:
    <http://www.aaai.org/Papers/Symposia/Fall/1993/FS-93-02/FS93-02-003.pdf>
