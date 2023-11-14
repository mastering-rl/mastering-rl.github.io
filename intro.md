
```{figure} ./logo-3D.svg
:name: logo
```

# Introduction

This book provides a foundational introduction to the problem of **reinforcement learning**.
It combines narrative, maths, and code, to help the reader gain an introduction to the area, why it exists, how to solve reinforcement learning problems, and the strengths and weaknesses of different approaches.

The **aim** of the book is to provide the reader with sufficient foundation that they can design and implement agents using reinforcement learning, and can understand more advanced topics covered in research papers.

We start with dynamic programming approaches, such as [value iteration](sec:value-iteration). While the current hype is around deep reinforcement learning, understanding value iteration helps learners to appreciate what techniques like [deep policy gradients](sec:policy-gradient:deep-policy-gradients) and [actor critic](sec:policy-gradient:actor-critic) methods are optimising, and how they work. Oh and, for many MDP problems, value iteration is a much better solution than deep reinforcement learning.

In Part I of these notes, we introduce  *Markov Decision Processes* (MDPs). MDPs allow us to model problems  in which the outcomes of actions are probabilistic; that is, we do not know the outcome beforehand, but we know there is some probability distribution over a set of possible outcomes. We look at *model-based* techniques, where these probabilistic outcomes are given to use, and *model-free* techniques, which are flexible enough the probabilities are unknown, but we can sample enough times that we can still learn good behaviour.

In Part II of these notes, we look at *multi-agent MDPs* (sometimes called *games*), in which there are multiple (possibly adversarial) actors in a problem, and we need to plan our actions while also considering what the other actors in the environment will do. Again, we look at both model-based and model-free techniques.

## The book

This book is written using Markdown, and compiled into HTML using [Jupyter Book](https://jupyterbook.org/) --- an extension of Jupyter notebooks.

Each individual HTML page can be downloaded individuallly as a [Jupyter notebook](https://jupyter.org/), but note that you need to install the code and dependencies below.

## Code

All code in this book is executable. You can download the code from [here](https://gibberblot.github.io/rl-notes/_static/code.zip).

```{note}
The code in this book is written for understandability rather than efficiency. It is not intended to be production-level code, such as the [Keras RL](https://github.com/keras-rl/keras-rl) reinforcement learning package.

If you understand the code in these notes, you will have little problem using production-level packages such as Keras RL.

The code in this book is written with the attempt to use very little Python-specific syntax, to enable those less familiar with Python to understand code snippets.

The code in this book is written using as few external libraries as possible, to make this easy to download and run yourself.
```

Once you have downloaded, unzip the code and add the folder to your PYTHONPATH variable if you want to download the Jupyter notebooks.

Most files in the code have a ``main`` function that can be run using just ``python <filename>py``. For most of these, no external libraries are required. However, if you want to plot the graphs or draw the trees, you will need to install:

1. The [Matplotlib library](https://matplotlib.org/) for plotting graphs. You can download from the website or install with ``pip install matplotlib``. 

2. The [Scipy library](https://www.scipy.org/) for helping with the graph plotting. You can download from the website or install with ``pip install scipy``.

3. The [Graphviz Python library](https://graphviz.readthedocs.io/en/stable/) for drawing trees. You can download from the website or use ``pip install graphviz``. To render the generated graphs, you will also need to install [Graphviz the tool](https://www.graphviz.org/download/), which is called by the Python package.

## The Author

These notes are written and maintained by [Tim Miller](https://uqtmiller.github.io/), Professor of Artifical Intelligence at  [The University of Queensland](https://uq.edu.au/), Brisbane/Meaanjin, Australia.

If you find any errors or would like to provide other feedback, feel free to [email me](mailto:timothy.miller@uq.edu.au).

If you use this as part of your teaching or learning in a course, please [let me know!](mailto:timothy.miller@uq.edu.au).

## Acknowledgements

Thanks to Alan Lewis for his excellent idea of demonstrating [policy gradients using a logistic regression policy](sec:policy-gradients:logistic-regression); and furthermore, for implementing the source for this and the [deep policy gradient agent](sec:policy-gradient:deep-policy-gradients). Thanks also to Alan for setting up the library for play GIF files, which supports the interactive visualisations that are so useful in this book.

Thanks to Emma Baillie for the idea and implementation of the Contested Crossing examples.