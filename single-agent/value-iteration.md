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
(sec:value-iteration)=
# Value Iteration
## Learning outcomes

The learning outcomes of this chapter are:

1.  Apply value iteration to solve small-scale MDP problems manually and program value iteration algorithms to  solve medium-scale MDP problems automatically
    
2.  Construct a policy from a value function

4.  Discuss the strengths and weaknesses of value iteration

## Overview

*Value Iteration* is a method for finding the optimal value function $V^*$ by solving the
Bellman equations iteratively. It uses the concept of dynamic programming to maintain  a value function $V$ that approximates the optimal value function $V^*$, iteratively improving $V$ until it converges to $V^*$ (or close to it). 

## Algorithm

Once we understand the Bellman equation, the value iteration algorithm is straightforward: we just repeatedly calculate $V$ using the Bellman equation until we converge to the solution or we execute a pre-determined number of iterations.

:::{admonition} Algorithm -- Value Iteration

**Input:** MDP $M = \langle S, s_0, A, P_a(s' \mid s), r(s,a,s')\rangle$\
**Output:** Value function $V$

Set $V$ to arbitrary value function; e.g., $V(s)=0$ for all $s$

$\text{Repeat}$\
$\quad\quad \Delta \leftarrow 0$\
$\quad\quad \text{For each}~ s \in S$\
$\quad\quad\quad\quad \underbrace{V'(s) \leftarrow \max_{a \in A(s)} \sum_{s' \in S}  P_a(s' \mid s)\ [r(s,a,s') +  \gamma\ V(s') ]}_{\text{Bellman equation}}$\
$\quad\quad\quad\quad \Delta \leftarrow \max(\Delta, |V'(s) - V(s)|)$\
$\quad\quad V \leftarrow V'$\
$\text{Until}~ \Delta \leq \theta$
:::

As we can see, this is just applying the Bellman equation iteratively until either the value function $V$ doesn't change anymore, or until it changes in by a very small amount ($\theta$).

We could also write the algorithm using the idea of Q-values, which is closer to a code-based implementation. For this, the loop is:

$\quad\quad \Delta \leftarrow 0$\
$\quad\quad \text{For each}~ s \in S$\
$\quad\quad\quad\quad \text{For each}~ a \in A(s)$\
$\quad\quad\quad\quad\quad\quad Q(s,a) \leftarrow \sum_{s' \in S}  P_a(s' \mid s)\ [r(s,a,s') +  \gamma\ V(s') ]$\
$\quad\quad\quad\quad \Delta \leftarrow \max(\Delta, |\max_{a \in A(s)} Q(s,a) - V(s)|)$\
$\quad\quad\quad\quad V(s) \leftarrow \max_{a \in A(s)} Q(s,a)$

Value iteration converges to the optimal policy as iterations continue: $V \mapsto V^*$ as $i \mapsto \infty$, where $i$ is the number of iterations. So, given an infinite amount of iterations, it will be optimal.


Value iteration converges to the optimal value function $V^*$ asymptotically, but in practice, the algorithm is stopped when the *residual*  $\Delta$ reaches some pre-determined threshold $\theta$ -- that is, when the largest change in the values between iterations is "small enough".

A policy can now be easily defined: in a state $s$, given $V$, choose the action with the highest expected reward using policy extraction. The loss of the result greedy policy is bound by $\frac{2 \gamma  \Delta}{1-\gamma}$.

Note that we do not need an optimal value function $V$ to obtain an optimal policy. A value function that is "close enough" can still give an optimal policy because the small values do not change the resulting policy. Of course, we would not *know* whether a policy is optimal unless we know the value function is optimal.

## Complexity

The complexity of each iteration is $O(|S|^2 |A|)$. On each iteration, we iterate in an outer loop over all states in $S$, and in each outer loop iteration, we need to iterate over all states ($\sum_{s' \in S}$), meaning $|S|^2$ iterations. But also within each outer loop iteration, we need to calculate the value for every action to find the maximum.

It is clear to see that the value iteration can be easily parallelised by updating the value of many states at once: the values of states at step $t + 1$ are dependent only on the value of other states at step $t$.

(sec:value-iteration:implementation)=
## Implementation

Below is a Python implementation for value iteration. In this implementation, the parameters `iterations` is the number of iterations around the loop, which will terminate before convergence is the maximum number of iterations is reach. The parameter `theta` is $\theta$ in the value iteration algorithm above. Once the difference ($\Delta$) is less than `theta` , the loop will terminate.

```{code-cell} ipython3
:load: "../code/value_iteration.py"
```

Given this, we can create a GridWorld MDP, and solve using value iteration. The code below prints the value function for value iteration after 1, 2, 3, 4, 5, 10, and 100 iterations:

```{code-cell} ipython3
mdp = GridWorld()

for iterations in [1, 2, 3, 4, 5, 10, 100]:
    values = TabularValueFunction()
    ValueIteration(mdp, values).value_iteration(iterations=iterations)
    print("After iteration " + str(iterations))
    print(mdp.value_function_to_string(values) + "\n")
```

From the policy, we extract a policy:

```{code-cell} ipython3
values = TabularValueFunction()
ValueIteration(mdp, values).value_iteration(iterations=100)
policy = values.extract_policy(mdp)
print("Policy after 100 iterations")
print(mdp.policy_to_string(policy))
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()


def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(60):
    x += np.pi / 15.
    y += np.pi / 20.
    im = ax.imshow(f(x, y), animated=True)
    if i == 0:
        ax.imshow(f(x, y))  # show an initial one first
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

# To save the animation, use e.g.

#ani.save("movie.mp4")
#
# or
#
writer = animation.FFMpegWriter(
    fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save("movie.mp4", writer=writer)

plt.show()
```

<div class="sphx-glr-animation docutils container">
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css">
<script language="javascript" type="bc88701884532dc48e24f315-text/javascript">
  function isInternetExplorer() {
    ua = navigator.userAgent;
    /* MSIE used to detect old browsers and Trident used to newer ones*/
    return ua.indexOf("MSIE ") > -1 || ua.indexOf("Trident/") > -1;
  }

  /* Define the Animation class */
  function Animation(frames, img_id, slider_id, interval, loop_select_id){
    this.img_id = img_id;
    this.slider_id = slider_id;
    this.loop_select_id = loop_select_id;
    this.interval = interval;
    this.current_frame = 0;
    this.direction = 0;
    this.timer = null;
    this.frames = new Array(frames.length);

    for (var i=0; i<frames.length; i++)
    {
     this.frames[i] = new Image();
     this.frames[i].src = frames[i];
    }
    var slider = document.getElementById(this.slider_id);
    slider.max = this.frames.length - 1;
    if (isInternetExplorer()) {
        // switch from oninput to onchange because IE <= 11 does not conform
        // with W3C specification. It ignores oninput and onchange behaves
        // like oninput. In contrast, Microsoft Edge behaves correctly.
        slider.setAttribute('onchange', slider.getAttribute('oninput'));
        slider.setAttribute('oninput', null);
    }
    this.set_frame(this.current_frame);
  }

  Animation.prototype.get_loop_state = function(){
    var button_group = document[this.loop_select_id].state;
    for (var i = 0; i < button_group.length; i++) {
        var button = button_group[i];
        if (button.checked) {
            return button.value;
        }
    }
    return undefined;
  }

  Animation.prototype.set_frame = function(frame){
    this.current_frame = frame;
    document.getElementById(this.img_id).src =
            this.frames[this.current_frame].src;
    document.getElementById(this.slider_id).value = this.current_frame;
  }

  Animation.prototype.next_frame = function()
  {
    this.set_frame(Math.min(this.frames.length - 1, this.current_frame + 1));
  }

  Animation.prototype.previous_frame = function()
  {
    this.set_frame(Math.max(0, this.current_frame - 1));
  }

  Animation.prototype.first_frame = function()
  {
    this.set_frame(0);
  }

  Animation.prototype.last_frame = function()
  {
    this.set_frame(this.frames.length - 1);
  }

  Animation.prototype.slower = function()
  {
    this.interval /= 0.7;
    if(this.direction > 0){this.play_animation();}
    else if(this.direction < 0){this.reverse_animation();}
  }

  Animation.prototype.faster = function()
  {
    this.interval *= 0.7;
    if(this.direction > 0){this.play_animation();}
    else if(this.direction < 0){this.reverse_animation();}
  }

  Animation.prototype.anim_step_forward = function()
  {
    this.current_frame += 1;
    if(this.current_frame < this.frames.length){
      this.set_frame(this.current_frame);
    }else{
      var loop_state = this.get_loop_state();
      if(loop_state == "loop"){
        this.first_frame();
      }else if(loop_state == "reflect"){
        this.last_frame();
        this.reverse_animation();
      }else{
        this.pause_animation();
        this.last_frame();
      }
    }
  }

  Animation.prototype.anim_step_reverse = function()
  {
    this.current_frame -= 1;
    if(this.current_frame >= 0){
      this.set_frame(this.current_frame);
    }else{
      var loop_state = this.get_loop_state();
      if(loop_state == "loop"){
        this.last_frame();
      }else if(loop_state == "reflect"){
        this.first_frame();
        this.play_animation();
      }else{
        this.pause_animation();
        this.first_frame();
      }
    }
  }

  Animation.prototype.pause_animation = function()
  {
    this.direction = 0;
    if (this.timer){
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  Animation.prototype.play_animation = function()
  {
    this.pause_animation();
    this.direction = 1;
    var t = this;
    if (!this.timer) this.timer = setInterval(function() {
        t.anim_step_forward();
    }, this.interval);
  }

  Animation.prototype.reverse_animation = function()
  {
    this.pause_animation();
    this.direction = -1;
    var t = this;
    if (!this.timer) this.timer = setInterval(function() {
        t.anim_step_reverse();
    }, this.interval);
  }
</script>
<style>
.animation {
    display: inline-block;
    text-align: center;
}
input[type=range].anim-slider {
    width: 374px;
    margin-left: auto;
    margin-right: auto;
}
.anim-buttons {
    margin: 8px 0px;
}
.anim-buttons button {
    padding: 0;
    width: 36px;
}
.anim-state label {
    margin-right: 8px;
}
.anim-state input {
    margin: 0;
    vertical-align: middle;
}
</style>
<div class="animation">
<img id="_anim_imgceb663c80a264b45baf016cda3e3bfd4">
<div class="anim-controls">
<input id="_anim_sliderceb663c80a264b45baf016cda3e3bfd4" type="range" class="anim-slider" name="points" min="0" max="1" step="1" value="0" oninput="if (!window.__cfRLUnblockHandlers) return false; animceb663c80a264b45baf016cda3e3bfd4.set_frame(parseInt(this.value));" data-cf-modified-bc88701884532dc48e24f315-=""></input>
<div class="anim-buttons">
<button title="Decrease speed" aria-label="Decrease speed" onclick="if (!window.__cfRLUnblockHandlers) return false; animceb663c80a264b45baf016cda3e3bfd4.slower()" data-cf-modified-bc88701884532dc48e24f315-="">
<i class="fa fa-minus"></i></button>
<button title="First frame" aria-label="First frame" onclick="if (!window.__cfRLUnblockHandlers) return false; animceb663c80a264b45baf016cda3e3bfd4.first_frame()" data-cf-modified-bc88701884532dc48e24f315-="">
<i class="fa fa-fast-backward"></i></button>
<button title="Previous frame" aria-label="Previous frame" onclick="if (!window.__cfRLUnblockHandlers) return false; animceb663c80a264b45baf016cda3e3bfd4.previous_frame()" data-cf-modified-bc88701884532dc48e24f315-="">
<i class="fa fa-step-backward"></i></button>
<button title="Play backwards" aria-label="Play backwards" onclick="if (!window.__cfRLUnblockHandlers) return false; animceb663c80a264b45baf016cda3e3bfd4.reverse_animation()" data-cf-modified-bc88701884532dc48e24f315-="">
<i class="fa fa-play fa-flip-horizontal"></i></button>
<button title="Pause" aria-label="Pause" onclick="if (!window.__cfRLUnblockHandlers) return false; animceb663c80a264b45baf016cda3e3bfd4.pause_animation()" data-cf-modified-bc88701884532dc48e24f315-="">
<i class="fa fa-pause"></i></button>
<button title="Play" aria-label="Play" onclick="if (!window.__cfRLUnblockHandlers) return false; animceb663c80a264b45baf016cda3e3bfd4.play_animation()" data-cf-modified-bc88701884532dc48e24f315-="">
<i class="fa fa-play"></i></button>
<button title="Next frame" aria-label="Next frame" onclick="if (!window.__cfRLUnblockHandlers) return false; animceb663c80a264b45baf016cda3e3bfd4.next_frame()" data-cf-modified-bc88701884532dc48e24f315-="">
<i class="fa fa-step-forward"></i></button>
<button title="Last frame" aria-label="Last frame" onclick="if (!window.__cfRLUnblockHandlers) return false; animceb663c80a264b45baf016cda3e3bfd4.last_frame()" data-cf-modified-bc88701884532dc48e24f315-="">
<i class="fa fa-fast-forward"></i></button>
<button title="Increase speed" aria-label="Increase speed" onclick="if (!window.__cfRLUnblockHandlers) return false; animceb663c80a264b45baf016cda3e3bfd4.faster()" data-cf-modified-bc88701884532dc48e24f315-="">
<i class="fa fa-plus"></i></button>
</div>
<form title="Repetition mode" aria-label="Repetition mode" action="#n" name="_anim_loop_selectceb663c80a264b45baf016cda3e3bfd4" class="anim-state">
<input type="radio" name="state" value="once" id="_anim_radio1_ceb663c80a264b45baf016cda3e3bfd4">
<label for="_anim_radio1_ceb663c80a264b45baf016cda3e3bfd4">Once</label>
<input type="radio" name="state" value="loop" id="_anim_radio2_ceb663c80a264b45baf016cda3e3bfd4" checked>
<label for="_anim_radio2_ceb663c80a264b45baf016cda3e3bfd4">Loop</label>
<input type="radio" name="state" value="reflect" id="_anim_radio3_ceb663c80a264b45baf016cda3e3bfd4">
<label for="_anim_radio3_ceb663c80a264b45baf016cda3e3bfd4">Reflect</label>
</form>
</div>
</div>
<script language="javascript" type="bc88701884532dc48e24f315-text/javascript">
  /* Instantiate the Animation class. */
  /* The IDs given should match those used in the template above. */
  (function() {
    var img_id = "_anim_imgceb663c80a264b45baf016cda3e3bfd4";
    var slider_id = "_anim_sliderceb663c80a264b45baf016cda3e3bfd4";
    var loop_select_id = "_anim_loop_selectceb663c80a264b45baf016cda3e3bfd4";
    var frames = new Array(60);

  frames[0] = "data:image/png;base64,1" 
    /* set a timeout to make sure all the above elements are created before
       the object is initialized. */
    setTimeout(function() {
        animceb663c80a264b45baf016cda3e3bfd4 = new Animation(frames, img_id, slider_id, 50.0,
                                 loop_select_id);
    }, 0);
  })()
</script></div>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">matplotlib.animation</span> <span class="k">as</span> <span class="nn">animation</span>

<a href="../../api/figure_api.html#matplotlib.figure.Figure" title="matplotlib.figure.Figure" class="sphx-glr-backref-module-matplotlib-figure sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">fig</span></a><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <a href="../../api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots" title="matplotlib.pyplot.subplots" class="sphx-glr-backref-module-matplotlib-pyplot sphx-glr-backref-type-py-function"><span class="n">plt</span><span class="o">.</span><span class="n">subplots</span></a><span class="p">()</span>


<span class="k">def</span> <span class="nf">f</span><span class="p">(</span><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray" title="numpy.ndarray" class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">x</span></a><span class="p">,</span> <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray" title="numpy.ndarray" class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">y</span></a><span class="p">):</span>
    <span class="k">return</span> <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.html#numpy.ufunc" title="numpy.ufunc" class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">np</span><span class="o">.</span><span class="n">sin</span></a><span class="p">(</span><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray" title="numpy.ndarray" class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">x</span></a><span class="p">)</span> <span class="o">+</span> <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.html#numpy.ufunc" title="numpy.ufunc" class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">np</span><span class="o">.</span><span class="n">cos</span></a><span class="p">(</span><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray" title="numpy.ndarray" class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">y</span></a><span class="p">)</span>

<a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray" title="numpy.ndarray" class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">x</span></a> <span class="o">=</span> <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html#numpy.linspace" title="numpy.linspace" class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-function"><span class="n">np</span><span class="o">.</span><span class="n">linspace</span></a><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">2</span> <span class="o">*</span> <a href="https://docs.python.org/3/library/functions.html#float" title="builtins.float" class="sphx-glr-backref-module-builtins sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">np</span><span class="o">.</span><span class="n">pi</span></a><span class="p">,</span> <span class="mi">120</span><span class="p">)</span>
<a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray" title="numpy.ndarray" class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">y</span></a> <span class="o">=</span> <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html#numpy.linspace" title="numpy.linspace" class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-function"><span class="n">np</span><span class="o">.</span><span class="n">linspace</span></a><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">2</span> <span class="o">*</span> <a href="https://docs.python.org/3/library/functions.html#float" title="builtins.float" class="sphx-glr-backref-module-builtins sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">np</span><span class="o">.</span><span class="n">pi</span></a><span class="p">,</span> <span class="mi">100</span><span class="p">)</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>

<span class="c1"># ims is a list of lists, each row is a list of artists to draw in the</span>
<span class="c1"># current frame; here we are just animating one artist, the image, in</span>
<span class="c1"># each frame</span>
<a href="https://docs.python.org/3/library/stdtypes.html#list" title="builtins.list" class="sphx-glr-backref-module-builtins sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">ims</span></a> <span class="o">=</span> <span class="p">[]</span>
<span class="k">for</span> <a href="https://docs.python.org/3/library/functions.html#int" title="builtins.int" class="sphx-glr-backref-module-builtins sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">i</span></a> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">60</span><span class="p">):</span>
    <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray" title="numpy.ndarray" class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">x</span></a> <span class="o">+=</span> <a href="https://docs.python.org/3/library/functions.html#float" title="builtins.float" class="sphx-glr-backref-module-builtins sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">np</span><span class="o">.</span><span class="n">pi</span></a> <span class="o">/</span> <span class="mf">15.</span>
    <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray" title="numpy.ndarray" class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">y</span></a> <span class="o">+=</span> <a href="https://docs.python.org/3/library/functions.html#float" title="builtins.float" class="sphx-glr-backref-module-builtins sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">np</span><span class="o">.</span><span class="n">pi</span></a> <span class="o">/</span> <span class="mf">20.</span>
    <a href="../../api/image_api.html#matplotlib.image.AxesImage" title="matplotlib.image.AxesImage" class="sphx-glr-backref-module-matplotlib-image sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">im</span></a> <span class="o">=</span> <a href="../../api/_as_gen/matplotlib.axes.Axes.imshow.html#matplotlib.axes.Axes.imshow" title="matplotlib.axes.Axes.imshow" class="sphx-glr-backref-module-matplotlib-axes sphx-glr-backref-type-py-method"><span class="n">ax</span><span class="o">.</span><span class="n">imshow</span></a><span class="p">(</span><span class="n">f</span><span class="p">(</span><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray" title="numpy.ndarray" class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">x</span></a><span class="p">,</span> <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray" title="numpy.ndarray" class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">y</span></a><span class="p">),</span> <span class="n">animated</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="k">if</span> <a href="https://docs.python.org/3/library/functions.html#int" title="builtins.int" class="sphx-glr-backref-module-builtins sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">i</span></a> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
        <a href="../../api/_as_gen/matplotlib.axes.Axes.imshow.html#matplotlib.axes.Axes.imshow" title="matplotlib.axes.Axes.imshow" class="sphx-glr-backref-module-matplotlib-axes sphx-glr-backref-type-py-method"><span class="n">ax</span><span class="o">.</span><span class="n">imshow</span></a><span class="p">(</span><span class="n">f</span><span class="p">(</span><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray" title="numpy.ndarray" class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">x</span></a><span class="p">,</span> <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray" title="numpy.ndarray" class="sphx-glr-backref-module-numpy sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">y</span></a><span class="p">))</span>  <span class="c1"># show an initial one first</span>
    <a href="https://docs.python.org/3/library/stdtypes.html#list" title="builtins.list" class="sphx-glr-backref-module-builtins sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">ims</span></a><span class="o">.</span><span class="n">append</span><span class="p">([</span><a href="../../api/image_api.html#matplotlib.image.AxesImage" title="matplotlib.image.AxesImage" class="sphx-glr-backref-module-matplotlib-image sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">im</span></a><span class="p">])</span>

<a href="../../api/_as_gen/matplotlib.animation.ArtistAnimation.html#matplotlib.animation.ArtistAnimation" title="matplotlib.animation.ArtistAnimation" class="sphx-glr-backref-module-matplotlib-animation sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">ani</span></a> <span class="o">=</span> <a href="../../api/_as_gen/matplotlib.animation.ArtistAnimation.html#matplotlib.animation.ArtistAnimation" title="matplotlib.animation.ArtistAnimation" class="sphx-glr-backref-module-matplotlib-animation sphx-glr-backref-type-py-class"><span class="n">animation</span><span class="o">.</span><span class="n">ArtistAnimation</span></a><span class="p">(</span><a href="../../api/figure_api.html#matplotlib.figure.Figure" title="matplotlib.figure.Figure" class="sphx-glr-backref-module-matplotlib-figure sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">fig</span></a><span class="p">,</span> <a href="https://docs.python.org/3/library/stdtypes.html#list" title="builtins.list" class="sphx-glr-backref-module-builtins sphx-glr-backref-type-py-class sphx-glr-backref-instance"><span class="n">ims</span></a><span class="p">,</span> <span class="n">interval</span><span class="o">=</span><span class="mi">50</span><span class="p">,</span> <span class="n">blit</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
                                <span class="n">repeat_delay</span><span class="o">=</span><span class="mi">1000</span><span class="p">)</span>

<span class="c1"># To save the animation, use e.g.</span>
<span class="c1">#</span>
<span class="c1"># ani.save(&quot;movie.mp4&quot;)</span>
<span class="c1">#</span>
<span class="c1"># or</span>
<span class="c1">#</span>
<span class="c1"># writer = animation.FFMpegWriter(</span>
<span class="c1">#     fps=15, metadata=dict(artist=&#39;Me&#39;), bitrate=1800)</span>
<span class="c1"># ani.save(&quot;movie.mp4&quot;, writer=writer)</span>

<a href="../../api/_as_gen/matplotlib.pyplot.show.html#matplotlib.pyplot.show" title="matplotlib.pyplot.show" class="sphx-glr-backref-module-matplotlib-pyplot sphx-glr-backref-type-py-function"><span class="n">plt</span><span class="o">.</span><span class="n">show</span></a><span class="p">()</span>
</pre></div>
</div>
<p class="sphx-glr-timing"><strong>Total running time of the script:</strong> ( 0 minutes 7.473 seconds)</p>
<div class="sphx-glr-footer class sphx-glr-footer-example docutils container" id="sphx-glr-download-gallery-animation-dynamic-image-py">
<div class="sphx-glr-download sphx-glr-download-python docutils container">
<a class="reference download internal" download="" href="../../_downloads/c71a756999a5cacfd767880b3f5f5a19/dynamic_image.py"><code class="xref download docutils literal notranslate"><span class="pre">Download</span> <span class="pre">Python</span> <span class="pre">source</span> <span class="pre">code:</span> <span class="pre">dynamic_image.py</span></code></a></div>
<div class="sphx-glr-download sphx-glr-download-jupyter docutils container">
<a class="reference download internal" download="" href="../../_downloads/85eea2a9b9c6a3a221733494325998c8/dynamic_image.ipynb"><code class="xref download docutils literal notranslate"><span class="pre">Download</span> <span class="pre">Jupyter</span> <span class="pre">notebook:</span> <span class="pre">dynamic_image.ipynb</span></code></a></div>
</div>
<p class="sphx-glr-signature">Keywords: matplotlib code example, codex, python plot, pyplot
<a class="reference external" href="https://sphinx-gallery.readthedocs.io">Gallery generated by Sphinx-Gallery</a></p>
</div>
</div>
</div>
</div>
<div class="clearer"></div>
</div>

## Summary

- Value iteration is an algorithm for calculating a  value function $V$, from which a policy can be extracted using policy extraction.

- It produces an optimal policy  an infinite amount of time.

- For medium-scale problems, it works well, but as the state-space grows, it does not scale well.
