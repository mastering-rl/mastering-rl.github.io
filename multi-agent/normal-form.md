# Normal form games


the known initial state $s_0$ or any state reachable from $s_0$ using
actions defined in the model. Whereas value/policy iteration methods
work from any state.

<center>

|                       |  C             | D           |
|  :--------------------- | :--------------------------: | :-------------:
| C                  | -1, -1     | -4, 0
| D                  | 0, -4      | -2, -2 | 

</center>

\begin{gather*}
a_1=b_1+c_1\\
a_2=b_2+c_2-d_2+e_2
\end{gather*}

\begin{align}
a_{11}& =b_{11}&
  a_{12}& =b_{12}\\
a_{21}& =b_{21}&
  a_{22}& =b_{22}+c_{22}
\end{align}

<div class="amsmath math notranslate nohighlight">
\[\begin{array}{|c|c|}
   \multicolumn{2}{c}{Player 1}\\
\hline
  -1,-1 & -4,0\\
\hline
   0,-4 & -2,-2\\
\hline
\end{array}\]
</div>