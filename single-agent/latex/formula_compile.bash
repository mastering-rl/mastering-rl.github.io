#!/etc/bash

pdflatex formula.tex
convert -trim -density 600 -background transparent formula.pdf formula.png
