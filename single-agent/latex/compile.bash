#!/etc/bash

TEX=*.tex

for f in $TEX; do
	basename=`basename $f .tex`
	pdflatex $f
	pdftoppm -png $basename.pdf $basename
	mv $basename-1.png $basename.png
done

rm *.aux *.log *.pdf

convert -delay 100 -loop 0 n-step*.png ../../assets/gifs/n-step-window.gif

rm n-step-rl-[0-7].png
