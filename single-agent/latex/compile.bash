#!/etc/bash

TEX=*.tex

for f in $TEX; do
	basename=`basename $f .tex`
	pdflatex $f
	pdftoppm -png $basename.pdf $basename
	mv $basename-1.png $basename.png
done

rm *.aux *.log *.pdf
