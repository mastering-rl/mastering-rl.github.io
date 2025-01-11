jb build . --builder latex
cd _build/latex/
#xelatex book
#xelatex book
pdflatex book
pdflatex book
cd ../../
cp _build/latex/book.pdf _static/mastering_reinforcement_learning.pdf
cp _build/latex/book.pdf _build/html/_static/mastering_reinforcement_learning.pdf
cp _build/latex/book.pdf  book.pdf
