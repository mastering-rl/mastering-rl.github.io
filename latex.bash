jb build . --builder latex
cd _build/latex/
xelatex book
xelatex book
cd ../../
cp _build/latex/book.pdf _static/COMP90054-notes.pdf
cp _build/latex/book.pdf _build/html/_static/COMP90054-notes.pdf
cp _build/latex/book.pdf  book.pdf
