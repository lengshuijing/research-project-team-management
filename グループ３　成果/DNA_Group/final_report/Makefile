makepdf:
	pdflatex -output-directory=tmp tex/main
	bibtex tmp/main
	pdflatex -output-directory=tmp tex/main
	pdflatex -output-directory=tmp tex/main
	./cleanup
