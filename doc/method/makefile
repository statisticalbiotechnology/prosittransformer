main=transform

all: ${main}.pdf # supplement.pdf coverletter.pdf biorxiv.pdf # diff.pdf

${main}.pdf: ${main}.tex ${main}.bib # img/*

diff.tex: ${main}.tex
	git show sub1:./$? > /tmp/$?
	latexdiff /tmp/$? $? > diff.tex


%.pdf: %.tex
	pdflatex -shell-escape $*.tex
	if ( grep -q citation $*.aux ) ; then \
		bibtex $* ; \
		pdflatex -shell-escape $*.tex ; \
	fi
	pdflatex -shell-escape $*.tex

html:
	rm -fr ${main}/
	latex2html -split 0 ${main}.tex

zip:
	zip -r ${main}.zip *.tex *.bib img makefile *.sty *.bst

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *~ .DS_Store
	rm -fr main/
