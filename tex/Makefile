all: presentation report
presentation:
	pdflatex presentation.tex
report:
	pdflatex report.tex
	pdflatex report.tex
	bibtex report
	pdflatex report.tex
clean:
	rm *.snm *.toc *.vrb *.log *.blg *.toc *.aux *.nav *.bbl *.out report.pdf presentation.pdf
