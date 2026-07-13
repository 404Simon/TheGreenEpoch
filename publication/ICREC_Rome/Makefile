TEX = main.tex
PDF = $(TEX:.tex=.pdf)

all: $(PDF)

$(PDF): $(TEX)
	latexmk -pdfps $(TEX)

clean:
	latexmk -c $(TEX)
	rm -f *.bbl *.dvi *.ps main.pdf

.PHONY: all clean
