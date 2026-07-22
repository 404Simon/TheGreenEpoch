MAIN := main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex preamble.sty $(wildcard sections/*.tex) $(wildcard styles/*) $(wildcard *.bib)
	latexmk -pdf -interaction=nonstopmode $(MAIN)

clean:
	latexmk -C
	rm -f *.bbl *.run.xml

fmt:
	tex-fmt *.tex --recursive

.PHONY: all clean fmt
