# ACM e-Energy 2027 (Winter) — working paper build (acmart sigconf)
#
# Requires: latexmk, pdflatex, acmart.cls (texlive-full provides it).
# Usage:
#   make            build main.pdf
#   make watch      continuous build (latexmk -pvc)
#   make clean      remove build artifacts
#   make fullclean  clean + remove main.pdf

MAIN = main
LATEXMK = latexmk
LATEXMKFLAGS = -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all watch clean fullclean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	$(LATEXMK) $(LATEXMKFLAGS) $(MAIN).tex

watch:
	$(LATEXMK) $(LATEXMKFLAGS) -pvc $(MAIN).tex

clean:
	$(LATEXMK) -c $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN).blg $(MAIN).aux $(MAIN).out $(MAIN).log \
	      $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).nav $(MAIN).snm

fullclean: clean
	rm -f $(MAIN).pdf
