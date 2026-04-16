TEXFILE = thesis.tex
PDFFILE = $(TEXFILE:.tex=.pdf)
ODTFILE = $(TEXFILE:.tex=.odt)
ODTFILE = $(TEXFILE:.tex=.odt)
DEPFILES = $(wildcard *.tex)
SVGS = $(wildcard figures/*.svg)
SVG_PDFS = $(SVGS:.svg=.pdf)
# SVG_EPSS = $(SVGS:.svg=.eps)

RSVG_CONVERT := $(shell command -v rsvg-convert 2> /dev/null)

.PHONY: $(PDFFILE) all figures clean

all: $(PDFFILE)

ifndef VERBOSE
.SILENT:

LATEXMK_SILENT	?= -silent
endif

$(SVG_PDFS): $(SVGS)

# $(SVG_EPSS): $(SVGS)

figures/%.pdf: figures/%.svg
ifndef RSVG_CONVERT
	@echo "INKSCAPE        $@"
	inkscape --export-pdf=$@ $<
else
	@echo "RSVG-CONVERT    $@"
	rsvg-convert --format=pdf --keep-aspect-ratio --output=$@ $<
endif

# figures/%.eps: figures/%.svg
	# @echo "INKSCAPE        $@"
#         inkscape --export-eps=$@ $<

figures: $(SVG_PDFS) # $(SVG_EPSS)

odt: $(ODTFILE)

$(ODTFILE): $(TEXFILE) $(DEPFILES) $(SVG_PDFS)
	pandoc $< -t odt -o $@

$(PDFFILE): $(TEXFILE) $(DEPFILES) $(SVG_PDFS) # $(SVG_EPSS)
	latexmk -synctex=1 -f $(LATEXMK_SILENT) -pdf -pdflatex="pdflatex -interactive=nonstopmode --shell-escape %O %S" $(TEXFILE)
# for gnuplot support: latexmk -pdf -pdflatex="pdflatex -interactive=nonstopmode --shell-escape" $(TEXFILE)
# For LuaTex: latexmk -pdf -pdflatex="lualatex -interactive=nonstopmode" $(TEXFILE)

clean:
	latexmk -CA
	rm -f $(TEXFILES:.tex=.synctex.gz) $(TEXFILE:.tex=.bbl) $(TEXFILE:.tex=.tdo)
	rm -f generated/* plots/*.tex
	rm -f $(SVG_PDFS) # $(SVG_EPSS)
	rm -f $(ODTFILE)
