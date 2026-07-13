TEX = main.tex
PDF = $(TEX:.tex=.pdf)
SVG_SRC = ../output/figures
ASSETS = assets

# Figures used in the paper
FIGURES = \
	savings_vs_overhead_all \
	threshold_space_overview \
	margin_vs_best_overview \
	multiyear_avg_score

SVG_TARGETS = $(addprefix $(ASSETS)/, $(addsuffix .svg, $(FIGURES)))
EPS_TARGETS = $(addprefix $(ASSETS)/, $(addsuffix .eps, $(FIGURES)))

all: $(PDF)

.PRECIOUS: $(SVG_TARGETS)

$(PDF): $(EPS_TARGETS) $(TEX)
	latexmk -pdfps $(TEX)

# Copy SVGs from publication/output/figures/ to assets/
$(ASSETS)/%.svg: $(SVG_SRC)/%.svg
	cp $< $@

# Convert SVG -> EPS for the latex -> dvi -> ps -> pdf pipeline
$(ASSETS)/%.eps: $(ASSETS)/%.svg
	rsvg-convert -f eps -o $@ $<

# Regenerate SVGs from the simulation results
svgs:
	cd ../.. && uv run scripts/visualize_results.py

.PHONY: clean all svgs

clean:
	latexmk -c $(TEX)
	rm -f *.bbl *.dvi *.ps main.pdf
	rm -f $(addprefix $(ASSETS)/, *.svg *.eps)
