TEX = main.tex
PDF = $(TEX:.tex=.pdf)
ASSETS = assets

# Figures used in the paper
FIGURES = \
	savings_vs_overhead_all \
	threshold_space_DeepSeek_V3_DE_0101 \
	margin_vs_best_Kimi_K2 \
	threshold_space_overview \
	margin_vs_best_overview \
	multiyear_avg_score

EPS_TARGETS = $(addprefix $(ASSETS)/, $(addsuffix .eps, $(FIGURES)))

all: $(PDF)

$(PDF): $(EPS_TARGETS) $(TEX)
	latexmk -pdfps $(TEX)

# Convert SVG -> EPS for the latex -> dvi -> ps -> pdf pipeline
$(ASSETS)/%.eps: $(ASSETS)/%.svg
	rsvg-convert -f eps -o $@ $<

.PHONY: clean all

clean:
	latexmk -c $(TEX)
	rm -f *.bbl *.dvi *.ps main.pdf
	rm -f $(addprefix $(ASSETS)/, *.eps)
