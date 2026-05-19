PUML_FILES := $(wildcard *.puml)

PNG_FILES  := $(PUML_FILES:.puml=.png)
SVG_FILES  := $(PUML_FILES:.puml=.svg)
PDF_FILES  := $(SVG_FILES:.svg=.pdf)

all: $(PNG_FILES) $(SVG_FILES) $(PDF_FILES)

%.png: %.puml
	plantuml -tpng $<

%.svg: %.puml
	plantuml -tsvg $<

%.pdf: %.svg
	rsvg-convert -f pdf -o $@ $<

clean:
	rm -f $(PNG_FILES) $(SVG_FILES) $(PDF_FILES)

.PHONY: all clean
