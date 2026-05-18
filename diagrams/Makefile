PUML_FILES := $(wildcard *.puml)

PNG_FILES  := $(PUML_FILES:.puml=.png)
SVG_FILES  := $(PUML_FILES:.puml=.svg)

all: $(PNG_FILES) $(SVG_FILES)

%.png: %.puml
	plantuml -tpng $<

%.svg: %.puml
	plantuml -tsvg $<

clean:
	rm -f $(PNG_FILES) $(SVG_FILES)

.PHONY: all clean
