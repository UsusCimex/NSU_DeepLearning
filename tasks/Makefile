ENGINE ?= pdflatex
DIRS := $(wildcard task-*)
PDFS := $(patsubst %,%/task.pdf,$(DIRS))

.PHONY: all clean

all: $(PDFS)

%/task.pdf: %/task.tex common/preamble.tex
	cd $(dir $<) && latexmk -$(ENGINE) -interaction=nonstopmode -halt-on-error task.tex

clean:
	latexmk -C task-*/task.tex
