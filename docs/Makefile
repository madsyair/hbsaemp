# Minimal makefile for Sphinx documentation.
#
# `-W` turns every warning into an error. Keep it here rather than in a
# wrapper script so a local build fails on exactly what CI fails on.
# `--keep-going` is deliberately absent: it has been the default since
# Sphinx 8.1 and the flag is retained only for compatibility.

SPHINXOPTS    ?= -W
SPHINXBUILD   ?= sphinx-build
SOURCEDIR      = source
BUILDDIR       = _build

.PHONY: help clean serve Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Remove the build output *and* the autosummary stubs, which are generated
# and never version-controlled.
clean:
	rm -rf "$(BUILDDIR)" "$(SOURCEDIR)/python/reference/generated"

serve: html
	python -m http.server -d "$(BUILDDIR)/html" 8000

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
