# Exact Computation Under Signed Program Authority

The [manuscript](paper.md) consolidates the NOR exact-computation research line. It contains the original engineering outcomes, algebraic derivations, an explicit assurance contract, and a reusable evaluation design. The [evidence ledger](../../research/nor_line_capstone/evidence_ledger.json) preserves each study's comparator and first verdict.

The bibliography uses primary sources for authenticated trees, Ed25519, component-cost reasoning, and systems benchmarking. [references.bib](references.bib) supplies the verified publication metadata.

A PDF can be built from this directory with Pandoc and a complete XeLaTeX installation:

```bash
pandoc paper.md --citeproc --lua-filter=table_layout.lua \
  --include-in-header=layout.tex --pdf-engine=xelatex \
  -V geometry:margin=0.85in -V fontsize=10pt -o paper.pdf
```

The figures and their plotted values are committed under `figures/`. With
Matplotlib and NumPy installed, `python build_figures.py` rebuilds both figures
directly from the byte-pinned first NOR60 archive. The builder checks the ZIP's
SHA-256 before reading any measurements.

Historical evidence remains pinned to its original source commit; final
experiment and gym artifacts carry their own receipts. The PDF was rendered
and inspected page by page for the capstone release.
