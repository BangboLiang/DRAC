# Paper synchronization status

The repository contains `paper.pdf` but no LaTeX source (`*.tex`, bibliography,
class/style files, or figure build manifest).  The PDF is therefore treated as an
immutable source-of-truth artifact.  Editing or replacing it without its source
would destroy reproducibility and could silently alter fonts, references, and
layout.

Completed preparatory work:

- Figure 6--11 PDFs and PNGs are generated from revised result CSVs.
- Table II is generated as LaTeX from the target/runtime validation CSV.
- `paper_evaluation_revision.tex` contains a source-ready Evaluation replacement
  with the required A--H structure and honest measurement TODO.
- obsolete metrics and figures are mapped in `evaluation_migration_plan.md`.

Still blocked until the current paper source project is supplied:

1. integrate the replacement section and figure/table paths;
2. update Abstract, Introduction, and Conclusion in their real context;
3. resolve labels and bibliography against the complete source;
4. compile and visually verify the revised paper PDF.

The existing `paper.pdf` was intentionally not overwritten.

