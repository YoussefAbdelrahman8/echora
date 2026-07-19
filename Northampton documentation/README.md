# Echora — UON Group Project Report (ENG4006, AS2)

LaTeX sources for the **University of Northampton group report** (the shorter, UON-marked
document — *not* the long AASTMT thesis). Formatting follows *Guidelines on the Format of the
Final Report*: English, Times New Roman 12 pt, 1.5 line spacing, justified body, numbered
chapters starting on new pages, and Harvard referencing.

## Structure

```
Northampton documentation/
├── main.tex                     # preamble + document assembly (edit formatting here)
├── chapters/
│   ├── 00_titlepage.tex         # Preliminaries
│   ├── 00_abstract.tex
│   ├── 00_acknowledgements.tex
│   ├── 00_glossary.tex
│   ├── 01_introduction.tex      # Ch 1  (rubric: Introduction, 5)
│   ├── 02_literature_review.tex # Ch 2  (rubric: Literature Review, 10)
│   ├── 03_project_development.tex# Ch 3 (rubric: Project Development, 15)
│   ├── 04_results.tex           # Ch 4  (rubric: Project Results, 15)
│   ├── 05_conclusions.tex       # Ch 5  (rubric: Conclusions, 10)
│   ├── 06_references.tex        # Harvard reference list (rubric: References, 5)
│   └── 07_appendices.tex        # mode screenshots, build photos, PR curve
└── images/                      # all figures (self-contained; copied from the repo)
```

## Building the PDF

**With Tectonic (recommended — self-contained, no TeX install needed):**

```bash
brew install tectonic       # once
tectonic main.tex           # produces Echora Group Proposal(Youssef Abdelrahman_24805073-Sherif hossam Amin_24805076-Abdallah Mohamed Ezzat_24809369).pdf
```

**With a full TeX distribution (TeX Live / MacTeX):**

```bash
pdflatex main
pdflatex main               # run twice so the ToC and lists resolve
```

## Word count

The body (Chapters 1–5) is ~6,950 words, within the 7,500 ± 10% group-report limit.
Preliminaries, tables, figures, references, and appendices are outside the count.

## Notes for editing

- The chapters condense the full AASTMT thesis in `../documentation/thesis/` for the shorter
  UON word limit and its different marking criteria — it is **not** a copy-paste of that thesis.
- To change fonts, spacing, or heading sizes, edit the preamble in `main.tex`.
- To add a figure, drop the file in `images/` and `\includegraphics{filename}` (the graphics
  path is already set to `images/`).
- References are Harvard style; add new entries with the `\refent{...}` macro in
  `06_references.tex` and cite in text as `(Author, year)`.
