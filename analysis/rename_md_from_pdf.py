#!/usr/bin/env python3
"""
rename_md_from_pdf.py
=====================
Reads the first page of each badly-named PDF in paper/references/
to extract the real paper title, then renames the corresponding .md
file to the canonical "NN - AuthorYEAR - Real Title.md" format.
Also generates a fresh .md from the full PDF text if the content
is still using the old header ("Published as a conference paper...").

Run from repo root:
    python analysis/rename_md_from_pdf.py [--dry-run]
"""

import re
import sys
import argparse
import pathlib

try:
    import fitz  # PyMuPDF — system Python
except ImportError:
    sys.exit("PyMuPDF not found. Run: pip install pymupdf")

REFS_DIR = pathlib.Path("paper/references")

# ----------------------------------------------------------------
# Manually-curated mapping: old filename prefix → new canonical name
# Derived from reading the actual PDF first pages.
# Format: (new_number, author_year, real_title)
# ----------------------------------------------------------------
RENAMES = {
    "3 - Published as a conference paper at ICLR 2017": (
        "03", "Kipf2017",
        "Semi-Supervised Classification with Graph Convolutional Networks"
    ),
    "7 - Published as a conference paper at ICLR 2018": (
        "07", "Velickovic2018",
        "Graph Attention Networks"
    ),
    "9 - Published as a conference paper at ICLR 2019": (
        "09", "Gasteiger2019",
        "Predict then Propagate - Graph Neural Networks meet Personalized PageRank (APPNP)"
    ),
    "10 - Published as a conference paper at ICLR 2019": (
        "10", "He2016",
        "Deep Residual Learning for Image Recognition"
    ),
    "12 - Published as a conference paper at ICLR 2020": (
        "12", "Zeng2020",
        "GraphSAINT Graph Sampling Based Inductive Learning Method"
    ),
    "13 - WELL-READ STUDENTS LEARN BETTER  ON THE IM-": (
        "14", "Turc2019",
        "Well-Read Students Learn Better On the Importance of Pre-training Compact Models (BERT-Tiny)"
    ),
    "20 - Published as a conference paper at ICLR 2021": (
        "20", "Abadal2021",
        "Computing Graph Neural Networks A Survey from Algorithms to Accelerators"
    ),
    "24 - Published as a conference paper at ICLR 2022": (
        "24", "Tailor2022",
        "Do We Need Anisotropic Graph Neural Networks (EGC)"
    ),
    "25 - Published as a conference paper at ICLR 2022": (
        "25", "Thomas2023",
        "Graph Neural Networks Designed for Different Graph Types A Survey"
    ),
    "27 - Published as a conference paper at ICLR 2022": (
        "27", "Shirzad2023",
        "Exphormer Sparse Transformers for Graphs"
    ),
    "30 - Published in Transactions on Machine Learning Research (032023)": (
        "30", "Thomas2023b",
        "Graph Neural Networks Designed for Different Graph Types A Survey (TMLR)"
    ),
    "31 - 2210.04055v1": (
        "31", "Tonshoff2023",
        "Where Did the Gap Go Reassessing the Long-Range Graph Benchmark"
    ),
    "36 - 2501.00636v1": (
        "36", "LightGNN2025",
        "LightGNN Simple Graph Neural Network for Recommendation"
    ),
    "23 - Corporate PowerPoint Template Use IntelOne Fonts For All Text  (General Employee Usage)": (
        None, None, None  # DELETE — not an academic paper
    ),
}


def safe_filename(s: str, max_len: int = 90) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s[:max_len].rstrip() if len(s) > max_len else s


def extract_full_text(pdf_path: pathlib.Path) -> str:
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        pages.append(f"## Page {i+1}\n\n{page.get_text()}")
    return "\n\n".join(pages)


def run(dry_run: bool = False) -> None:
    print(f"{'[DRY RUN] ' if dry_run else ''}Processing {REFS_DIR}\n")

    for old_prefix, (num, slug, title) in RENAMES.items():
        # Find the existing .md file
        md_matches = list(REFS_DIR.glob(f"{old_prefix}*.md"))
        pdf_matches = list(REFS_DIR.glob(f"{old_prefix}*.pdf"))

        if not md_matches and not pdf_matches:
            print(f"SKIP   (not found): {old_prefix!r}")
            continue

        # Handle DELETE case (Intel PPT)
        if num is None:
            for f in md_matches + pdf_matches:
                print(f"DELETE {f.name}")
                if not dry_run:
                    f.unlink()
            continue

        new_name_stem = f"{num} - {slug} - {safe_filename(title)}"
        new_md_path = REFS_DIR / f"{new_name_stem}.md"

        if md_matches:
            old_md = md_matches[0]
            if old_md == new_md_path:
                print(f"OK     (unchanged): {old_md.name}")
                continue

            # Read existing content and update the H1 header
            content = old_md.read_text(encoding="utf-8", errors="replace")
            # Replace the first H1 if it's the old filename
            first_line = content.split('\n')[0].strip()
            if first_line.startswith('#') and (
                'Published as' in first_line or
                'WELL-READ' in first_line or
                '2210.' in first_line or
                '2501.' in first_line or
                'Corporate' in first_line or
                'Transactions on Machine Learning' in first_line
            ):
                content = f"# {title}\n\n" + '\n'.join(content.split('\n')[1:])

            print(f"RENAME {old_md.name}")
            print(f"    -> {new_md_path.name}")
            if not dry_run:
                new_md_path.write_text(content, encoding="utf-8")
                old_md.unlink()

        elif pdf_matches:
            # MD doesn't exist yet — generate from PDF
            pdf = pdf_matches[0]
            print(f"GENERATE from PDF: {pdf.name}")
            print(f"              ->  {new_md_path.name}")
            if not dry_run:
                full_text = extract_full_text(pdf)
                header = f"# {title}\n\n<!-- SOURCE: {pdf.name} -->\n\n"
                new_md_path.write_text(header + full_text, encoding="utf-8")
        else:
            print(f"SKIP   (md only, no pdf): {old_prefix!r}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
