import os
import glob
import re
import argparse
from pathlib import Path
from PyPDF2 import PdfReader

def clean_filename(title):
    # Remove invalid characters for Windows filenames
    title = re.sub(r'[<>:"/\\|?*\n\r\t]', '', title)
    title = title.strip()
    return title[:120].strip() # Limit length

def get_pdf_title(reader, filepath):
    try:
        meta = reader.metadata
        if meta and meta.title:
            title = meta.title.strip()
            if len(title) > 5 and 'untitled' not in title.lower():
                return clean_filename(title)
    except Exception:
        pass
    
    # Fallback: read first page
    try:
        page = reader.pages[0]
        text = page.extract_text()
        if text:
            first_line = text.split('\n')[0].strip()
            if len(first_line) > 5:
                return clean_filename(first_line)
    except Exception:
        pass
    
    # Ultimate fallback: base filename without extension
    return clean_filename(os.path.basename(filepath).replace('.pdf', ''))

def main():
    parser = argparse.ArgumentParser(description='Convert PDFs to Markdown and rename files with numbers and titles')
    parser.add_argument('--dir', '-d', help='Target directory containing PDFs', default=None)
    parser.add_argument('--start', '-s', help='Starting index for numbering (overrides auto-detect)', type=int, default=None)
    args = parser.parse_args()

    # Default to repository relative pre_thesis/website if not provided
    if args.dir:
        target_dir = Path(args.dir)
    else:
        target_dir = Path(__file__).resolve().parent.parent / 'pre_thesis' / 'website'

    if not target_dir.exists():
        print(f"Target directory does not exist: {target_dir}")
        return

    pdf_files = sorted(target_dir.glob("*.pdf"))

    # Determine starting index: use --start if provided, otherwise detect from existing numbered files
    if args.start and args.start > 0:
        current_index = args.start
    else:
        # Scan for existing files like "<number>- Title.*" and find max number
        max_idx = 0
        for p in target_dir.iterdir():
            m = re.match(r'^(\d+)\s*-', p.name)
            if m:
                try:
                    idx = int(m.group(1))
                    if idx > max_idx:
                        max_idx = idx
                except Exception:
                    pass
        current_index = max_idx + 1 if max_idx >= 1 else 1
    
    for pdf_path in pdf_files:
        filename = pdf_path.name

        # Skip already renamed ones if they exist (format: "<number>- ...")
        if re.match(r'^\d+\s*-', filename):
            print(f"Skipping already-numbered file: {filename}")
            continue

        print(f"Processing: {filename}")

        try:
            reader = PdfReader(str(pdf_path))
            title = get_pdf_title(reader, str(pdf_path))

            if not title:
                title = f"Document_{current_index}"

            # Construct new base name: number - title
            new_base_name = f"{current_index} - {title}"

            # Paths
            new_pdf_path = target_dir / f"{new_base_name}.pdf"
            new_md_path = target_dir / f"{new_base_name}.md"

            # Extract full text
            full_text = []
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                except Exception:
                    page_text = None
                if page_text:
                    # simple markdown escape for lines
                    full_text.append(f"## Page {i+1}\n\n{page_text}")

            md_content = f"# {title}\n\n" + "\n\n".join(full_text)

            # Save MD
            with open(new_md_path, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(md_content)

            # Ensure reader dereferenced
            del reader

            # Rename PDF
            pdf_path.rename(new_pdf_path)

            print(f" -> Converted and renamed to: {new_base_name}")
            current_index += 1

        except Exception as e:
            print(f"Failed to process {filename}: {str(e)}")

if __name__ == '__main__':
    main()
