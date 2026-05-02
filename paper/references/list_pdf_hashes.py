import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='List SHA256 hashes of PDFs in a directory')
    parser.add_argument('--dir', '-d', help='Target directory', default=None)
    parser.add_argument('--group', '-g', action='store_true', help='Show duplicate groups')
    args = parser.parse_args()

    if args.dir:
        target = Path(args.dir)
    else:
        target = Path(__file__).resolve().parent.parent / 'publications' / 'journal-article' / 'references' / 'topological_analysis'

    if not target.exists():
        print(f"Target directory not found: {target}")
        return

    pdfs = sorted(target.glob('*.pdf'), key=lambda p: p.name.lower())
    if not pdfs:
        print('No PDF files found in', target)
        return

    hashes = {}
    print('PDF count:', len(pdfs))
    for p in pdfs:
        try:
            h = sha256_file(p)
            size = p.stat().st_size
            print(f"{p.name}  {h}  {size} bytes")
            hashes.setdefault(h, []).append(p.name)
        except Exception as e:
            print(f"Failed to hash {p.name}: {e}")

    if args.group:
        print('\nDuplicate groups (identical SHA256):')
        found = False
        for h, files in hashes.items():
            if len(files) > 1:
                found = True
                print(f"Hash: {h}")
                for fn in files:
                    print('  -', fn)
        if not found:
            print('No identical-hash duplicates found.')

if __name__ == '__main__':
    main()
