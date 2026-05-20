import nbformat as nbf
from pathlib import Path

code = Path('npu_gnn_benchmarking_v2.py').read_text(encoding='utf-8')

# Split by cell markers
cell_blocks = code.split('# =============================================================================')

cells = []
for block in cell_blocks:
    block = block.strip()
    if not block:
        continue
    
    lines = block.split('\n')
    
    # Check if it's a markdown/description cell (starts with # or """)
    if lines[0].startswith('# CELL') or lines[0].startswith('# ----'):
        # Extract title and description
        title_line = [l for l in lines if l.startswith('# CELL')]
        if title_line:
            # This is a code cell with header
            cell_code = '\n'.join([l for l in lines if not l.startswith('# ===') and not l.startswith('# ----')])
            cells.append(nbf.v4.new_code_cell(cell_code.strip()))
    elif '"""' in block:
        # Split docstring from code
        parts = block.split('"""')
        if len(parts) >= 3:
            # Markdown part
            md_text = parts[1].strip()
            cells.append(nbf.v4.new_markdown_cell(md_text))
            # Code part
            code_text = '"""'.join(parts[2:]).strip()
            if code_text:
                cells.append(nbf.v4.new_code_cell(code_text))
        else:
            cells.append(nbf.v4.new_code_cell(block))
    else:
        # Regular code cell
        if block.strip():
            cells.append(nbf.v4.new_code_cell(block))

# Create notebook
nb = nbf.v4.new_notebook()
nb['cells'] = cells

# Write
output_path = Path('npu_gnn_benchmarking_v2.ipynb')
nbf.write(nb, str(output_path))
print(f"Created: {output_path.absolute()}")
print(f"Cells: {len(cells)}")
