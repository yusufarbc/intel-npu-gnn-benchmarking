
import json
from pathlib import Path

def clean_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # 1. SGC Filtresini ekle
            if 'def load_clean_data():' in source:
                new_source = source.replace(
                    "df[~df['model'].str.contains('GAT', case=False, na=False)]",
                    "df[~df['model'].str.contains('GAT|SGC', case=False, na=False)]"
                )
                cell['source'] = [line + '\n' for line in new_source.split('\n') if line]
            
            # 2. FINAL_ isimlerini temizle
            if 'results/FINAL_' in source:
                new_source = source.replace('results/FINAL_', 'results/')
                cell['source'] = [line + '\n' for line in new_source.split('\n') if line]

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Notebook {nb_path} cleaned successfully.")

if __name__ == "__main__":
    clean_notebook("npu_gnn_benchmarking.ipynb")
