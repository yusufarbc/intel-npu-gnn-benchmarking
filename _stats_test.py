import pandas as pd
import numpy as np
from scipy import stats

models = ['GCN', 'GAT', 'GATv2', 'GraphTransformer', 'APPNP', 'SGC', 'GraphSAGE', 'GIN']
results = []

for model in models:
    path = f'c:/Users/yusuf/Projects/npu-graph-opt-benchmarking/results/{model}/scalability_matrix.csv'
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"{model}: scalability_matrix.csv not found — skipping")
        continue
    
    for ds in df['dataset'].unique():
        cpu = df[(df['dataset'] == ds) & (df['device'] == 'CPU') & (df['precision'] == 'fp32')]
        npu = df[(df['dataset'] == ds) & (df['device'] == 'NPU') & (df['precision'] == 'fp32')]
        gpu = df[(df['dataset'] == ds) & (df['device'] == 'GPU') & (df['precision'] == 'fp32')]
        
        n = 300  # 100 iter × 3 repeats
        
        if not cpu.empty and not npu.empty:
            cm = cpu['o_mean_ms'].values[0]
            cs = cpu['o_std_ms'].values[0]
            nm = npu['o_mean_ms'].values[0]
            ns = npu['o_std_ms'].values[0]
            
            se = np.sqrt(cs**2/n + ns**2/n)
            t = (cm - nm) / se
            df_w = (cs**2/n + ns**2/n)**2 / ((cs**2/n)**2/(n-1) + (ns**2/n)**2/(n-1))
            p = 2 * stats.t.sf(abs(t), df=df_w)
            pooled = np.sqrt((cs**2 + ns**2) / 2)
            d = abs(cm - nm) / pooled
            
            results.append({
                'model': model, 'dataset': ds, 'comparison': 'CPU vs NPU',
                'cpu_ms': round(cm,2), 'npu_ms': round(nm,2),
                'diff_pct': round(abs(cm-nm)/cm*100,1),
                't': round(t,3), 'p': '{:.2e}'.format(p),
                'sig_005': 'Yes' if p < 0.05 else 'No',
                'cohens_d': round(d,3)
            })
        
        if not gpu.empty and not npu.empty:
            gm = gpu['o_mean_ms'].values[0]
            gs = gpu['o_std_ms'].values[0]
            
            se2 = np.sqrt(gs**2/n + ns**2/n)
            t2 = (gm - nm) / se2
            df_w2 = (gs**2/n + ns**2/n)**2 / ((gs**2/n)**2/(n-1) + (ns**2/n)**2/(n-1))
            p2 = 2 * stats.t.sf(abs(t2), df=df_w2)
            pooled2 = np.sqrt((gs**2 + ns**2) / 2)
            d2 = abs(gm - nm) / pooled2
            
            results.append({
                'model': model, 'dataset': ds, 'comparison': 'GPU vs NPU',
                'cpu_ms': round(gm,2), 'npu_ms': round(nm,2),
                'diff_pct': round(abs(gm-nm)/gm*100,1),
                't': round(t2,3), 'p': '{:.2e}'.format(p2),
                'sig_005': 'Yes' if p2 < 0.05 else 'No',
                'cohens_d': round(d2,3)
            })

print(f"{'Model':15s} {'Dataset':15s} {'Comparison':15s} {'Dev1(ms)':10s} {'Dev2(ms)':10s} {'Diff%':8s} {'t':8s} {'p':12s} {'Sig':5s} {'d':8s}")
print("="*110)
for r in results:
    dev1 = r['cpu_ms'] if r['comparison'] == 'CPU vs NPU' else r['cpu_ms']
    dev2 = r['npu_ms']
    print(f"{r['model']:15s} {r['dataset']:15s} {r['comparison']:15s} {dev1:<10.2f} {dev2:<10.2f} {r['diff_pct']:<8.1f} {r['t']:<8.3f} {r['p']:<12s} {r['sig_005']:<5s} {r['cohens_d']:<8.3f}")
