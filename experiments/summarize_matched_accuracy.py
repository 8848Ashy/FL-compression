"""Validate preserved runs and produce comparison figures without selecting winners."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    with path.open(encoding='utf-8') as f: return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(); p.add_argument('run'); args=p.parse_args(); out=Path(args.run)
    rows=read(out/'rounds.csv'); ends=read(out/'endpoints.csv'); paired=read(out/'paired.csv'); targets=read(out/'targets.csv')
    assert len(rows)==2250 and len(ends)==45 and len(targets)==135
    for seed in range(5):
        for r in range(1,51):
            group=[x for x in rows if int(x['seed'])==seed and int(x['round'])==r]
            assert len(group)==9 and len({x['clients'] for x in group})==1
            for bits in (1,2):
                vals=[x for x in group if int(x['bits'])==bits]
                assert len(vals)==4 and {int(x['total_uplink_bits']) for x in vals}=={(65536*bits+64)*3*r}
    print('PASS: 5 seeds x 50 rounds x 9 configurations; identical participants and compressed payloads')
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),gridspec_kw={'width_ratios':[1.1,1]})
    colors={'Fourier-direct':'gray','Kashin-legacy':'tab:blue','Kashin-balanced':'tab:orange'}
    labels=[]
    for i,z in enumerate(paired):
        mean=float(z['delta_accuracy_pp']); low=float(z['ci_low']); high=float(z['ci_high'])
        values=np.array([float(s) for s in z['seed_deltas'].split(';')])
        axes[0].scatter(values,np.full(5,i)+np.linspace(-.12,.12,5),s=20,color='gray',alpha=.65)
        axes[0].errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt='D',color=colors[z['method']],capsize=4)
        labels.append(f'{z["method"]}, {z["bits"]}-bit')
    axes[0].set_yticks(range(len(labels)),labels); axes[0].invert_yaxis()
    axes[0].axvline(0,color='black',ls='--',lw=1)
    axes[0].set_xlabel('Accuracy difference vs SRK (percentage points)')
    axes[0].set_title('Round 50: paired seeds and unadjusted 95% t intervals')
    axes[0].grid(axis='x',alpha=.2)
    groups=['SRK','Kashin-legacy','Kashin-balanced']; xpos=np.arange(2)
    palette=['tab:red','tab:blue','tab:orange']
    for j,(m,c) in enumerate(zip(groups,palette)):
        means=[]; stds=[]
        for bits in (1,2):
            vals=[float(z['uplink_bits'])/8/1024/1024 for z in targets if z['label']==f'{m}-{bits}bit' and float(z['target'])==.92 and z['status']=='reached']
            assert len(vals)==5
            means.append(np.mean(vals)); stds.append(np.std(vals,ddof=1))
        axes[1].bar(xpos+(j-1)*.23,means,.23,yerr=stds,capsize=3,label=m,color=c)
    axes[1].set_xticks(xpos,['1-bit','2-bit']); axes[1].set_ylabel('Uplink to first 92% accuracy (MiB, lower is better)')
    axes[1].set_title('All 5 seeds reached target; mean +/- SD')
    axes[1].legend(fontsize=8); axes[1].grid(axis='y',alpha=.2)
    fig.tight_layout(); fig.savefig(out/'accuracy_and_communication.png',dpi=200); plt.close(fig)
    # Explicit fixed cumulative budgets: last affordable round, no config maximum.
    budgets=[.1,.25,.5,1.,2.]
    budget_rows=[]
    for budget in budgets:
        for label in sorted({z['label'] for z in rows}):
            for seed in range(5):
                affordable=[z for z in rows if z['label']==label and int(z['seed'])==seed and float(z['total_uplink_bits'])<=budget*8388608]
                if affordable:
                    point=max(affordable,key=lambda z:int(z['round']))
                    budget_rows.append(dict(budget_MiB=budget,label=label,seed=seed,
                        round=point['round'],accuracy=point['accuracy'],used_bits=point['total_uplink_bits'],
                        round_limit_reached=int(point['round'])==50))
    with (out/'fixed_budget_accuracy.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(budget_rows[0])); w.writeheader(); w.writerows(budget_rows)
    print(f'Wrote figures and fixed-budget per-configuration values to {out}')


if __name__=='__main__': main()
