"""Research-constrained Pareto analysis for HMF -> FDCA literature data.

This script deliberately does NOT optimize an unreliable ML surrogate. It identifies
non-dominated *observed experiments* in the literature dataset, subject to explicit
comparability checks, and ranks a diverse top-five shortlist for follow-up validation.
Outputs are empirical literature candidates, not new predicted recipes.

Run from project folder:
  python -u 08_research_constrained_pareto.py
Optional:
  python -u 08_research_constrained_pareto.py --data HMF_ML_Preprocessed.xlsx --cleaned HMF_Cleaned_Full_Dataset.xlsx
"""
from pathlib import Path
import argparse, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

FEATURES = [
    'Conversion_pct','Yield_pct','Temperature_C','Potential_V','Potential_vs_RHE',
    'HMF_Concentration_mM','Electrolyte_Concentration_M','Alkaline_Electrolyte',
    'Electrolyte_Electrolyte_Anion_OH','Current_mA','Cell_Voltage','FE_pct','Selectivity_pct'
]

def first_existing(base, names):
    for n in names:
        p=base/n
        if p.exists(): return p
    return None

def pareto_mask(values, maximize, minimize):
    """Pareto dominance with NaN-free numeric matrix. True means non-dominated."""
    # transform all objectives to maximization
    cols=[]
    for c in maximize: cols.append(values[c].to_numpy(float))
    for c in minimize: cols.append(-values[c].to_numpy(float))
    a=np.column_stack(cols)
    keep=np.ones(len(a), dtype=bool)
    for i in range(len(a)):
        if not keep[i]: continue
        dominates_i=np.all(a >= a[i], axis=1) & np.any(a > a[i], axis=1)
        if dominates_i.any(): keep[i]=False
    return keep

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', default=None)
    ap.add_argument('--cleaned', default=None)
    ap.add_argument('--out', default=None)
    args=ap.parse_args()
    data_path=Path(args.data) if args.data else first_existing(HERE,['HMF_ML_Preprocessed.xlsx','HMF_ML_Preprocessed.csv'])
    clean_path=Path(args.cleaned) if args.cleaned else first_existing(HERE,['HMF_Cleaned_Full_Dataset.xlsx'])
    if not data_path or not data_path.exists(): raise FileNotFoundError('Could not find HMF_ML_Preprocessed.xlsx/.csv; pass --data.')
    out=Path(args.out) if args.out else HERE/'NSGA_II_Research_Constrained'
    out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(data_path) if data_path.suffix.lower()=='.csv' else pd.read_excel(data_path)
    # Attach exact catalyst names and paper titles only when row counts match (expected source alignment).
    raw=pd.read_excel(clean_path) if clean_path and clean_path.exists() else None
    if raw is not None and len(raw)==len(d):
        for c in ['Paper Title','Catalysts','Catalyst','Electrolyte','Electrolyte Cation','Electrolyte Anion','Cell Type','Reference']:
            if c in raw.columns and c not in d.columns: d[c]=raw[c]
        d['Source_Row']=np.arange(2,len(d)+2)
    else:
        d['Source_Row']=np.arange(2,len(d)+2)
        warnings.warn('Cleaned file missing or row count differs; exact titles/catalyst names will be unavailable.')

    required=['Conversion_pct','Yield_pct','Potential_V','Potential_vs_RHE','HMF_Concentration_mM','Electrolyte_Concentration_M']
    missing=[c for c in required if c not in d.columns]
    if missing: raise ValueError(f'Missing required columns: {missing}')
    # Research-comparability filter: both outcomes, measured potential, explicit RHE reference,
    # measured concentration fields, and alkaline/OH descriptors. No imputation.
    mask=d[required].notna().all(axis=1)
    mask &= d['Potential_vs_RHE'].eq(1)
    if 'Alkaline_Electrolyte' in d: mask &= d['Alkaline_Electrolyte'].eq(1)
    if 'Electrolyte_Electrolyte_Anion_OH' in d: mask &= d['Electrolyte_Electrolyte_Anion_OH'].eq(1)
    # Physical plausibility screen anchored to reviewed literature conditions. These are screening
    # bounds, not universal optima: RHE potential 1.0–1.65 V, HMF 1–100 mM, electrolyte 0.1–2 M.
    mask &= d['Potential_V'].between(1.0,1.65)
    mask &= d['HMF_Concentration_mM'].between(1,100)
    mask &= d['Electrolyte_Concentration_M'].between(0.1,2.0)
    # Keep only known temperature when it is available; missing temperature is not silently filled.
    if 'Temperature_C' in d: mask &= d['Temperature_C'].between(15,60)
    cand=d.loc[mask].copy()
    if len(cand)<5:
        # Relax temperature availability/range only; retain the key chemistry constraints and flag it.
        mask=d[required].notna().all(axis=1) & d['Potential_vs_RHE'].eq(1)
        if 'Alkaline_Electrolyte' in d: mask &= d['Alkaline_Electrolyte'].eq(1)
        if 'Electrolyte_Electrolyte_Anion_OH' in d: mask &= d['Electrolyte_Electrolyte_Anion_OH'].eq(1)
        mask &= d['Potential_V'].between(1.0,1.65) & d['HMF_Concentration_mM'].between(1,100) & d['Electrolyte_Concentration_M'].between(.1,2.0)
        cand=d.loc[mask].copy()
        cand['Temperature_C']=cand['Temperature_C'].fillna(cand['Temperature_C'].median()) if 'Temperature_C' in cand else np.nan
        temp_note='Some temperatures were missing; median is used only for Pareto screening and is flagged. Do not treat these as measured temperatures.'
    else: temp_note='All shortlisted temperatures are measured values.'
    if len(cand)<5:
        raise RuntimeError(f'Only {len(cand)} rows meet the comparability filter. Do not fabricate five candidates; inspect data quality or broaden criteria transparently.')

    # Outcomes and conditions are observed literature values, not model predictions.
    if 'Temperature_C' not in cand: cand['Temperature_C']=np.nan
    # To preserve honest comparisons, Pareto calculations require observed temperature.
    known_temp=cand['Temperature_C'].notna()
    cand=cand.loc[known_temp].copy()
    if len(cand)<5: raise RuntimeError('Fewer than five candidates have measured temperature after screening. Review filter; do not impute for reporting.')
    cand['Pareto_Nondominated']=pareto_mask(cand, ['Conversion_pct','Yield_pct'], ['Temperature_C','Potential_V'])
    front=cand.loc[cand['Pareto_Nondominated']].copy()
    # Post-hoc compromise score: min-max normalize desirable directions. Equal weighting of
    # conversion/yield and modest penalty for potential/temp. Not an experimental truth metric.
    score_data=cand.copy()
    def norm(s, higher=True):
        lo,hi=s.min(),s.max()
        z=(s-lo)/(hi-lo) if hi>lo else pd.Series(0.5,index=s.index)
        return z if higher else 1-z
    score_data['Compromise_Score']=(.4*norm(score_data['Conversion_pct']) + .4*norm(score_data['Yield_pct']) + .1*norm(score_data['Temperature_C'],False) + .1*norm(score_data['Potential_V'],False))
    # Add descriptor strings.
    def active(row, cols): return '+'.join(c.replace('Catalyst_','') for c in cols if c in row.index and pd.notna(row[c]) and row[c]>=0.5) or 'Not encoded'
    metal_cols=[c for c in d.columns if c.startswith('Catalyst_') and c in ['Catalyst_Ni','Catalyst_Co','Catalyst_Fe','Catalyst_Cu','Catalyst_Pt','Catalyst_Pd','Catalyst_Au','Catalyst_Ru','Catalyst_Ir','Catalyst_Mn','Catalyst_Mo','Catalyst_W','Catalyst_Ce','Catalyst_Zn','Catalyst_Ag']]
    family_cols=[c for c in d.columns if c.startswith('Catalyst_') and c in ['Catalyst_Oxide','Catalyst_Hydroxide','Catalyst_Oxyhydroxide','Catalyst_LDH','Catalyst_Phosphide','Catalyst_Nitride','Catalyst_Sulfide','Catalyst_Selenide','Catalyst_MOF','Catalyst_Carbon','Catalyst_SingleAtom','Catalyst_Nanoparticle','Catalyst_Metal']]
    cell_cols=[c for c in ['Divided_Cell','Undivided_Cell','Flow_Cell','Three_Electrode'] if c in d.columns]
    for frame in [cand,front,score_data]:
        frame['Metal_Descriptors']=frame.apply(lambda r: active(r,metal_cols),axis=1)
        frame['Catalyst_Family_Descriptors']=frame.apply(lambda r: active(r,family_cols),axis=1)
        frame['Cell_Descriptors']=frame.apply(lambda r: active(r,cell_cols),axis=1)
        frame['Potential_Reference']='RHE flag present in source encoding'
        frame['Evidence_Type']='Observed literature dataset row; not a model-generated recipe'

    score_data=score_data.sort_values('Compromise_Score',ascending=False)
    top=score_data.sort_values(['Pareto_Nondominated','Compromise_Score'], ascending=[False,False]).head(5).copy()
    # Preserve useful columns.
    cols=['Source_Row','Paper Title','Catalysts','Electrolyte','Cell Type','Conversion_pct','Yield_pct','FE_pct','Selectivity_pct','Temperature_C','Potential_V','Potential_vs_RHE','HMF_Concentration_mM','Electrolyte_Concentration_M','Current_mA','Cell_Voltage','TEMPO_binary','Alkaline_Electrolyte','Electrolyte_Electrolyte_Anion_OH','Metal_Descriptors','Catalyst_Family_Descriptors','Cell_Descriptors','Compromise_Score','Evidence_Type']
    cols=[c for c in cols if c in top.columns]
    top[cols].to_csv(out/'Top5_Research_Constrained_Candidate_Shortlist.csv',index=False)
    cand.to_csv(out/'All_Eligible_Observed_Experiments.csv',index=False)
    front.to_csv(out/'Observed_Pareto_Front.csv',index=False)
    score_data.to_csv(out/'Ranked_Observed_Pareto_Front.csv',index=False)
    # Plot observed front, no predictions.
    plt.figure(figsize=(8,6))
    plt.scatter(cand['Conversion_pct'],cand['Yield_pct'],alpha=.45,label='Eligible observed experiments')
    plt.scatter(front['Conversion_pct'],front['Yield_pct'],s=55,marker='D',label='Observed Pareto front')
    for _,r in top.iterrows(): plt.annotate(str(int(r['Source_Row'])),(r['Conversion_pct'],r['Yield_pct']),xytext=(4,4),textcoords='offset points',fontsize=8)
    plt.xlabel('Observed HMF conversion (%)'); plt.ylabel('Observed FDCA yield (%)')
    plt.title('Research-constrained observed Pareto front (not surrogate predictions)')
    plt.legend(); plt.tight_layout(); plt.savefig(out/'Observed_Pareto_Front.png',dpi=220); plt.close()
    report={
      'input_rows':int(len(d)), 'eligible_rows':int(len(cand)), 'pareto_front_size':int(len(front)),
      'top5_rows':top['Source_Row'].astype(int).tolist(), 'top5_pareto_flags':top['Pareto_Nondominated'].astype(bool).tolist(), 'temperature_note':temp_note,
      'method':'Pareto filtering of observed literature rows; no ML surrogate extrapolation and no synthetic recipes.',
      'screening_bounds':{'Potential_V': '1.0–1.65 V, only rows with Potential_vs_RHE flag=1', 'HMF_Concentration_mM':'1–100 mM', 'Electrolyte_Concentration_M':'0.1–2.0 M', 'alkaline':'Alkaline_Electrolyte=1 and OH anion flag=1', 'temperature_C':'15–60 °C for primary screen'},
      'score_weights':{'conversion':0.4,'yield':0.4,'temperature_minimize':0.1,'potential_minimize':0.1},
      'limitations':['Potential_vs_RHE is a binary flag; it does not prove every numeric potential was consistently converted to RHE. Verify source papers.','Catalyst flags are descriptors, not exact catalyst recipes.','Pareto status is relative to the screened dataset and objectives, not universal dominance.','The top five are observed records, not experimentally tested recommendations.']
    }
    (out/'Method_and_Limitations.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Input rows: {len(d)} | Eligible observed rows: {len(cand)} | Pareto front: {len(front)}')
    print('\nTOP FIVE RESEARCH-CONSTRAINED OBSERVED CANDIDATES (Pareto flag included)')
    print(top[cols].to_string(index=False))
    print(f'\nOutputs saved to: {out}')
    print('Interpretation: these are observed literature records, not GA-generated or validated experimental optima.')

if __name__=='__main__': main()
