"""Open the held-back period only after candidate selection is saved."""
import hashlib
import json
import numpy as np
import pandas as pd
import causal_native as native
from prepare import OUT
from search import compact,ts

def main():
    frozen=OUT/'frozen_candidate.json'
    candidate=json.loads(frozen.read_text())
    specs=[('frozen',candidate['base'],candidate['risk'],candidate['rules']),
           ('Hyper baseline','hyper',.05,[]),('Velocity baseline','velocity',.06,[])]
    begin,end=ts('2018-03-16'),ts('2026-08-19')
    rows=[];annual=[]
    for name,base,risk,rules in specs:
        a=np.load(OUT/(base+'_opportunities.npy'))
        tests=np.asarray(rules,dtype=float).reshape(-1,3)
        d=native.account(a,tests,risk,begin,end)
        row=dict(name=name,base=base,risk=risk,**compact(d));rows.append(row)
        np.savez(OUT/(name.replace(' ','_')+'_evaluated_trace.npz'),time=d['trace_time'],equity=d['trace_equity'])
        for year in range(2018,2027):
            b=max(begin,ts(f'{year}-01-01'));e=min(end,ts(f'{year+1}-01-01'))
            v=compact(native.account(a,tests,risk,b,e))
            annual.append(dict(name=name,year=year,**v))
        print(row,flush=True)
    (OUT/'locked_evaluation.json').write_text(json.dumps({'frozen_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),
        'results':rows,'note':'closed-equity historical trace; historical first passage is NOT Monte Carlo median'},indent=2))
    pd.DataFrame(annual).to_csv(OUT/'annual_evaluation.csv',index=False)

if __name__=='__main__':main()
