"""Pre-2018 causal reconstruction of documented non-bounce benchmark policies.

Not an exact reproduction of published performance; no ranking writes or risk fit.
Native C++ performs zone detection, path simulation and account walks.
"""
import json
import os
import numpy as np
import causal_native as native
from prepare import DATA, SYMBOLS
from search import ts, compact

CONFIGS = {
    'Velocity-King_four_touch_source': (5., .20, .20, 4, .06),
    'Velocity-King_sixteen_touch_documentation': (5., .20, .20, 16, .06),
    'Apex-King': (4.5, .18, .30, 8, .05),
    'Hyper-King': (4., .15, .10, 16, .05),
    'Speed-King': (7., .20, .175, 4, .08),
}
CONFIGS = {name: (*cfg, False) for name,cfg in CONFIGS.items()}
SESSIONS={}
if os.environ.get('BENCHMARK_CORRECTED')=='1':
    # Geometry from goal47/goal46, with the RFI filter removed per Git4cfe20a.
    # Dispatch latency is not reproduced; these are explicitly reconstructions.
    CONFIGS={
        'Apex-Ultra_source_geometry_no_RFI':(4.,.15,.08,20,.05,False),
        'Vortex-King_source_geometry_no_RFI':(4.,.15,.15,12,.05,False),
    }
if os.environ.get('BENCHMARK_FIXED')=='1':
    CONFIGS={
        'SOMEGA (live)':(3.343496,1.,3.366682,4,.04,False),
        'goal70_final_winner':(2.,1.,{'EURUSD':3.366682,'GBPUSD':2.5,'EURGBP':2.5,'USDJPY':2.5},4,.03,True),
        'Titan-76':(2.,.76,.20,6,.03,False),
    }
    SESSIONS={
        'SOMEGA (live)':{'EURUSD':None,'GBPUSD':[20,21,22],'EURGBP':list(range(18,24)),'USDJPY':[19,20,21,22]},
        'goal70_final_winner':{'EURUSD':[19,20,21,22],'GBPUSD':[20,21,22],'EURGBP':list(range(18,24)),'USDJPY':[19,20,21,22]},
    }
if os.environ.get('BENCHMARK_BOUNCE')=='1':
    # goal23_margin_capacity_all_strategies.py; fixed recorded risks.
    CONFIGS={
        'Quality-King':(7.,.35,.40,8,.08,True),
        'Hybrid-Ultra':(6.,.4,2.,4,.08,False),
        'DTFX-Ultra':(6.,.4,2.,8,.08,True),
        'SOMEGA-Ultra':(6.,.4,3.366682,4,.08,False),
        'CANA-NarrowTarget':(3.,.8,2.,4,.04,True),
        'SOMEGA-NarrowTarget':(3.343496,.7,3.366682,4,.04,False),
        'DTFX-NarrowTarget':(2.,.7,2.,4,.03,True),
        'CANA (faithful)':(3.,1.,2.,4,.04,True),
    }

def main():
    out = DATA/'causal_discovery'/'benchmark'/'base_policies'
    if os.environ.get('BENCHMARK_BOUNCE')=='1':out=out.parent/'bounce_policies'
    if os.environ.get('BENCHMARK_FIXED')=='1':out=out.parent/'fixed_policies'
    if os.environ.get('BENCHMARK_CORRECTED')=='1':out=out.parent/'corrected_source_policies'
    out.mkdir(parents=True, exist_ok=True)
    pieces = {(name, path): [] for name in CONFIGS for path in (0, 1)}
    end = ts('2018-03-09')
    for inst, symbol in enumerate(SYMBOLS):
        raw = np.load(DATA/'causal_discovery'/'market_cache'/(symbol+'.npy'), mmap_mode='r')
        raw = raw[:np.searchsorted(raw[:, 0], end)]
        t = np.rint(raw[:, 0]*1e9).astype(np.int64)
        o,h,l,c = [np.ascontiguousarray(raw[:, j]) for j in (1,2,3,4)]
        zone_cache={2:native.zones(o,h,l,c,t,2)}
        if 'Titan-76' in CONFIGS:zone_cache[1]=native.zones(o,h,l,c,t,1)
        for name,(n,rr,age,touches,risk_pct,bounce) in CONFIGS.items():
            z=zone_cache[1 if name=='Titan-76' else 2]
            age=age[symbol] if isinstance(age,dict) else age
            keep = n*(z[0]-z[1])/2/z[2] >= .000170
            top,bottom,mid,buy,created,start = [np.ascontiguousarray(x[keep]) for x in z]
            for path in (0,1):
                # Native output has no zone id; retain direction through separate walks.
                rows=[]
                for side in (0,1):
                    m=buy==side
                    en,ex,entry,risk,net,touch=native.path_trades(
                        top[m],bottom[m],mid[m],buy[m],start[m],created[m],t,o,l,h,c,
                        age,n,rr,touches,1.6*(.01 if inst==3 else .0001),path,bounce)
                    rows.append(np.column_stack([t[en]/1e9,t[ex]/1e9+60,
                        np.full(len(en),inst),np.full(len(en),side),entry,risk,net]))
                combined=np.concatenate(rows)
                hours=SESSIONS.get(name,{}).get(symbol)
                if hours is not None:combined=combined[np.isin((combined[:,0]//3600).astype(int)%24,hours)]
                pieces[name,path].append(combined)
        print(symbol+' completed',flush=True)
    results=[]
    for (name,path),blocks in pieces.items():
        a=np.concatenate(blocks)
        a=np.ascontiguousarray(a[np.lexsort((a[:,3],a[:,2],a[:,0]))])
        np.save(out/(name+'_path'+str(path)+'.npy'),a)
        result={'name':name,'path':path,'config':CONFIGS[name],'utc_hours':SESSIONS.get(name),
            'pivot_retracement':1 if name=='Titan-76' else 2,
            'screening':compact(native.account(a,np.empty((0,3)),CONFIGS[name][4],ts('2013-01-08'),end))}
        results.append(result)
        print(json.dumps(result),flush=True)
    (out/'results.json').write_text(json.dumps({'limitations':[
        'Realized-balance screening, not final NAV or live execution verification',
        'Two admissible OHLC paths, not exhaustive tick-path bounds',
        'Idealized immediate order eligibility and cancellation; no dispatch latency',
        'Bounce repeats require a completed probe/excursion candle after the preceding exit candle',
        'Velocity touch count provenance conflict is retained as two variants',
        'No post-2018 bars loaded; fixed recorded risks, no optimization'],
        'results':results},indent=2))

if __name__=='__main__':main()
