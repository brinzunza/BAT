"""I/O orchestration; all indicator, zone, and trade loops execute in C++."""
from pathlib import Path
import hashlib
import json
import time
import os
import numpy as np
import pandas as pd
import causal_native as native

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[3] / 'trading-data'
OUT = DATA / 'causal_discovery'
if os.environ.get('CAUSAL_EXECUTION')=='1': OUT = OUT / 'execution'
if os.environ.get('CAUSAL_STRUCTURAL')=='1': OUT = OUT / 'structural'
TIMEFRAME=int(os.environ.get('CAUSAL_TIMEFRAME','1'))
if TIMEFRAME!=1:OUT=OUT/f'{TIMEFRAME}min'
OUT.mkdir(exist_ok=True)
SYMBOLS = ['EURUSD', 'GBPUSD', 'EURGBP', 'USDJPY']
FILES = ['lse_eurusd_1min_full.csv', 'lse_gbpusd_1min_full.csv',
         'lse_eurgbp_1min_full.csv', 'lse_usdjpy_1min_2018on.csv']
BASES = {'hyper': (4., .15, .10, 16), 'velocity': (5., .20, .20, 4)}
if os.environ.get('CAUSAL_STRUCTURAL')=='1':
    BASES = {'reclaim_1r':(0,1.,0.,1),'reclaim_half':(0,.5,0.,1),
             'failure_1r':(1,1.,0.,1),'failure_2r':(1,2.,0.,1),
             'expansion_1r':(2,1.,0.,1),'expansion_2r':(2,2.,0.,1)}
FEATURES = ['pressure', 'efficiency', 'volatility_ratio', 'displacement60', 'rejection',
            'semivariance_balance', 'trend_acceleration', 'relative_atr', 'sign_switching',
            'range_shock', 'displacement240', 'absorption', 'momentum60', 'autocorrelation',
            'touch_number', 'risk_atr', 'cost_r', 'cross_agreement15', 'cross_agreement60',
            'cross_divergence15', 'cross_stress60']

def main():
    started = time.time()
    buckets = {b: [] for b in BASES}
    quotes = {}
    manifest = {'schema': ['entry_s','exit_s','instrument','buy','entry','risk','net_r']+FEATURES,
                'bases': BASES, 'data': {}, 'feature_lag': 1, 'zone_start': 'bar after confirmation',
                'volume_available': False, 'protocol_sha256': hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest()}
    for inst, (symbol, filename) in enumerate(zip(SYMBOLS, FILES)):
        print('Loading',symbol,flush=True)
        path=DATA/filename
        frame=pd.read_csv(path,parse_dates=['date'])
        duplicates=int(frame.date.duplicated().sum())
        frame=frame.sort_values('date').drop_duplicates('date').reset_index(drop=True)
        t=frame.date.to_numpy(dtype='datetime64[ns]').astype(np.int64)
        o,h,l,c=[np.ascontiguousarray(frame[k],dtype=np.float64) for k in ['open','high','low','close']]
        assert np.isfinite(np.column_stack([o,h,l,c])).all()
        assert np.all(h>=np.maximum(o,c)) and np.all(l<=np.minimum(o,c)) and np.all(l>0)
        assert np.all(np.diff(t)>0)
        manifest['data'][symbol]={'file':filename,'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,
                                 'rows':len(frame),'duplicates_removed':duplicates,
                                 'start':str(frame.date.iloc[0]),'end':str(frame.date.iloc[-1])}
        quotes[symbol]=(t,c)
        if TIMEFRAME!=1:
            t,o,h,l,c=native.aggregate(t,o,h,l,c,TIMEFRAME)
        f=native.features(o,h,l,c)
        z=native.zones(o,h,l,c,t,2)
        # Prefix invariance: future suffix must not alter any already-confirmed zone.
        cut=min(50000,len(t)//2)
        short=native.zones(o[:cut],h[:cut],l[:cut],c[:cut],t[:cut],2)
        eligible=z[5]<cut
        for i in range(6):
            # Last prefix-confirmed zone can have start_idx==cut (not yet tradable).
            assert np.array_equal(z[i][eligible],short[i][short[5]<cut]), ('zone causality',symbol,i)
        assert np.array_equal(f[:cut],native.features(o[:cut],h[:cut],l[:cut],c[:cut])), ('feature causality',symbol)
        for name,(n,rr,age,touches) in BASES.items():
            structural=os.environ.get('CAUSAL_STRUCTURAL')=='1'
            keep=np.ones(len(z[0]),dtype=bool) if structural else (n*(z[0]-z[1])/2/z[2]>=.000170)
            top,bottom,mid,buy,created,start=[np.ascontiguousarray(a[keep]) for a in z]
            frames=[]
            for side in (0,1):
                k=buy==side
                actual_side=1-side if structural and n==1 else side
                if structural:
                    en,ex,entry,risk,net,touch=native.structural_trades(top[k],bottom[k],mid[k],buy[k],start[k],created[k],t,o,l,h,c,
                        np.ascontiguousarray(f[:,7]*c),int(n),rr,1.6*(.01 if symbol=='USDJPY' else .0001),max(1,240//TIMEFRAME))
                    win=net>0
                elif os.environ.get('CAUSAL_EXECUTION')=='1':
                    en,ex,entry,risk,net,touch=native.real_trades(top[k],bottom[k],mid[k],buy[k],start[k],created[k],t,o,l,h,c,age,n,rr,touches,1.6*(.01 if symbol=='USDJPY' else .0001))
                    win=net>0
                else:
                    en,ex,entry,risk,win,touch=native.trades(top[k],bottom[k],mid[k],buy[k],start[k],created[k],t,l,h,age,n,rr,touches)
                    net=np.where(win,rr,-1-1.6*(.01 if symbol=='USDJPY' else .0001)/risk)
                good=en>=240
                en,ex,entry,risk,win,touch,net=[a[good] for a in [en,ex,entry,risk,win,touch,net]]
                feat=f[en-1].astype(np.float64)
                sign=2*actual_side-1
                feat[:,[0,3,4,5,6,10,11,12]]*=sign
                cost=1.6*(.01 if symbol=='USDJPY' else .0001)/risk
                # Intrabar exit ordering is unknown: release capacity only at candle close.
                exit_seconds=t[ex]/1e9+(60*TIMEFRAME if os.environ.get('CAUSAL_EXECUTION')=='1' or structural else 0)
                rows=np.column_stack([t[en]/1e9,exit_seconds,np.full(len(en),inst),np.full(len(en),actual_side),entry,risk,
                                      net,feat,touch,risk/(f[en-1,7]*c[en-1]),cost])
                frames.append(rows)
            combined=np.concatenate(frames)
            buckets[name].append(combined)
            print(symbol,name,len(combined),'causal opportunities',flush=True)
        del frame,f,o,h,l,z
    for name,pieces in buckets.items():
        a=np.concatenate(pieces)
        a=a[np.lexsort((a[:,3],a[:,2],a[:,0]))]
        qt=np.rint(a[:,0]*1e9).astype(np.int64)
        foreign=[native.asof(qt,*quotes[s]) for s in SYMBOLS]
        # EURUSD ~= GBPUSD*EURGBP; GBPUSD ~= EURUSD/EURGBP; EURGBP ~= EURUSD/GBPUSD.
        # USDJPY's USD direction is the opposite of the mean EURUSD/GBPUSD direction.
        own15=np.choose(a[:,2].astype(int),[x[:,1] for x in foreign])
        own60=np.choose(a[:,2].astype(int),[x[:,2] for x in foreign])
        synth15=np.choose(a[:,2].astype(int),[foreign[1][:,1]+foreign[2][:,1],foreign[0][:,1]-foreign[2][:,1],
                                           foreign[0][:,1]-foreign[1][:,1],-(foreign[0][:,1]+foreign[1][:,1])/2])
        synth60=np.choose(a[:,2].astype(int),[foreign[1][:,2]+foreign[2][:,2],foreign[0][:,2]-foreign[2][:,2],
                                           foreign[0][:,2]-foreign[1][:,2],-(foreign[0][:,2]+foreign[1][:,2])/2])
        scale=np.maximum(a[:,7+7],1e-10)
        sign=2*a[:,3]-1
        a=np.column_stack([a,sign*synth15/(scale*np.sqrt(15)),sign*synth60/(scale*np.sqrt(60)),
                           sign*(own15-synth15)/(scale*np.sqrt(15)),np.abs(own60-synth60)/(scale*np.sqrt(60))])
        assert a.shape[1]==len(manifest['schema'])
        np.save(OUT/(name+'_opportunities.npy'),a)
    manifest['elapsed_seconds']=time.time()-started
    manifest['timeframe_minutes']=TIMEFRAME
    manifest['execution_model']=('next-open bid/ask structural signals, 240-observation exit' if os.environ.get('CAUSAL_STRUCTURAL')=='1'
                                 else 'bid/ask conservative entry-candle exits' if os.environ.get('CAUSAL_EXECUTION')=='1'
                                 else 'legacy outcomes with causal zones')
    manifest['prefix_invariance']='passed for zone and feature prefixes on all four symbols'
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Prepared in',round(time.time()-started,1),'seconds',flush=True)

if __name__=='__main__': main()
