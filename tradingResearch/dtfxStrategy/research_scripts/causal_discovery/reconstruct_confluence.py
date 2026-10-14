"""Reconstruct #1 from immutable Git prose. Does not change official rankings."""
import json
import subprocess
import os
import numpy as np
import pandas as pd
import causal_native as native
from prepare import DATA,SYMBOLS
from search import ts,compact

OUT=DATA/'causal_discovery'/'benchmark';OUT.mkdir(exist_ok=True)
PATH_MODE=int(os.environ.get('CAUSAL_PATH','-1'))
if PATH_MODE>=0:OUT=OUT/f'path{PATH_MODE}';OUT.mkdir(exist_ok=True)

def main():
    provenance=subprocess.check_output(['git','-C',str(DATA),'show','304a2f1:STRATEGY_RANKINGS.md'],text=True)
    (OUT/'confluence_source_v21.md').write_text(provenance)
    quotes={};pieces=[]
    for inst,symbol in enumerate(SYMBOLS):
        raw=np.load(DATA/'causal_discovery'/'market_cache'/(symbol+'.npy'),mmap_mode='r')
        t=np.rint(raw[:,0]*1e9).astype(np.int64)
        o,h,l,c=[np.ascontiguousarray(raw[:,j]) for j in [1,2,3,4]]
        quotes[symbol]=(t,c)
        f=native.confluence_features(o,h,l,c)
        cut=50000
        assert np.allclose(f[:cut],native.confluence_features(o[:cut],h[:cut],l[:cut],c[:cut]),atol=1e-12,rtol=0)
        z=native.zones(o,h,l,c,t,2);keep=4*(z[0]-z[1])/2/z[2]>=.000170
        top,bottom,mid,buy,created,start=[np.ascontiguousarray(x[keep]) for x in z]
        for side in [0,1]:
            mask=buy==side
            args=(top[mask],bottom[mask],mid[mask],buy[mask],start[mask],created[mask],t,o,l,h,c,
                  .08,4.,.15,30,1.6*(.01 if inst==3 else .0001))
            en,ex,entry,risk,net,touch=native.path_trades(*args,PATH_MODE) if PATH_MODE>=0 else native.real_trades(*args)
            keep=en>=240;en,ex,entry,risk,net=[x[keep] for x in [en,ex,entry,risk,net]]
            indicators=f[en-1];bb=(indicators[:,0]<=.2)|(indicators[:,0]>=.8)
            counter=(indicators[:,1]==(1 if side==0 else -1))&(indicators[:,2]>=.30)
            pieces.append(np.column_stack([t[en]/1e9,t[ex]/1e9+60,np.full(len(en),inst),np.full(len(en),side),entry,risk,net,bb,counter]))
        print(symbol,'reconstructed',flush=True)
    a=np.concatenate(pieces);a=a[np.lexsort((a[:,3],a[:,2],a[:,0]))]
    qt=np.rint(a[:,0]*1e9).astype(np.int64)
    momentum=np.column_stack([native.asof(qt,*quotes[s])[:,2] for s in SYMBOLS])
    own=momentum[np.arange(len(a)),a[:,2].astype(int)];matches=np.sum(np.sign(momentum)==np.sign(own[:,None]),axis=1)-1
    cross=matches>=2;keep=(a[:,7]>0)|(a[:,8]>0)|cross
    a=np.ascontiguousarray(a[keep,:7]);np.save(OUT/'confluence_causal_execution.npy',a)
    result=compact(native.account(a,np.empty((0,3)),.05,ts('2013-01-08'),ts('2018-03-09')))
    (OUT/'confluence_pre2018.json').write_text(json.dumps({'result':result,'source_commit':'304a2f15a0fa3810ebd9a14f93b293d05d76f37a',
        'known_reconstruction_choices':['ATR14 simple trailing mean','ATR percentile over 20160 observations, inclusive rank',
                                      'cross momentum 60 observations on last completed as-of candle; five-minute staleness limit',
                                      'same-bar ties deterministic instrument/side order'],
        'execution_path':PATH_MODE,
        'comparison':'reconstructed prose + corrected fills; not exact reproduction of lost original implementation'},indent=2))
    print(result,flush=True)

if __name__=='__main__':main()
