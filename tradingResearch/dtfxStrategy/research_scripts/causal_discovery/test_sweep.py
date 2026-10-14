import numpy as np
import causal_native as native

def run(side=1,future=100.):
    t=np.arange(5,dtype=np.int64)*60_000_000_000
    o=np.array([100.,100.,98.5,100.,future]);l=np.array([99.5,98.,98.,99.5,future-1])
    h=np.array([100.5,100.,100.5,101.,future+1]);c=np.array([100.,98.5,100.,100.,future])
    if not side:o,l,h,c=200-o,200-h,200-l,200-c
    return native.sweep_samples(np.array([101.]),np.array([99.]),np.array([side],dtype=np.uint8),
        np.array([1],dtype=np.int64),t,o,l,h,c,np.ones(5),.2,1)
for side in (0,1):
    a=run(side);b=run(side,999.)
    assert a.shape==(1,12) and a[0,0]==180 and a[0,3]==side
    assert np.isclose(a[0,5],2.2) and a[0,9]==1
    assert np.array_equal(a,b)
print('Sweep waits for completed recapture; mirrored stops and future invariance passed')
