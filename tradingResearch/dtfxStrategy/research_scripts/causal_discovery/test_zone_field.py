import numpy as np
import causal_native as native

def run(future=105.):
    t=np.arange(5,dtype=np.int64)*60_000_000_000
    return native.zone_field(np.array([102.,future]),np.array([98.,103.]),
        np.array([1,0],dtype=np.uint8),np.array([1,4],dtype=np.int64),t,
        np.array([99.,99.,97.,99.,99.]),np.array([101.,101.,101.,104.,104.]),
        np.full(5,100.),np.ones(5),t,1)
a=run();b=run(999.)
assert a[0,6]==0 and np.isclose(a[1,6],np.log(2))
assert a[2,0]==1 and a[3,6]==0 # Completed wick invalidates the live buy zone.
assert np.array_equal(a[:4],b[:4]) # Future zone geometry cannot affect the prefix.
assert a[4,1]>0 # Zone becomes available only following its confirmation candle.
print('Zone-field confirmation, wick invalidation and future-zone tests passed')
