import numpy as np
import causal_native as native

a=np.array([[0,10,0,1,1,.1,1],[5,7,0,1,1,.1,-1],
            [7,11,0,1,1,.1,4],[8,20,0,1,1,.1,2],
            [12,21,0,1,1,.1,3],[13,22,1,1,1,.1,8]],dtype=float)
f=native.resolved_features(a,16)
assert f[2,7]==0 # Same-timestamp close excluded.
assert f[3,7]==-1 # Earlier entry is still open, later entry has resolved.
assert f[5,10]==0 # Instrument state is separate.
b=a.copy();b[b[:,1]>=12,6]=99999
g=native.resolved_features(b,16)
assert np.array_equal(f[:5,7:],g[:5,7:])
assert np.array_equal(f[:4,7:],native.resolved_features(a[:4],16)[:,7:])
print('Resolved-feature timestamp, instrument isolation and future-mutation tests passed')
