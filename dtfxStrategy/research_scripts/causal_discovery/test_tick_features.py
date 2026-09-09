import numpy as np
import causal_native as native
a=np.array([[1,101,99,1,3],[20,102,100,1,3],[61,200,198,3,1]],dtype=float)
x=native.tick_features(a,1.)
assert list(x[:,0])==[60.,120.] and x[0,1]==2
assert x[0,2]==1 and x[0,3]==.5 and x[0,4]==.5
assert x[0,5]==2 and x[0,7]==1 and x[0,8]==1
b=a.copy();b[2,1:]=[9999,9998,100,200]
assert np.array_equal(native.tick_features(b,1.)[0],x[0])
assert np.array_equal(native.tick_features(a[:2],1.),x[:1])
print('Tick quote imbalance, repricing and completed-minute invariance passed')
