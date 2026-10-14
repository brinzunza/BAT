import numpy as np
import causal_native as native
m=np.ones((16,11));m[:,0]=np.arange(1,17)*60;m[:,1]=10;m[:,3]=.25
a=np.array([[900,999,0,1,1,1,1],[900,999,0,0,1,1,1]],dtype=float)
empty=np.empty((0,11))
x=native.quote_context(a,[m,empty,empty,empty])
assert x[0,2]==.25 and x[1,2]==-.25 and np.all(x[:,18]==1)
b=m.copy();b[-1,1:]=99999
assert np.array_equal(x,native.quote_context(a,[b,empty,empty,empty]))
assert np.isnan(native.quote_context(a,[np.delete(m,3,axis=0),empty,empty,empty])).all()
print('Quote context: direction, exact lag, future exclusion and missing-minute rejection passed')
