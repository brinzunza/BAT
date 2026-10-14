import numpy as np
import causal_native as native
from test_nav import markets

def main():
    m=markets();q=np.array([60.,120.,180.])
    a=native.basis_features(m,q)
    # Known constant mids imply fair EURUSD=GBPUSD*EURGBP=1.105.
    assert np.allclose(a[:,1],50.)
    assert np.allclose(a[:,0],0.) and np.allclose(a[:,4:],0.)
    future=[x.copy() for x in m];future[1][2:,1:]=2.
    b=native.basis_features(future,q)
    assert np.array_equal(a[:2],b[:2]) and not np.array_equal(a[2],b[2])
    assert np.array_equal(a[:2],native.basis_features(m,q[:2]))
    print('Currency graph identity, prefix and completed-candle availability checks passed',flush=True)

if __name__=='__main__':main()
