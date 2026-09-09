import numpy as np
import causal_native as native

def main():
    a=np.zeros((5,29));a[:,0]=[60,60,60,120,180];a[:,2]=0;a[:,3]=[1,1,1,0,1]
    a[:,28]=[5,15,60,5,5];a[:,9]=1
    full=native.consensus(a)
    assert np.allclose(full[:3,29],1)
    assert np.allclose(full[:3,30],0)
    assert full[3,30]>0 and full[3,29]==1/3
    assert np.array_equal(full[:4],native.consensus(a[:4]))
    altered=a.copy();altered[-1,3]=0
    assert np.array_equal(full[:4],native.consensus(altered)[:4])
    print('Consensus prefix and simultaneous-signal checks passed',flush=True)

if __name__=='__main__':main()
