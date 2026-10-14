import numpy as np
import causal_native as native
from test_nav import markets

def main():
    rng=np.random.default_rng(81)
    a=np.zeros((3000,10));a[:,7:]=rng.uniform(size=(3000,3))
    a[:,6]=np.where((a[:,7]>.5)&(a[:,8]>.5),1.,-1.)
    train=a[:2000].copy();query=a[2000:].copy()
    _,_,_,p=native.learn_trees(train,query,60)
    assert np.mean(np.sign(p)==np.sign(query[:,6]))>.94
    query[:,0:7]=999
    _,_,_,p2=native.learn_trees(train,query,60)
    assert np.array_equal(p,p2), 'Query outcomes/timestamps must not be predictors'
    m=markets();m[0][1,2]=1.106;m[0][1,3]=1.099;m[0][1,4]=1.105
    samples=np.array([[60.,240.,0.,1.,1.1,.01,1000.]])
    trade=native.forecast_execute(samples,np.array([1.]),m,3,.5)
    assert np.isclose(trade[0,4],1.10008) and np.isclose(trade[0,6],.5) and trade[0,1]==120
    assert len(native.forecast_execute(samples,np.array([0.]),m,3,.5))==0
    short=native.forecast_execute(samples,np.array([-1.]),m,3,.5)
    assert short[0,3]==0 and np.isclose(short[0,4],1.09992)
    changed=[x.copy() for x in m]
    for x in changed:x[x[:,0]>=240,1:]*=10
    assert np.array_equal(short,native.forecast_execute(samples,np.array([-1.]),changed,3,.5))
    print('Interaction learning, query-label isolation, bidirectional bid/ask and horizon-causality tests passed',flush=True)

if __name__=='__main__':main()
