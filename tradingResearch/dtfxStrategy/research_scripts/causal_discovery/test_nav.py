"""Analytic fixtures for smaller-NAV sizing, FX conversion and boundary causality."""
import json
import numpy as np
import causal_native as native
from prepare import DATA

def markets():
    out=[]
    for price in [1.1,1.3,.85,140.]:
        x=np.column_stack([np.arange(12)*60.,np.full((12,4),price)])
        out.append(x)
    return out

def main():
    m=markets();m[0][1,4]=1.09;m[0][2:4,1:]=1.09
    rows=np.array([[60,240,0,1,1.10008,.01,.2],
                   [120,180,1,0,1.29992,.01,-.5]],dtype=float)
    d=native.nav_account(rows,np.empty((0,3)),m,.1,60.,600.)
    # First trade: 1000 units. At the second entry its liquidation NAV is 89.84:
    # 100 + 1000*(1.09-.00008-1.10008). Short sizes off that smaller NAV -> 898 units.
    assert d['n']==2 and np.isclose(d['close_pnl'][0],-4.49)
    assert np.isclose(d['equity'],197.51)
    assert d['nav_dd']<d['closed_dd']
    assert d['max_effective_risk_fraction']<=.1+1e-10 and d['max_initial_leverage']<=50+1e-10
    # Exit label beyond the boundary is never used. Mark the position to known prices.
    crossing=rows[:1].copy();crossing[0,1]=660
    a=native.nav_account(crossing,np.empty((0,3)),m,.1,60.,180.)
    crossing[0,6]=1000
    b=native.nav_account(crossing,np.empty((0,3)),m,.1,60.,180.)
    assert a['n']==1 and a['forced_boundary_exits']==1
    assert a['equity']==b['equity'] and np.array_equal(a['nav_values'],b['nav_values'])
    # Appending/mutating future prices likewise cannot alter the boundary result.
    future=[x.copy() for x in m]
    for x in future:x[x[:,0]>=180,1:]*=100
    c=native.nav_account(crossing,np.empty((0,3)),future,.1,60.,180.)
    assert a['equity']==c['equity']
    # EURGBP P&L conversion responds to known GBPUSD, rather than a fixed 1.36.
    m=markets();row=np.array([[60,120,2,1,.85008,.01,1.]])
    d=native.nav_account(row,np.empty((0,3)),m,.1,60.,180.)
    units=np.floor(10/(.01*1.3))
    assert np.isclose(d['close_pnl'][0],units*.01*(1.3-.00008))
    out={'nav_sizing':'passed: second account sizes against floating loss on the other account',
         'boundary':'open position included; future exit label and future prices do not alter result',
         'fx':'EURGBP uses contemporaneous GBPUSD conversion',
         'limits':'integer units, max requested risk and initial 50x leverage passed'}
    (DATA/'causal_discovery'/'nav_tests.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out),flush=True)

if __name__=='__main__':main()
