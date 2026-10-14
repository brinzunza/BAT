import numpy as np
import causal_native as native

def run(exit_r):
    m=np.array([[0,1,1,1,1],[60,1001,1001,1001,1001],[120,1,1,1,1]],dtype=float)
    a=np.array([[0,180,0,1,1.00008,.001,exit_r]],dtype=float)
    return native.nav_account(a,np.empty((0,3)),[m]*4,.04,0.,180.)

# Deliberately extreme synthetic marks isolate reporting from signal logic:
# floating NAV reaches the target at an open, then retreats before a flat close.
r=run(0.)
assert np.isclose(r['observed_nav_days_to_1m'],60/86400)
assert r['observed_closed_days_to_1m']==-1 and r['equity']==200
# A supplied large realized payoff settles only at its specified exit time.
r=run(300000.)
assert np.isclose(r['observed_closed_days_to_1m'],180/86400)
print('First passage distinguishes transient NAV and settled balance; no retrospective inference')
