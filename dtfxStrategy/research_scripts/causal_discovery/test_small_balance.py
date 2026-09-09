import numpy as np
import causal_native as native
m=np.array([[t,1.,1.,1.,1.] for t in (0,60,120,180)],dtype=float)
a=np.array([[0,60,0,1,1.00008,.001,-24.875],
            [60,120,0,1,1.00008,.0001,1.]],dtype=float)
r=native.nav_account(a,np.empty((0,3)),[m]*4,.04,0.,240.)
# First settled loss leaves the buy account at$0.50. A24-unit order fits
# its margin and risk budgets; the former arbitrary$1 gate rejected it.
assert r['n']==2 and np.isclose(r['buy_equity'],.5024)
assert r['max_effective_risk_fraction']<=.04+1e-12 and r['max_initial_leverage']<=50
# A nonpositive smaller account still cannot fund another order.
b=a.copy();b[0,6]=-25.
assert native.nav_account(b,np.empty((0,3)),[m]*4,.04,0.,240.)['n']==1
# Strict screening must reject an order if even one unit exceeds its risk budget.
c=np.array([[0,60,0,1,1.,10.,1.]],dtype=float)
assert native.account(c,np.empty((0,3)),.005,0.,240.,strict_units=True)['n']==0
print('Small positive NAV permits affordable units; nonpositive NAV cannot fund an entry')
