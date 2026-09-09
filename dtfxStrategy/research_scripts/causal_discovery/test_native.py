"""Small independent accounting and information-boundary checks, no broker imports."""
import ast
import heapq
import json
from pathlib import Path
import numpy as np
import pandas as pd
import causal_native as native
from prepare import HERE, OUT, SYMBOLS

def extract(path,name,env):
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    return env[name]

def main():
    env=dict(np=np,pd=pd,heapq=heapq,RISK_PCT=.05,GBP_USD_APPROX=1.36,
             START_PER_SIDE=100.,MAX_LEVERAGE=50.,ACCOUNT_CAP=1e100)
    env['position_size_and_margin']=extract(HERE.parent/'goal20_speed_king_margin_report.py','position_size_and_margin',env)
    reference=extract(HERE.parent/'goal30_apex_king_deployment.py','simulate_two_account',env)
    rng=np.random.default_rng(7351)
    n=1000
    en=np.arange(n,dtype=float)*600+1e9
    inst=rng.integers(0,4,n);side=rng.integers(0,2,n);price=np.choose(inst,[1.1,1.3,.85,140.])
    a=np.column_stack([en,en+rng.integers(1,35,n)*60+1,inst,side,price,price*rng.uniform(.0002,.004,n),
                       np.where(rng.random(n)<.85,.2,-1.15)])
    t=pd.DataFrame(dict(entry_time=pd.to_datetime(a[:,0],unit='s'),exit_time=pd.to_datetime(a[:,1],unit='s'),
        instrument=np.array(SYMBOLS)[inst],is_buy=side.astype(bool),entry_price=price,stop_risk=a[:,5],net_r=a[:,6]))
    for risk in [.01,.05,.10]:
        got=native.account(a,np.empty((0,3)),risk,en[0],en[-1]+1e5)
        accepted,buy,sell,dd,trace=reference(t,risk,cap=1e100)
        assert np.array_equal(got['accepted'],np.flatnonzero(accepted.accepted))
        assert np.allclose([got['buy_equity'],got['sell_equity'],got['dd']],[buy,sell,dd],rtol=1e-10,atol=1e-10)
    # Foreign candle stamped at 60 is not available at query 90 (closes at 120).
    times=np.arange(100,dtype=np.int64)*60_000_000_000
    close=np.arange(100,dtype=float)+100
    q=np.array([61*60_000_000_000+30_000_000_000],dtype=np.int64)
    assert native.asof(q,times,close)[0,0]==160
    def execute(o,h,l,c):
        t=np.arange(len(o),dtype=np.int64)*60_000_000_000
        return native.real_trades(np.array([101.]),np.array([99.]),np.array([100.]),np.array([1],dtype=np.uint8),
            np.array([1],dtype=np.int64),np.array([0],dtype=np.int64),t,
            *[np.array(x,dtype=float) for x in [o,l,h,c]],0.,4.,.15,4,1.6)
    # Breach after a possible fill cannot erase the trade.
    r=execute([101,101,99],[102,101,102],[100,98,98],[101,99,101])
    assert r[0].tolist()==[1] and r[1].tolist()==[2] and np.allclose(r[4],[.15])
    # A stop on the entry candle counts, unlike the legacy simulator.
    r=execute([101,101,99],[102,101,102],[100,95,98],[101,99,101])
    assert r[0].tolist()==[1] and r[1].tolist()==[1] and np.allclose(r[4],[-1.])
    # Mid-price touching the limit is insufficient if ask never reaches a buy limit.
    r=execute([101,101],[102,101],[100,99.5],[101,100])
    assert len(r[0])==0
    # No free target when high could precede the entry; unresolved positions are included.
    r=execute([101,101],[102,103],[100,98],[101,99])
    assert len(r[0])==1 and r[4][0]<0
    # A close beyond target proves the target happened after an intrabar fill.
    r=execute([101,101],[102,103],[100,98],[101,102])
    assert len(r[0])==1 and np.allclose(r[4],[.15])
    (OUT/'tests.json').write_text(json.dumps({'account_parity':'1000 synthetic trades at 3 risk levels match original two-account reference',
                                            'asof':'foreign candles unavailable before their close',
                                            'prefix':'zone and feature prefix checks passed in prepare',
                                            'execution':'post-fill invalidation, entry-bar stops, ask fills and unresolved liquidation passed'},indent=2))
    print('Native accounting differential and as-of checks passed',flush=True)

if __name__=='__main__':main()
