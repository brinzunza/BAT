"""Causal spatial zone-field filters. Pre-2018 raw execution; C++ heavy work."""
import json,itertools
import numpy as np
import causal_native as native
from prepare import DATA,SYMBOLS
from search import ts,compact,score

def main():
    root=DATA/'causal_discovery';out=root/'zone_field';out.mkdir(exist_ok=True)
    markets=[]
    for symbol in SYMBOLS:
        m=np.load(root/'market_cache'/(symbol+'.npy'),mmap_mode='r')
        markets.append(m[:np.searchsorted(m[:,0],ts('2018-03-09'))])
    results=[]
    for tf in (5,15,60):
        context=[]
        for m in markets:
            t=np.rint(m[:,0]*1e9).astype(np.int64)
            t,o,h,l,c=native.aggregate(t,*[np.ascontiguousarray(m[:,j]) for j in (1,2,3,4)],tf)
            f=native.features(o,h,l,c);z=native.zones(o,h,l,c,t,2)
            context.append((z[0],z[1],z[3],z[5],t,l,h,c,np.ascontiguousarray(f[:,7]*c)))
        for path in sorted((root/'structural'/f'{tf}min').glob('*_opportunities.npy')):
            a=np.load(path,mmap_mode='r');a=np.ascontiguousarray(a[a[:,0]<ts('2018-03-09')])
            a=a[np.argsort(a[:,0],kind='stable')]
            rr=.5 if 'half' in path.name else (2. if '2r' in path.name else 1.)
            a=native.forecast_execute(a,np.ones(len(a)),markets,(240//tf+1)*tf,rr)
            fields=np.zeros((len(a),10))
            for inst in range(4):
                mask=a[:,2]==inst
                fields[mask]=native.zone_field(*context[inst],np.rint(a[mask,0]*1e9).astype(np.int64),tf)
            # Orient densities and distances by entry direction. Columns 6:8 are unsigned.
            sell=a[:,3]==0
            fields[sell,:6]=fields[sell,:6][:,[1,0,3,2,5,4]]
            fields[:,9]*=2*a[:,3]-1
            cols=a.shape[1];x=np.ascontiguousarray(np.column_stack([a,fields]))
            np.save(out/(str(tf)+'min_'+path.name),x)
            train=(x[:,0]>=ts('2013-01-08'))&(x[:,1]<ts('2016-01-01'))
            rules=[[]]
            for col in range(cols,cols+10):
                for v in np.unique(np.quantile(x[train,col],[.1,.25,.5,.75,.9])):
                    for sign in (-1,1):rules.append([[col,sign,float(v)]])
            evaluated=[]
            def sweep(rs):
                for rule in rs:
                    for risk in (.005,.01,.02,.04):
                        d=compact(native.account(x,np.asarray(rule,dtype=float).reshape(-1,3),risk,ts('2013-01-08'),ts('2016-01-01')))
                        evaluated.append({'rules':rule,'risk':risk,'train':d,'score':score(d)})
            sweep(rules)
            leaders=[];seen=set()
            for row in sorted(evaluated,key=lambda r:r['score'],reverse=True):
                if row['rules'] and row['rules'][0][0] not in seen:
                    leaders.append(row['rules']);seen.add(row['rules'][0][0])
                if len(leaders)==4:break
            sweep([sum(pair,[]) for pair in itertools.combinations(leaders,2)])
            best=max(evaluated,key=lambda r:r['score']);rule=np.asarray(best['rules'],dtype=float).reshape(-1,3)
            best['source']=str(path.relative_to(root));best['timeframe']=tf;best['training_evaluations']=len(evaluated)
            best['validation']=compact(native.account(x,rule,best['risk'],ts('2016-01-08'),ts('2018-03-09')))
            best['eligible']=best['train']['n']>=200 and best['train']['pf']>1 and best['validation']['n']>=200 and best['validation']['pf']>1
            results.append(best)
            (out/'results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(best),flush=True)
    print('Eligible:',sum(r['eligible'] for r in results),flush=True)

if __name__=='__main__':main()
