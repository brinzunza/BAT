"""Pre-2018 shadow-outcome regime test; every long walk runs in C++."""
import json
import numpy as np
import causal_native as native
from prepare import DATA
from search import ts,compact,score

def main():
    root=DATA/'causal_discovery';out=root/'resolved';out.mkdir(exist_ok=True)
    results=[]
    for folder in [root/'structural',*[root/'structural'/f'{tf}min' for tf in (5,15,60)]]:
        for path in sorted(folder.glob('*_opportunities.npy')):
            raw=np.load(path,mmap_mode='r');a=np.ascontiguousarray(raw[raw[:,0]<ts('2018-03-09')])
            if not len(a):continue
            a=a[np.argsort(a[:,0],kind='stable')];cols=a.shape[1];best=None
            for memory in (16,64,256):
                x=native.resolved_features(a,float(memory))
                for penalty in (0.,.5,1.):
                    # Dispersion penalty is a robustness heuristic, not an IID confidence bound.
                    evidence=x[:,cols]-penalty*x[:,cols+1]/np.sqrt(np.maximum(1.,x[:,cols+2]))
                    x[:,cols]=evidence
                    for threshold in (0.,.05,.1,.2):
                        rules=np.array([[cols,1,threshold],[cols+3,1,32.]])
                        for risk in (.005,.01,.02,.04):
                            d=compact(native.account(x,rules,risk,ts('2013-01-08'),ts('2016-01-01')))
                            r={'source':str(path.relative_to(root)),'memory':memory,'penalty':penalty,
                               'threshold':threshold,'risk':risk,'train':d,'score':score(d)}
                            if best is None or r['score']>best['score']:best=r
                    # Restore original mean before evaluating the next penalty.
                    x[:,cols]=evidence+penalty*x[:,cols+1]/np.sqrt(np.maximum(1.,x[:,cols+2]))
            x=native.resolved_features(a,float(best['memory']))
            x[:,cols]-=best['penalty']*x[:,cols+1]/np.sqrt(np.maximum(1.,x[:,cols+2]))
            rules=np.array([[cols,1,best['threshold']],[cols+3,1,32.]])
            best['validation']=compact(native.account(x,rules,best['risk'],ts('2016-01-08'),ts('2018-03-09')))
            best['eligible']=best['train']['pf']>1 and best['train']['n']>=200 and best['validation']['pf']>1 and best['validation']['n']>=200
            results.append(best)
            print(json.dumps(best),flush=True)
            (out/'results.json').write_text(json.dumps(results,indent=2))
    print('Eligible:',sum(x['eligible'] for x in results),flush=True)

if __name__=='__main__':main()
