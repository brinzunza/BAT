"""Resumable bounded public tick acquisition. I/O in Python; tick loops in C++."""
import argparse,concurrent.futures,datetime,hashlib,json,lzma,subprocess
from pathlib import Path
import numpy as np
import causal_native as native
from prepare import DATA
from tick_source_check import DT

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=128)
    parser.add_argument('--root',type=Path,default=DATA/'causal_discovery'/'tick_context')
    args=parser.parse_args();root=args.root
    rawdir=root/'raw';features=root/'minutes';records=root/'records'
    for p in (rawdir,features,records):p.mkdir(exist_ok=True)
    schedule=json.loads((root/'acquisition_manifest.json').read_text())['schedule']
    # Hash ordering spreads the pilot over dates/instruments, without outcome input.
    def key(r):return r['symbol']+'_'+str(r['hour_start_s'])
    schedule.sort(key=lambda r:hashlib.sha256(key(r).encode()).hexdigest())
    pending=[r for r in schedule if not (records/(key(r)+'.json')).exists()]
    if args.limit>0:pending=pending[:args.limit]
    def fetch(r):
        k=key(r);s=r['symbol'];epoch=r['hour_start_s'];day=datetime.datetime.fromtimestamp(epoch,datetime.timezone.utc)
        url=f'https://datafeed.dukascopy.com/datafeed/{s}/{day.year}/{day.month-1:02d}/{day.day:02d}/{day.hour:02d}h_ticks.bi5'
        result={**r,'url':url}
        try:
            path=rawdir/(k+'.bi5')
            if path.exists():blob=path.read_bytes()
            else:
                p=subprocess.run(['curl','--fail','--silent','--show-error','--location','--max-time','25',url],capture_output=True)
                if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace').strip())
                blob=p.stdout
            decoded=lzma.decompress(blob)
            if not decoded or len(decoded)%20:raise ValueError('Empty or malformed records')
            a=np.frombuffer(decoded,DT)
            if not (np.all(a['ask']>=a['bid']) and np.all(np.diff(a['ms'].astype('i8'))>=0) and np.all(a['ms']<3600000)):
                raise ValueError('Price/time integrity check failed')
            scale=1000. if s=='USDJPY' else 100000.;pip=.01 if s=='USDJPY' else .0001
            ticks=np.column_stack([epoch+a['ms'].astype(float)/1000,a['ask']/scale,a['bid']/scale,a['av'],a['bv']])
            if not (np.isfinite(ticks).all() and np.all(ticks[:,3:]>=0)):raise ValueError('Invalid sizes')
            f=native.tick_features(ticks,pip)
            tmp=path.with_suffix('.tmp');tmp.write_bytes(blob);tmp.replace(path)
            np.save(features/(k+'.npy'),f)
            result.update(status='complete',ticks=len(a),minutes=len(f),sha256=hashlib.sha256(blob).hexdigest(),
                feature_sha256=hashlib.sha256((features/(k+'.npy')).read_bytes()).hexdigest())
            tmp=records/(k+'.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(records/(k+'.json'))
        except Exception as e:result.update(status='failed',error=str(e))
        return result
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(fetch,pending):
            results.append(r)
            if len(results)%16==0:print('Completed batch attempts',len(results),flush=True)
    summary={'scheduled':len(schedule),'attempted':len(results),'complete_total':len(list(records.glob('*.json'))),
             'failures':[r for r in results if r['status']=='failed']}
    (root/'acquisition_status.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
