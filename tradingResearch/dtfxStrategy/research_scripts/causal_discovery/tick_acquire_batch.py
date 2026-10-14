"""Connection-reusing curl batches; validated atomic cache and native tick loops."""
import argparse,datetime,hashlib,json,lzma,subprocess,time
import numpy as np
import causal_native as native
from prepare import DATA
from tick_source_check import DT

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);p.add_argument('--reverse',action='store_true');args=p.parse_args()
    root=DATA/'causal_discovery'/'tick_context';stage=root/'transfer_stage';stage.mkdir(exist_ok=True)
    rawdir=root/'raw';features=root/'minutes';records=root/'records'
    for path in (rawdir,features,records):path.mkdir(exist_ok=True)
    def key(r):return r['symbol']+'_'+str(r['hour_start_s'])
    schedule=json.loads((root/'acquisition_manifest.json').read_text())['schedule']
    schedule.sort(key=lambda r:hashlib.sha256(key(r).encode()).hexdigest(),reverse=args.reverse)
    pending=[r for r in schedule if not (records/(key(r)+'.json')).exists()]
    if args.limit:pending=pending[:args.limit]
    results=[];began=time.monotonic()
    for start in range(0,len(pending),64):
        batch=pending[start:start+64];config=[]
        for r in batch:
            day=datetime.datetime.fromtimestamp(r['hour_start_s'],datetime.timezone.utc);s=r['symbol']
            r['url']=f'https://datafeed.dukascopy.com/datafeed/{s}/{day.year}/{day.month-1:02d}/{day.day:02d}/{day.hour:02d}h_ticks.bi5'
            dest=stage/(key(r)+'.part')
            if dest.exists():dest.unlink()
            config+=['url = '+json.dumps(r['url']),'output = '+json.dumps(str(dest)),
                     'max-time = 25','fail','location','silent','show-error','next']
        cfg=stage/'requests.conf';cfg.write_text('\n'.join(config[:-1])+'\n')
        proc=subprocess.run(['curl','--parallel','--parallel-max','4','--config',str(cfg)],capture_output=True)
        for r in batch:
            k=key(r);result=dict(r);path=stage/(k+'.part')
            try:
                blob=path.read_bytes();decoded=lzma.decompress(blob)
                if not decoded or len(decoded)%20:raise ValueError('Empty or malformed records')
                a=np.frombuffer(decoded,DT)
                if not (np.all(a['ask']>=a['bid']) and np.all(np.diff(a['ms'].astype('i8'))>=0) and np.all(a['ms']<3600000)):
                    raise ValueError('Invalid quote prices/timestamps')
                scale=1000. if r['symbol']=='USDJPY' else 100000.;pip=.01 if r['symbol']=='USDJPY' else .0001
                ticks=np.column_stack([r['hour_start_s']+a['ms'].astype(float)/1000,a['ask']/scale,a['bid']/scale,a['av'],a['bv']])
                if not (np.isfinite(ticks).all() and np.all(ticks[:,3:]>=0)):raise ValueError('Invalid sizes')
                f=native.tick_features(ticks,pip)
                path.replace(rawdir/(k+'.bi5'));np.save(features/(k+'.npy'),f)
                result.update(status='complete',ticks=len(a),minutes=len(f),sha256=hashlib.sha256(blob).hexdigest(),
                    feature_sha256=hashlib.sha256((features/(k+'.npy')).read_bytes()).hexdigest())
                tmp=records/(k+'.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(records/(k+'.json'))
            except Exception as e:result.update(status='failed',error=str(e))
            results.append(result)
        summary={'scheduled':len(schedule),'attempted':len(results),'successes':sum(r['status']=='complete' for r in results),
                 'complete_total':len(list(records.glob('*.json'))),'elapsed_seconds':time.monotonic()-began,
                 'failures':[r for r in results if r['status']=='failed'],'last_curl_stderr':proc.stderr.decode(errors='replace')[-1000:]}
        (root/'acquisition_batch_status.json').write_text(json.dumps(summary,indent=2))
        print({k:v for k,v in summary.items() if k not in ('failures','last_curl_stderr')},flush=True)

if __name__=='__main__':main()
