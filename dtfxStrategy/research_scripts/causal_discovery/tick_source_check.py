"""Small fixed cross-instrument/date feasibility sample; no return optimization."""
import concurrent.futures,datetime,hashlib,json,lzma,subprocess
import numpy as np
import causal_native as native
from prepare import DATA

CASES=[('EURUSD','2015-01-05'),('GBPUSD','2015-01-05'),('EURGBP','2015-01-05'),
       ('USDJPY','2015-01-05'),('EURUSD','2013-01-07'),('EURUSD','2016-03-07')]
DT=np.dtype([('ms','>u4'),('ask','>u4'),('bid','>u4'),('av','>f4'),('bv','>f4')])

def main():
    root=DATA/'causal_discovery';out=root/'tick_probe';out.mkdir(exist_ok=True)
    def fetch(case):
        symbol,date=case;day=datetime.date.fromisoformat(date)
        url=f'https://datafeed.dukascopy.com/datafeed/{symbol}/{day.year}/{day.month-1:02d}/{day.day:02d}/12h_ticks.bi5'
        path=out/f'{symbol}_{date.replace("-", "")}_12.bi5'
        result={'symbol':symbol,'date':date,'url':url}
        try:
            if not path.exists():
                blob=subprocess.run(['curl','--fail','--silent','--show-error','--location',
                    '--max-time','25',url],check=True,capture_output=True).stdout
                # Validate a download before caching it as a complete hour.
                raw=lzma.decompress(blob);assert len(raw)%20==0 and len(raw)>0
                path.write_bytes(blob)
            blob=path.read_bytes();raw=lzma.decompress(blob);a=np.frombuffer(raw,DT)
            assert np.all(a['ask']>=a['bid']) and np.all(np.diff(a['ms'].astype('i8'))>=0) and np.all(a['ms']<3600000)
            scale=1000. if symbol=='USDJPY' else 100000.;pip=.01 if symbol=='USDJPY' else .0001
            epoch=datetime.datetime(day.year,day.month,day.day,12,tzinfo=datetime.timezone.utc).timestamp()
            ticks=np.column_stack([epoch+a['ms'].astype(float)/1000,a['ask']/scale,a['bid']/scale,a['av'],a['bv']])
            assert np.isfinite(ticks).all() and np.all(ticks[:,3:]>=0)
            f=native.tick_features(ticks,pip);np.save(path.with_suffix('.npy'),f)
            m=np.load(root/'market_cache'/(symbol+'.npy'),mmap_mode='r')
            ix=np.searchsorted(m[:,0],f[:,0]-60);keep=(ix<len(m))&(m[np.minimum(ix,len(m)-1),0]==f[:,0]-60)
            gap=(f[keep,10]-m[ix[keep],4])/pip
            result.update(sha256=hashlib.sha256(blob).hexdigest(),ticks=len(a),minutes=len(f),matched_minutes=int(keep.sum()),
                available_size_imbalance_mean=float(f[:,3].mean()),price_scale=scale,pip=pip)
            if len(gap):result.update(signed_close_gap_pips_quantiles=np.quantile(gap,[0,.5,1]).tolist(),
                demeaned_close_gap_rmse_pips=float(np.sqrt(np.mean((gap-gap.mean())**2))))
        except Exception as e:result['error']=str(e)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(fetch,CASES))
    (out/'cross_source_checks.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)

if __name__=='__main__':main()
