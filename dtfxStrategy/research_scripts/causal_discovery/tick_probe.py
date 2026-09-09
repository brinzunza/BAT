"""Decode one acquired public hour and check alignment; no strategy selection."""
import json,lzma
import numpy as np
import causal_native as native
from prepare import DATA
from search import ts

def main():
    root=DATA/'causal_discovery';out=root/'tick_probe'
    dt=np.dtype([('ms','>u4'),('ask','>u4'),('bid','>u4'),('ask_size','>f4'),('bid_size','>f4')])
    a=np.frombuffer(lzma.decompress((out/'EURUSD_20150105_12.bi5').read_bytes()),dt)
    ticks=np.column_stack([ts('2015-01-05 12:00')+a['ms'].astype(float)/1000,
        a['ask']/1e5,a['bid']/1e5,a['ask_size'],a['bid_size']])
    assert np.all(np.isfinite(ticks)) and np.all(ticks[:,3:]>=0)
    f=native.tick_features(ticks,.0001);np.save(out/'EURUSD_20150105_12_minutes.npy',f)
    m=np.load(root/'market_cache'/'EURUSD.npy',mmap_mode='r');ix=np.searchsorted(m[:,0],f[:,0]-60)
    keep=(ix<len(m))&(m[np.minimum(ix,len(m)-1),0]==f[:,0]-60)
    gap=(f[keep,10]-m[ix[keep],4])/.0001
    result={'minutes':len(f),'ticks':len(ticks),'matched_lse_minutes':int(keep.sum()),
        'median_abs_mid_close_difference_pips':float(np.median(np.abs(gap))),
        'max_abs_mid_close_difference_pips':float(np.max(np.abs(gap))),
        'mean_available_size_imbalance':float(f[:,3].mean()),
        'mean_ticks_per_minute':float(f[:,1].mean()),
        'columns':['minute_close_s','quote_count','directional_repricing_balance','mean_available_size_imbalance',
                   'mean_microprice_deviation_pips','mean_observed_spread_pips','mid_range_pips','mid_change_pips',
                   'root_sum_squared_mid_changes_pips','mean_available_total_size','last_mid'],
        'caveats':['One public hour is a feasibility test, not strategy evidence',
                   'Available quoted size is not executed trading volume or global FX volume',
                   'Observed spread is an indicator only; required execution cost remains fixed1.6pips',
                   'Binary layout inferred and sanity checked; validate more dates and instruments before bulk use']}
    (out/'alignment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
