import numpy as np
import causal_native as native

def execute(path,top=101.,bottom=99.,low=98.):
    return native.path_trades(np.array([top]),np.array([bottom]),np.array([100.]),np.array([1],dtype=np.uint8),
        np.array([1],dtype=np.int64),np.array([0],dtype=np.int64),np.array([0,60_000_000_000],dtype=np.int64),
        np.array([101.,101.]),np.array([100.,low]),np.array([102.,103.]),np.array([101.,100.]),0.,4.,.15,4,1.6,path)

def main():
    # On H-first path, target occurs before fill; on L-first, target follows fill.
    a,b=execute(0),execute(1)
    assert np.allclose(a[4],[-.2]) and np.allclose(b[4],[.15])
    # A structural cancellation reached before the ask-side limit prevents filling.
    for path in [0,1]:
        assert len(execute(path,100.5,99.5)[0])==0
        assert np.allclose(execute(path,low=95.)[4],[-1.])
    def repeat(bounce,side=1):
        o=np.array([101.]*5);l=np.array([100.5,99.8,99.8,99.8,99.8])
        h=np.array([102.,101.5,102.,103.5,101.5]);c=np.array([101.]*5)
        if not side:o,l,h,c=200-o,200-h,200-l,200-c
        return native.path_trades(np.array([101.]),np.array([99.]),np.array([100.]),
            np.array([side],dtype=np.uint8),np.array([1],dtype=np.int64),np.array([0],dtype=np.int64),
            np.arange(5,dtype=np.int64)*60_000_000_000,o,l,h,c,0.,4.,.05,8,.2,
            1 if side else 0,bounce)
    for side in (0,1):
        assert list(repeat(False,side)[0])==[1,2,3,4]
        # First touch is allowed; repeats wait until the probe/excursion candle closes.
        assert list(repeat(True,side)[0])==[1,4]
    print('OHLC ordering, causal cancel-before-fill and entry-bar stop checks passed',flush=True)

if __name__=='__main__':main()
