import unittest
import numpy as np
import pandas as pd
from beid_fine_execution import replay
from beid_data_quality import flags
from beid_clean_htf_study import hour_direction
from beid_bidask_check import quote_fill

class FineExecutionTests(unittest.TestCase):
    def test_target_precedes_later_stop_inside_coarse_bar(self):
        times=np.arange(3,dtype=np.int64)*60_000_000_000
        p=np.array([[101,101,99,100],[100,103,99,102],[102,103,97,98]],dtype=float)
        r=replay(times,p,np.array([0]),np.array([100.]),np.array([98.]),np.array([102.]),np.array([1]))
        self.assertEqual(r[0,1],1)
        self.assertEqual(r[0,2],102)
        self.assertEqual(r[0,3],2)

    def test_sell_gap_stop(self):
        p=np.array([[99,101,99,100],[104,105,103,104]],dtype=float)
        r=replay(np.arange(2,dtype=np.int64)*60_000_000_000,p,np.array([0]),np.array([100.]),np.array([102.]),np.array([98.]),np.array([-1]))
        self.assertEqual(r[0,2],104)
        self.assertEqual(r[0,3],1)

    def test_quality_flags_internally_consistent_bad_bars(self):
        d=pd.DataFrame({k:[1.]*60+[1965.,-.0001] for k in ['open','high','low','close']})
        f=flags(d)
        self.assertTrue(f.iloc[-2].extreme_vs_prior_median)
        self.assertTrue(f.iloc[-1].nonpositive)
        self.assertFalse(f.iloc[-2].inconsistent)

    def test_mid_touch_without_ask_touch_does_not_fill_buy(self):
        q=pd.DataFrame([dict(time=pd.Timestamp('2024-01-01'),expiry=pd.Timestamp('2024-01-02'),
            ask_l=100.1,ask_h=101.,ask_o=100.5,ask_c=100.5,
            bid_l=99.9,bid_h=100.8,bid_o=100.3,bid_c=100.3)])
        self.assertIsNone(quote_fill(q,100.,98.,102.,1))

    def test_buy_target_uses_bid_not_ask(self):
        q=pd.DataFrame([dict(time=pd.Timestamp('2024-01-01'),expiry=pd.Timestamp('2024-01-02'),
            ask_l=99.9,ask_h=102.2,ask_o=100.1,ask_c=102.1,
            bid_l=99.7,bid_h=101.9,bid_o=99.9,bid_c=101.8)])
        result=quote_fill(q,100.,98.,102.,1)
        self.assertEqual(result[3],'END_MARK')
        self.assertEqual(result[2],101.8)

    def test_hour_feature_ignores_incomplete_hour(self):
        times=pd.date_range('2024-01-01',periods=1500,freq='min')
        d=pd.DataFrame(dict(time=times,close=np.arange(1500,dtype=float)+100))
        at=pd.to_datetime(['2024-01-02 00:15'])
        before=hour_direction(d,at)
        d.loc[d.time>='2024-01-02 00:00','close']=-10000
        after=hour_direction(d,at)
        np.testing.assert_array_equal(before,after)
        self.assertEqual(before[0],1)

if __name__=='__main__':unittest.main()
