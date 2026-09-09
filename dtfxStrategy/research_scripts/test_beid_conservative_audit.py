"""Regression tests for the execution biases identified in BEID.md's source engine."""
import unittest
import pandas as pd
from beid_conservative_audit import backtest_beid_conservative

class ExecutionTests(unittest.TestCase):
    def run_bars(self, tail, gap=False):
        rows=[(100,102,99,101),(101,101,98,99),(99,104,99,103)]+tail
        times=list(pd.date_range('2024-01-01',periods=len(rows),freq='3min'))
        if gap:
            times[3:]=list(pd.date_range('2024-01-04',periods=len(rows)-3,freq='3min'))
        d=pd.DataFrame(rows,columns=['open','high','low','close'])
        d['time']=times
        return backtest_beid_conservative(d,'USDJPY',3)

    def test_fill_bar_stop_is_loss_not_discarded(self):
        t=self.run_bars([(103,104,97,103)])
        self.assertEqual(len(t),1)
        self.assertEqual(t.iloc[0].reason,'STOP')
        self.assertTrue(t.iloc[0].fill_bar_stop)
        self.assertAlmostEqual(t.iloc[0].net_r,-1.008)

    def test_later_ambiguous_bar_stop_first(self):
        t=self.run_bars([(101,101,99,100),(100,103,97,103)])
        self.assertEqual(t.iloc[0].reason,'STOP')

    def test_proven_fill_bar_target(self):
        t=self.run_bars([(101,103,99,103)])
        self.assertEqual(t.iloc[0].reason,'TARGET')
        self.assertAlmostEqual(t.iloc[0].net_r,.992)

    def test_stop_gap_worse_than_stop_price(self):
        t=self.run_bars([(101,101,99,100),(97,99,96,98)])
        self.assertAlmostEqual(t.iloc[0].net_r,-1.508)

    def test_expiry_precedes_touch(self):
        t=self.run_bars([(101,103,99,103)],gap=True)
        self.assertTrue(t.empty)

    def test_open_position_marked_at_end(self):
        t=self.run_bars([(101,101,99,100)])
        self.assertEqual(t.iloc[0].reason,'END_MARK')
        self.assertAlmostEqual(t.iloc[0].net_r,-.008)

if __name__=='__main__': unittest.main()
