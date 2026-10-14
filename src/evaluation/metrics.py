import numpy as np
def directional_accuracy(pred, y_ret_or_dir, is_logit=True):
    p = (np.asarray(pred) > 0) if is_logit else (np.asarray(pred) > 0.5); t = np.asarray(y_ret_or_dir) > (0.5 if is_logit else 0); return float((p == t).mean())
def information_coefficient(pred, fwd_ret):
    a, b = np.asarray(pred), np.asarray(fwd_ret)
    if len(a) < 3 or a.std() == 0 or b.std() == 0: return float("nan")
    from scipy.stats import spearmanr; return float(spearmanr(a, b)[0])
def sign_strategy_returns(pred, fwd_ret, cost=0.0, threshold=0.0):
    """Position = sign(pred) when |pred|>threshold else flat; per-bar return = pos * fwd_ret - cost*|change in pos|. Overlapping horizons are NOT de-overlapped here."""
    pos = np.where(np.abs(pred) > threshold, np.sign(pred), 0.0); turn = np.abs(np.diff(pos, prepend=0.0)); return pos * np.asarray(fwd_ret) - cost * turn
def sharpe(r, periods_per_year=252 * 24 * 4):
    r = np.asarray(r); return float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else float("nan")
def max_drawdown(r):
    eq = np.cumsum(r); return float((eq - np.maximum.accumulate(eq)).min())
