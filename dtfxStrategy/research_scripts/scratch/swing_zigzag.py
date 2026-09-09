def detect_confirmed_swings(high, low):
    """Zigzag swing confirmation, no fixed lookback window and no retracement threshold:
    - A candidate high is the running max of `high` since the last confirmed low.
    - The instant a bar fails to extend it, the candidate is confirmed as a real swing high
      right there -- no waiting, no re-test needed -- and we flip to tracking a candidate
      low (running min) from that same bar.
    - Symmetric on the way down: the candidate low confirms the instant a bar fails to
      extend it, flipping back to tracking a new candidate high.
    Confirmation timing is entirely driven by real price behavior (pullback then reversal),
    never a fixed bar count.
    Returns a chronological list of (confirm_idx, pivot_idx, 'H'/'L', price).
    """
    n = len(high)
    pivots = []
    direction = "up"
    cand_price, cand_idx = high[0], 0
    for i in range(1, n):
        if direction == "up":
            if high[i] > cand_price:
                cand_price, cand_idx = high[i], i
            else:
                pivots.append((i, cand_idx, "H", cand_price))
                direction = "down"
                cand_price, cand_idx = low[i], i
        else:
            if low[i] < cand_price:
                cand_price, cand_idx = low[i], i
            else:
                pivots.append((i, cand_idx, "L", cand_price))
                direction = "up"
                cand_price, cand_idx = high[i], i
    return pivots
