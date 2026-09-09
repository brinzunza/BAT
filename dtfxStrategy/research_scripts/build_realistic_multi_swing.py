import json
import random
from datetime import datetime, timedelta

def create_realistic_candle(prev_close, open_delta, body_delta, upper_wick, lower_wick):
    o = round(prev_close + open_delta, 2)
    c = round(o + body_delta, 2)
    h = round(max(o, c) + max(0.05, upper_wick), 2)
    l = round(min(o, c) - max(0.05, lower_wick), 2)
    return o, h, l, c

def generate_realistic_dataset():
    candles = []
    current_time = datetime(2026, 9, 9, 9, 30, 0)
    
    def append_c(o, h, l, c):
        nonlocal current_time
        candles.append({
            "t": current_time.strftime("%H:%M:%S"),
            "o": round(float(o), 2),
            "h": round(float(h), 2),
            "l": round(float(l), 2),
            "c": round(float(c), 2)
        })
        current_time += timedelta(seconds=15)

    # Initial anchor candle
    append_c(100.0, 101.2, 99.8, 100.8) # Green

    # =========================================================================
    # PHASE 1: 5-SWING UPTREND (100.0 -> 136.0)
    # =========================================================================
    
    # --- SWING 1 ---
    # Impulse (3 green candles)
    append_c(100.8, 102.6, 100.6, 102.2) # Green
    append_c(102.2, 104.2, 102.0, 103.8) # Green
    append_c(103.8, 106.0, 103.5, 105.5) # Green (pHH = 106.0, green body [103.8, 105.5])
    # Pullback (2 red candles, 1st closes below 103.8)
    append_c(105.4, 105.6, 103.0, 103.2) # Red (close 103.2 < 103.8! cHH1 = 106.0, pHL = 103.0)
    append_c(103.2, 103.6, 102.4, 102.6) # Red (pHL = 102.4)
    # Breakout (3 candles up, closes above 106.0)
    append_c(102.7, 104.5, 102.5, 104.2) # Green
    append_c(104.2, 107.5, 104.0, 107.0) # Green (close 107.0 > 106.0! cHL1 = 102.4. BUY ZONE 1 [106.0 - 102.4])

    # --- SWING 2 ---
    # Impulse
    append_c(107.0, 109.4, 106.8, 109.0) # Green
    append_c(109.0, 111.5, 108.8, 111.0) # Green (pHH = 111.5, green body [109.0, 111.0])
    # Pullback (closes below 109.0)
    append_c(110.8, 111.0, 108.2, 108.5) # Red (close 108.5 < 109.0! cHH2 = 111.5, pHL = 108.2)
    append_c(108.5, 108.8, 107.5, 107.8) # Red (pHL = 107.5)
    # Breakout
    append_c(107.9, 110.2, 107.7, 110.0) # Green
    append_c(110.0, 113.2, 109.8, 112.8) # Green (close 112.8 > 111.5! cHL2 = 107.5. BUY ZONE 2 [111.5 - 107.5])

    # --- SWING 3 ---
    # Impulse
    append_c(112.8, 115.6, 112.5, 115.0) # Green
    append_c(115.0, 117.8, 114.7, 117.2) # Green (pHH = 117.8, green body [115.0, 117.2])
    # Pullback (closes below 115.0)
    append_c(117.0, 117.2, 114.2, 114.4) # Red (close 114.4 < 115.0! cHH3 = 117.8, pHL = 114.2)
    append_c(114.4, 114.8, 113.6, 113.8) # Red (pHL = 113.6)
    # Breakout
    append_c(113.9, 116.5, 113.7, 116.0) # Green
    append_c(116.0, 119.8, 115.8, 119.2) # Green (close 119.2 > 117.8! cHL3 = 113.6. BUY ZONE 3 [117.8 - 113.6])

    # --- SWING 4 ---
    # Impulse
    append_c(119.2, 122.4, 118.9, 122.0) # Green
    append_c(122.0, 124.8, 121.7, 124.2) # Green (pHH = 124.8, green body [122.0, 124.2])
    # Pullback (closes below 122.0)
    append_c(124.0, 124.2, 121.2, 121.5) # Red (close 121.5 < 122.0! cHH4 = 124.8, pHL = 121.2)
    append_c(121.5, 121.8, 120.4, 120.6) # Red (pHL = 120.4)
    # Breakout
    append_c(120.7, 123.5, 120.5, 123.0) # Green
    append_c(123.0, 127.2, 122.8, 126.5) # Green (close 126.5 > 124.8! cHL4 = 120.4. BUY ZONE 4 [124.8 - 120.4])

    # --- SWING 5 (Grand Peak) ---
    # Impulse
    append_c(126.5, 129.8, 126.2, 129.2) # Green
    append_c(129.2, 132.5, 128.9, 132.0) # Green (pHH = 132.5, green body [129.2, 132.0])
    # Pullback (closes below 129.2)
    append_c(131.8, 132.0, 128.5, 128.8) # Red (close 128.8 < 129.2! cHH5 = 132.5, pHL = 128.5)
    append_c(128.8, 129.0, 127.0, 127.2) # Red (pHL = 127.0)
    # Breakout
    append_c(127.3, 130.8, 127.1, 130.2) # Green
    append_c(130.2, 134.5, 130.0, 134.0) # Green (close 134.0 > 132.5! cHL5 = 127.0 (YELLOW FLOOR!). BUY ZONE 5 [132.5 - 127.0])
    # Extended blow-off peak
    append_c(134.0, 136.8, 133.7, 136.2) # Green (pHH = 136.8 - TOP RED PEAK)
    append_c(136.2, 136.5, 134.0, 134.5) # Red pause/distribution candle

    # =========================================================================
    # PHASE 2: 1 TREND FLIP ➔ 3-SWING DOWNTREND (136.8 -> 88.0)
    # =========================================================================

    # Aggressive breakdown through last_cHL5 (127.0)
    append_c(134.5, 134.8, 128.0, 128.5) # Red
    append_c(128.5, 128.8, 123.5, 124.0) # Red (close 124.0 < 127.0! FLIPS TO DOWNTREND! SELL FLIP ZONE [136.8 - 127.0], Mid = 131.90)
    append_c(124.0, 124.5, 120.0, 120.5) # Red (pLL = 120.0)

    # --- SWING 1 of Downtrend ---
    # Pullback up: retests into Flip Zone (touches 132.2 >= 131.90 50% Mid) and closes above 124.0
    append_c(120.6, 127.5, 120.4, 127.0) # Green
    append_c(127.0, 132.4, 126.8, 132.0) # Green (touches 132.4! Close 132.0 > 124.0! cLL1 = 120.0, pLH = 132.4)
    # Breakdown continuation
    append_c(131.8, 132.0, 122.5, 123.0) # Red
    append_c(123.0, 123.5, 116.0, 116.5) # Red (close 116.5 < 120.0! cLH1 = 132.4. SELL ZONE 7 [132.4 - 120.0])

    # --- SWING 2 of Downtrend ---
    append_c(116.5, 116.8, 111.0, 111.5) # Red (pLL = 111.0, red body [111.5, 116.5])
    # Pullback up: green closes above 116.5
    append_c(111.6, 118.2, 111.4, 117.8) # Green (close 117.8 > 116.5! cLL2 = 111.0, pLH = 118.2)
    # Breakdown continuation
    append_c(117.6, 117.8, 108.5, 109.0) # Red
    append_c(109.0, 109.4, 103.5, 104.0) # Red (close 104.0 < 111.0! cLH2 = 118.2. SELL ZONE 8 [118.2 - 111.0])

    # --- SWING 3 of Downtrend (Grand Trough) ---
    append_c(104.0, 104.2, 98.0, 98.5) # Red (pLL = 98.0, red body [98.5, 104.0])
    # Pullback up: green closes above 104.0
    append_c(98.6, 105.6, 98.4, 105.0) # Green (close 105.0 > 104.0! cLL3 = 98.0, pLH = 105.6)
    # Breakdown continuation
    append_c(104.8, 105.0, 95.0, 95.5) # Red
    append_c(95.5, 95.8, 91.0, 91.5) # Red (close 91.5 < 98.0! cLH3 = 105.6 (YELLOW CEILING!). SELL ZONE 9 [105.6 - 98.0])
    # Extended capitulation trough
    append_c(91.5, 91.8, 88.0, 88.5) # Red (pLL = 88.0 - BOTTOM GREEN TROUGH)
    append_c(88.5, 91.0, 88.0, 90.5) # Green base building

    # =========================================================================
    # PHASE 3: 1 TREND FLIP ➔ 6-SWING UPTREND (88.0 -> 160.0)
    # =========================================================================

    # Aggressive breakout surge through last_cLH3 (105.6)
    append_c(90.5, 99.2, 90.2, 98.8) # Green
    append_c(98.8, 108.5, 98.5, 108.0) # Green (close 108.0 > 105.6! FLIPS TO UPTREND! BUY FLIP ZONE [105.6 - 88.0], Mid = 96.80)
    append_c(108.0, 111.0, 107.6, 110.5) # Green (pHH = 111.0)

    # --- SWING 1 of Uptrend 2 ---
    # Pullback down: retests into Buy Flip Zone (touches 96.5 <= 96.80 50% Mid) and closes below 108.0
    append_c(110.2, 110.5, 102.5, 103.0) # Red
    append_c(103.0, 103.2, 96.5, 97.0) # Red (touches 96.5! close 97.0 < 108.0! cHH1 = 111.0, pHL = 96.5)
    # Breakout continuation
    append_c(97.2, 105.5, 97.0, 105.0) # Green
    append_c(105.0, 113.8, 104.8, 113.2) # Green (close 113.2 > 111.0! cHL1 = 96.5. BUY ZONE 11 [111.0 - 96.5])

    # --- SWING 2 of Uptrend 2 ---
    append_c(113.2, 116.8, 113.0, 116.2) # Green
    append_c(116.2, 119.5, 115.9, 119.0) # Green (pHH = 119.5, green body [116.2, 119.0])
    # Pullback (closes below 116.2)
    append_c(118.8, 119.0, 115.0, 115.5) # Red (close 115.5 < 116.2! cHH2 = 119.5, pHL = 115.0)
    append_c(115.5, 115.8, 114.2, 114.5) # Red (pHL = 114.2)
    # Breakout
    append_c(114.6, 118.0, 114.4, 117.5) # Green
    append_c(117.5, 122.4, 117.2, 121.8) # Green (close 121.8 > 119.5! cHL2 = 114.2. BUY ZONE 12 [119.5 - 114.2])

    # --- SWING 3 of Uptrend 2 ---
    append_c(121.8, 125.6, 121.5, 125.0) # Green
    append_c(125.0, 128.2, 124.7, 127.8) # Green (pHH = 128.2, green body [125.0, 127.8])
    # Pullback (closes below 125.0)
    append_c(127.6, 127.8, 123.8, 124.2) # Red (close 124.2 < 125.0! cHH3 = 128.2, pHL = 123.8)
    append_c(124.2, 124.5, 122.8, 123.2) # Red (pHL = 122.8)
    # Breakout
    append_c(123.3, 127.0, 123.0, 126.5) # Green
    append_c(126.5, 131.5, 126.2, 131.0) # Green (close 131.0 > 128.2! cHL3 = 122.8. BUY ZONE 13 [128.2 - 122.8])

    # --- SWING 4 of Uptrend 2 ---
    append_c(131.0, 134.8, 130.6, 134.2) # Green
    append_c(134.2, 137.6, 133.9, 137.0) # Green (pHH = 137.6, green body [134.2, 137.0])
    # Pullback (closes below 134.2)
    append_c(136.8, 137.0, 133.0, 133.5) # Red (close 133.5 < 134.2! cHH4 = 137.6, pHL = 133.0)
    append_c(133.5, 133.8, 131.8, 132.2) # Red (pHL = 131.8)
    # Breakout
    append_c(132.4, 136.0, 132.2, 135.5) # Green
    append_c(135.5, 141.0, 135.2, 140.4) # Green (close 140.4 > 137.6! cHL4 = 131.8. BUY ZONE 14 [137.6 - 131.8])

    # --- SWING 5 of Uptrend 2 ---
    append_c(140.4, 144.5, 140.0, 144.0) # Green
    append_c(144.0, 147.8, 143.6, 147.2) # Green (pHH = 147.8, green body [144.0, 147.2])
    # Pullback (closes below 144.0)
    append_c(147.0, 147.2, 143.0, 143.5) # Red (close 143.5 < 144.0! cHH5 = 147.8, pHL = 143.0)
    append_c(143.5, 143.8, 141.6, 142.0) # Red (pHL = 141.6)
    # Breakout
    append_c(142.2, 146.5, 142.0, 146.0) # Green
    append_c(146.0, 151.2, 145.8, 150.6) # Green (close 150.6 > 147.8! cHL5 = 141.6. BUY ZONE 15 [147.8 - 141.6])

    # --- SWING 6 of Uptrend 2 ---
    append_c(150.6, 154.5, 150.2, 154.0) # Green
    append_c(154.0, 157.6, 153.6, 157.0) # Green (pHH = 157.6, green body [154.0, 157.0])
    # Pullback (closes below 154.0)
    append_c(156.8, 157.0, 152.8, 153.2) # Red (close 153.2 < 154.0! cHH6 = 157.6, pHL = 152.8)
    append_c(153.2, 153.5, 151.0, 151.4) # Red (pHL = 151.0)
    # Breakout
    append_c(151.6, 156.0, 151.4, 155.5) # Green
    append_c(155.5, 161.0, 155.2, 160.2) # Green (close 160.2 > 157.6! cHL6 = 151.0. BUY ZONE 16 [157.6 - 151.0])
    # Final high
    append_c(160.2, 163.5, 160.0, 163.0) # Green

    return candles

if __name__ == "__main__":
    candles = generate_realistic_dataset()
    print(f"Generated realistic dataset with {len(candles)} candles.")
