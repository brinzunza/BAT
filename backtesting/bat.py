onedf = pandasDF # pandas dataframe with candle high, low, open, close
fifteendf = pandasDF # pandas dataframe with candle high, low, open, close

onetrend = 0 # predetermined input to start; 1 is up, 0 is down
fifteentrend = 0

onehigh = CANDLE # last valid high
onelow = CANDLE # last valid low
fifteenhigh = CANDLE
fifteenlow = CANDLE

onepossible_high = CANDLE #potential high
onepossible_low = CANDLE #potential low
fifteenpossible_high = CANDLE 
fifteenpossible_low = CANDLE

one_last_candle = (candle_high, candle_low) # remember last candle for closure outside
one_curr_candle = (candle_close) #keep track of the current important candle
fifteen_last_candle = ()
fifteen_curr_candle = ()

onesignals = []
fifteensignals = []

finalSignals = []

def structure(): # trends with swings
    if uptrend and curr_candle closes below last candle:
        track possible_high and possible_low
        if curr_candle closes above possible high:
            set new valid high and valid low
            signal()
        if curr_candle closes below valid low:
            set new valid high and valid low
            change to downtrend
            signal()
    if uptrend and curr_candle closes above:
        set new possible_high

    if downtrend and curr_candle closes below above candle:
        keep track of possible_low and possible_high
        if curr_candle closes below possible_low:
            set new valid high and valid low
            signal(trend)
        if curr_candle closes above valid_high:
            set new valid high and valid low
            change to uptrend
            signal(trend)
    if downtrend and curr_candle closes below:
        set new possible_low

def signal():
    zone_entry_buy = (valid high - valid_low / 2) + valid_low
    zone_entry_low = (valid_low - valid_high / 2) + valid_high

def finalSignal():
    if uptrend and curr_candle closes below zone_entry_buy:
        buy(valid_low, valid_high)
    if downtrend and curr_candle closes above zone_entry_low:
        sell(valid_high, valid_low)

def buy(stoploss, takeProfit):


def sell(stoploss, takeProfit):


def run():
    if price gets to a signal:
        buy() or sell()