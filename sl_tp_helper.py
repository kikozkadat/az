# core/sl_tp_helper.py

def calculate_sl_tp(entry_price: float, direction: str = "BUY", 
                   sl_percent: float = 1.5, tp_percent: float = 2.5) -> tuple:
    """
    Stop Loss és Take Profit szintek számítása
    
    Args:
        entry_price: Belépési ár
        direction: "BUY" vagy "SELL"
        sl_percent: Stop Loss százalék (alapértelmezett: 1.5%)
        tp_percent: Take Profit százalék (alapértelmezett: 2.5%)
        
    Returns:
        tuple: (stop_loss_price, take_profit_price)
    """
    if entry_price <= 0:
        raise ValueError("Entry price must be positive")
        
    if sl_percent <= 0 or tp_percent <= 0:
        raise ValueError("Percentages must be positive")
        
    sl_multiplier = sl_percent / 100.0
    tp_multiplier = tp_percent / 100.0
    
    if direction.upper() == "BUY":
        # Buy pozíciónál: SL alacsonyabb, TP magasabb
        stop_loss = entry_price * (1 - sl_multiplier)
        take_profit = entry_price * (1 + tp_multiplier)
    elif direction.upper() == "SELL":
        # Sell pozíciónál: SL magasabb, TP alacsonyabb
        stop_loss = entry_price * (1 + sl_multiplier)
        take_profit = entry_price * (1 - tp_multiplier)
    else:
        raise ValueError("Direction must be 'BUY' or 'SELL'")
        
    return round(stop_loss, 2), round(take_profit, 2)

def calculate_atr_sl_tp(prices: list, entry_price: float, direction: str = "BUY",
                       atr_multiplier: float = 2.0, tp_multiplier: float = 1.5) -> tuple:
    """
    ATR alapú Stop Loss és Take Profit számítása
    
    Args:
        prices: Ár lista (OHLC adatok)
        entry_price: Belépési ár
        direction: "BUY" vagy "SELL"
        atr_multiplier: ATR szorzó SL-hez
        tp_multiplier: TP/SL arány
        
    Returns:
        tuple: (stop_loss_price, take_profit_price)
    """
    if len(prices) < 14:
        # Ha nincs elég adat ATR számításhoz, visszatérünk fix percentagere
        return calculate_sl_tp(entry_price, direction)
        
    # Egyszerűsített ATR számítás (True Range átlag)
    true_ranges = []
    for i in range(1, len(prices)):
        if len(prices[i]) >= 4:  # OHLC formátum
            high = float(prices[i][2])
            low = float(prices[i][3])
            prev_close = float(prices[i-1][4])
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_ranges.append(max(tr1, tr2, tr3))
            
    if not true_ranges:
        return calculate_sl_tp(entry_price, direction)
        
    atr = sum(true_ranges[-14:]) / min(len(true_ranges), 14)
    
    if direction.upper() == "BUY":
        stop_loss = entry_price - (atr * atr_multiplier)
        take_profit = entry_price + (atr * atr_multiplier * tp_multiplier)
    else:  # SELL
        stop_loss = entry_price + (atr * atr_multiplier)
        take_profit = entry_price - (atr * atr_multiplier * tp_multiplier)
        
    return round(stop_loss, 2), round(take_profit, 2)

def calculate_risk_reward_ratio(entry_price: float, stop_loss: float, 
                               take_profit: float, direction: str) -> float:
    """
    Kockázat/hozam arány számítása
    
    Args:
        entry_price: Belépési ár
        stop_loss: Stop loss ár
        take_profit: Take profit ár
        direction: Pozíció iránya
        
    Returns:
        float: Risk/Reward ratio
    """
    if direction.upper() == "BUY":
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
    else:  # SELL
        risk = abs(stop_loss - entry_price)
        reward = abs(entry_price - take_profit)
        
    if risk == 0:
        return 0.0
        
    return reward / risk

def adjust_position_size(account_balance: float, risk_percent: float, 
                        entry_price: float, stop_loss: float) -> float:
    """
    Pozícióméret számítása kockázat alapján
    
    Args:
        account_balance: Számla egyenleg
        risk_percent: Kockázati százalék (pl. 1.0 = 1%)
        entry_price: Belépési ár
        stop_loss: Stop loss ár
        
    Returns:
        float: Ajánlott pozícióméret
    """
    if account_balance <= 0 or risk_percent <= 0:
        return 0.0
        
    risk_amount = account_balance * (risk_percent / 100.0)
    price_diff = abs(entry_price - stop_loss)
    
    if price_diff == 0:
        return 0.0
        
    position_size = risk_amount / price_diff
    return round(position_size, 6)
