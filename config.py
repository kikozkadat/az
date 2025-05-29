import os
from dotenv import load_dotenv

load_dotenv()

# 🔐 API kulcsok
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET")

# ⚙️ Kereskedési logika kapcsoló - ÉLES MÓD!
TRADE_ENABLED = True

# 🎯 MIKRO-TRADING ALAPBEÁLLÍTÁSOK ($50 pozíciók)
TRADE_VOLUME = 0.001
DEFAULT_TRADE_USD = 50          
PER_TRADE_USD = 50              
MAX_PARALLEL_POSITIONS = 1      # ✅ MÓDOSÍTVA: Max 1 pozíció ($66 egyenleghez)
DEFAULT_PAIRS = ["BTCUSD", "ETHUSD"]

# 💰 MIKRO-TRADING SPECIFIKUS ÉRTÉKEK
MICRO_TRADING_ENABLED = True
MICRO_POSITION_SIZE = 50.0      # $50 pozíció méret
MICRO_PROFIT_TARGET = 0.15      # $0.15 profit cél
MICRO_MAX_LOSS = 2.0           # $2 maximum veszteség
MICRO_TARGET_PROFIT_PCT = 0.30  # 0.30% profit százalék
MICRO_MAX_LOSS_PCT = 4.0       # 4% maximum veszteség százalék

# 📊 Technikai indikátor beállítások
RSI_PERIOD = 14
EMA_FAST = 9
EMA_SLOW = 21
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14

# 🎯 MIKRO-TRADING SL/TP BEÁLLÍTÁSOK
DEFAULT_STOPLOSS_PERCENT = 4.0    
DEFAULT_TAKEPROFIT_PERCENT = 0.8  

# ⏰ IDŐKEZELÉSI BEÁLLÍTÁSOK
MAX_HOLD_SECONDS = 7200           # 2 óra max
MIN_HOLD_SECONDS = 300            # Min 5 perc tartás
QUICK_PROFIT_TIME_MINUTES = 15    # 15 perc után gyors profit ellenőrzés

# 🚀 AI ÉS AUTOMATIZÁLÁSI BEÁLLÍTÁSOK
AI_CONFIDENCE_THRESHOLD = 0.65    
AI_SCAN_INTERVAL_SECONDS = 120    # 2 perces scan
AI_POSITION_CHECK_SECONDS = 30    # 30 másodperces pozíció ellenőrzés

# 🛡️ KOCKÁZATKEZELÉSI BEÁLLÍTÁSOK - SZIGORÚBB $66 EGYENLEGHEZ
MAX_DAILY_LOSS = 10.0            # ✅ MÓDOSÍTVA: $10 maximum napi veszteség ($66 egyenleghez)
MAX_RISK_PER_TRADE_PCT = 8.0     
PORTFOLIO_MAX_EXPOSURE_PCT = 15.0 # ✅ MÓDOSÍTVA: 15% max kitettség (volt 20%)

# 📈 TELJESÍTMÉNY CÉLOK - REÁLIS $66 EGYENLEGHEZ
DAILY_PROFIT_TARGET = 0.15       # ✅ MÓDOSÍTVA: $0.15 napi profit cél (1 pozíció)
WEEKLY_PROFIT_TARGET = 0.75      # ✅ MÓDOSÍTVA: $0.75 heti profit cél (5 nap × $0.15)
MONTHLY_PROFIT_TARGET = 3.0      # ✅ MÓDOSÍTVA: $3.00 havi profit cél (20 nap × $0.15)

# 🎲 MIKRO-TRADING STRATÉGIA BEÁLLÍTÁSOK
MICRO_STRATEGY_SETTINGS = {
    "position_size": MICRO_POSITION_SIZE,
    "profit_target": MICRO_PROFIT_TARGET,
    "max_loss": MICRO_MAX_LOSS,
    "target_profit_pct": MICRO_TARGET_PROFIT_PCT,
    "max_loss_pct": MICRO_MAX_LOSS_PCT,
    "min_hold_time_minutes": 5,
    "max_hold_time_hours": 2,
    "quick_profit_threshold": 0.10,  
    "emergency_exit_loss": 3.0,      # $3 vészkilépés
    "fee_aware_trading": True,
    "scalping_mode": True
}

# 🔍 SCANNER BEÁLLÍTÁSOK MIKRO-TRADINGHEZ
SCANNER_SETTINGS = {
    "min_volume_24h": 500000,        
    "max_pairs_to_analyze": 80,      
    "scan_interval_seconds": 180,    
    "batch_size": 8,                 
    "min_score_threshold": 0.4,      
    "quick_scan_enabled": True,
    "priority_pairs_only": False
}

# 🎯 MIKRO-TRADING INDIKÁTOR SÚLYOK
MICRO_INDICATOR_WEIGHTS = {
    "rsi_weight": 0.35,              
    "macd_weight": 0.25,             
    "bollinger_weight": 0.20,        
    "volume_weight": 0.10,           
    "correlation_weight": 0.10       
}

# ⚡ GYORS PROFIT BEÁLLÍTÁSOK
QUICK_PROFIT_SETTINGS = {
    "enabled": True,
    "target_pct": 0.6,               
    "min_hold_minutes": 15,          
    "rsi_threshold": 25,             
    "volume_multiplier": 2.0         
}

# 📊 LOGGING ÉS MONITORING
LOGGING_CONFIG = {
    "log_all_trades": True,
    "log_micro_details": True,
    "log_ai_decisions": True,
    "log_performance_metrics": True,
    "detailed_pnl_tracking": True,
    "profit_loss_notifications": True
}

# 🌐 API ÉS KAPCSOLAT BEÁLLÍTÁSOK
API_SETTINGS = {
    "rate_limit_delay": 0.5,         
    "request_timeout": 10,           
    "max_retries": 3,               
    "batch_request_delay": 2,        
    "websocket_enabled": False,      
    "fallback_mode": True           
}

# 🎨 UI ÉS MEGJELENÍTÉSI BEÁLLÍTÁSOK
UI_SETTINGS = {
    "update_interval_ms": 20000,     
    "chart_candle_count": 100,       
    "real_time_updates": True,       
    "show_micro_details": True,      
    "profit_loss_colors": True,      
    "sound_notifications": False     
}

# 🔧 FEJLESZTŐI BEÁLLÍTÁSOK - ✅ ÉLES MÓD BEKAPCSOLVA!
DEBUG_SETTINGS = {
    "debug_mode": False,
    "verbose_logging": True,
    "simulation_mode": False,         # ✅ MÓDOSÍTVA: False = ÉLES MÓD!
    "paper_trading": False,           # ✅ MÓDOSÍTVA: False = VALÓDI KERESKEDÉS!
    "test_mode": False,
    "dry_run": False                 # ✅ MÓDOSÍTVA: False = VALÓDI VÉGREHAJTÁS!
}

# 🎯 MIKRO-TRADING VALIDÁCIÓS SZABÁLYOK - $66 EGYENLEGHEZ IGAZÍTVA
VALIDATION_RULES = {
    "min_position_size": 25.0,       # Min $25 pozíció
    "max_position_size": 60.0,       # ✅ MÓDOSÍTVA: Max $60 pozíció ($66 egyenleghez)
    "min_profit_target": 0.05,       # Min $0.05 profit
    "max_profit_target": 1.0,        # Max $1.00 profit
    "min_confidence": 0.6,           # Min 60% konfidencia
    "max_daily_trades": 10,          # ✅ MÓDOSÍTVA: Max 10 trade naponta (volt 20)
    "min_account_balance": 50.0      # ✅ MÓDOSÍTVA: Min $50 számla egyenleg (volt $200)
}

# 🚨 VÉSZHELYZETI BEÁLLÍTÁSOK - SZIGORÚBB $66 EGYENLEGHEZ
EMERGENCY_SETTINGS = {
    "emergency_stop_loss_pct": 8.0,  # ✅ MÓDOSÍTVA: 8% emergency stop (volt 10%)
    "max_consecutive_losses": 3,     # ✅ MÓDOSÍTVA: Max 3 egymás utáni veszteség (volt 5)
    "circuit_breaker_enabled": True, 
    "daily_loss_limit": 10.0,       # ✅ MÓDOSÍTVA: $10 napi veszteség limit (volt $20)
    "auto_pause_on_loss": True,     
    "recovery_mode_enabled": True    
}

# 🎓 TANULÁSI ÉS AI BEÁLLÍTÁSOK
LEARNING_SETTINGS = {
    "learning_enabled": True,
    "min_trades_for_learning": 10,   
    "learning_update_frequency": 25, 
    "adaptive_weights": True,        
    "performance_tracking": True,    
    "pattern_recognition": True,     
    "auto_optimization": False       
}

# 📁 FÁJL ÚTVONALAK
FILE_PATHS = {
    "trade_log": "logs/micro_trade_log.csv",
    "performance_log": "logs/micro_performance.json", 
    "learning_data": "data/micro_learning_data.json",
    "ai_weights": "data/micro_ai_weights.pkl",
    "settings_backup": "backup/micro_settings.json",
    "error_log": "logs/micro_errors.log"
}

# 💼 PORTFÓLIÓ BEÁLLÍTÁSOK - 1 POZÍCIÓRA OPTIMALIZÁLVA
PORTFOLIO_SETTINGS = {
    "max_portfolio_value": 100.0,    # ✅ MÓDOSÍTVA: Max $100 portfólió érték (volt $1000)
    "diversification_enabled": False, # ✅ MÓDOSÍTVA: Kikapcsolva 1 pozíciónál
    "correlation_limit": 0.8,        
    "rebalancing_enabled": False,    
    "hedging_enabled": False,        
    "position_sizing_method": "fixed" 
}

# 🎉 PROFIT CELEBRATION BEÁLLÍTÁSOK
CELEBRATION_SETTINGS = {
    "celebrate_profits": True,
    "daily_target_notification": True,
    "weekly_summary": True,
    "milestone_tracking": True,
    "profit_streak_counting": True,
    "achievement_system": True
}

# 🔄 EXPORT/IMPORT BEÁLLÍTÁSOK  
EXPORT_SETTINGS = {
    "auto_export_enabled": False,
    "export_format": "json",
    "backup_frequency_hours": 24,
    "cloud_backup_enabled": False,
    "local_backup_enabled": True,
    "export_on_shutdown": True
}
