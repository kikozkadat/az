# utils/logger.py - Egyszerű logger

import logging
import os
import sys
from datetime import datetime

# Logs mappa létrehozása
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Logger konfigurálása
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/trading_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Main logger
logger = logging.getLogger("trading_bot")

# Additional loggers for specific components
api_logger = logging.getLogger("trading_bot.api")
strategy_logger = logging.getLogger("trading_bot.strategy")
ui_logger = logging.getLogger("trading_bot.ui")
trading_logger = logging.getLogger("trading_bot.trading")

# Set levels
logger.setLevel(logging.INFO)
api_logger.setLevel(logging.INFO)
strategy_logger.setLevel(logging.INFO)
ui_logger.setLevel(logging.INFO)
trading_logger.setLevel(logging.INFO)

def log_trade(pair: str, action: str, price: float, amount: float, reason: str = ""):
    """Trade-specifikus log"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trading_logger.info(f"[TRADE] {timestamp} - {pair} {action} {amount:.6f} @ {price:.2f} - {reason}")

def log_error(component: str, error: Exception, context: str = ""):
    """Error log formázással"""
    error_msg = f"[ERROR] {component}: {str(error)}"
    if context:
        error_msg += f" - Context: {context}"
    logger.error(error_msg)

def log_performance(metrics: dict):
    """Performance metrikai logolása"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    perf_msg = f"[PERFORMANCE] {timestamp} - "
    perf_msg += f"Trades: {metrics.get('total_trades', 0)}, "
    perf_msg += f"Win Rate: {metrics.get('win_rate', 0):.1%}, "
    perf_msg += f"Total P&L: ${metrics.get('total_profit', 0):.2f}"
    logger.info(perf_msg)

def log_micro_trading(position_size: float, profit_target: float, max_loss: float):
    """Mikro-trading specifikus log"""
    logger.info(f"[MICRO] Position: ${position_size:.0f}, Target: ${profit_target:.2f}, Max Loss: ${max_loss:.2f}")

# Export main logger
__all__ = ['logger', 'api_logger', 'strategy_logger', 'ui_logger', 'trading_logger', 
           'log_trade', 'log_error', 'log_performance', 'log_micro_trading']
