# trade_data.py - Trade data interface

import time
from typing import List, Dict, Optional

def get_open_positions() -> List[Dict]:
    """
    Get all open positions - mock implementation
    This would normally interface with your position manager
    """
    # This is a mock implementation - in real use, you'd connect to your position manager
    return []

def get_current_price(symbol: str) -> Optional[float]:
    """
    Get current price for a symbol - mock implementation
    """
    # Mock prices for testing
    mock_prices = {
        'XBTUSD': 43250.0,
        'ETHUSD': 2650.0,
        'ADAUSD': 0.48,
        'XRPUSD': 0.62,
        'SOLUSD': 95.5
    }
    
    return mock_prices.get(symbol, 1000.0)

def get_portfolio_summary() -> Dict:
    """Get portfolio summary"""
    return {
        'total_value': 750.0,
        'available_cash': 750.0,
        'total_positions': 0,
        'daily_pnl': 0.0
    }

def get_trade_history(limit: int = 100) -> List[Dict]:
    """Get recent trade history"""
    return []

# Export functions
__all__ = ['get_open_positions', 'get_current_price', 'get_portfolio_summary', 'get_trade_history']
