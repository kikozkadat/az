# core/position_manager.py

import time
from typing import Dict, Optional, List

class PositionManager:
    def __init__(self, max_positions: int = 5):
        self.positions: Dict[str, dict] = {}  # kulcs: pair, érték: pozíció adatok
        self.max_positions = max_positions
        self.position_history: List[dict] = []
        
    def open_position(self, pair: str, side: str, entry_price: float, volume: float, 
                     stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> bool:
        """
        Új pozíció nyitása
        
        Args:
            pair: Trading pair (pl. "BTCUSD")
            side: "buy" vagy "sell"
            entry_price: Belépési ár
            volume: Pozíció mérete
            stop_loss: Stop loss ár (opcionális)
            take_profit: Take profit ár (opcionális)
            
        Returns:
            bool: Sikeres volt-e a pozíció nyitása
        """
        # Ellenőrzések
        if pair in self.positions:
            print(f"[ERROR] Position already exists for {pair}")
            return False
            
        if len(self.positions) >= self.max_positions:
            print(f"[ERROR] Maximum positions ({self.max_positions}) reached")
            return False
            
        if volume <= 0 or entry_price <= 0:
            print(f"[ERROR] Invalid volume or price: {volume}, {entry_price}")
            return False
            
        # Pozíció létrehozása
        position = {
            "pair": pair,
            "side": side.lower(),
            "entry_price": float(entry_price),
            "volume": float(volume),
            "stop_loss": float(stop_loss) if stop_loss else None,
            "take_profit": float(take_profit) if take_profit else None,
            "open_time": time.time(),
            "unrealized_pnl": 0.0,
            "status": "open"
        }
        
        self.positions[pair] = position
        print(f"[POSITION] Opened: {pair} {side.upper()} {volume:.6f} @ {entry_price:.2f}")
        return True
        
    def close_position(self, pair: str, exit_price: Optional[float] = None) -> Optional[dict]:
        """
        Pozíció zárása
        
        Args:
            pair: Trading pair
            exit_price: Kilépési ár (opcionális)
            
        Returns:
            dict: Lezárt pozíció adatai vagy None
        """
        if pair not in self.positions:
            print(f"[ERROR] No position found for {pair}")
            return None
            
        position = self.positions[pair].copy()
        
        # Kilépési ár és P&L számítás
        if exit_price:
            position["exit_price"] = float(exit_price)
            position["realized_pnl"] = self.calculate_pnl(pair, exit_price)
        else:
            position["exit_price"] = None
            position["realized_pnl"] = 0.0
            
        position["close_time"] = time.time()
        position["status"] = "closed"
        position["hold_duration"] = position["close_time"] - position["open_time"]
        
        # Pozíció eltávolítása az aktívakból és hozzáadása a históriához
        del self.positions[pair]
        self.position_history.append(position)
        
        # História méretének korlátozása
        if len(self.position_history) > 1000:
            self.position_history = self.position_history[-1000:]
            
        print(f"[POSITION] Closed: {pair} P&L: {position['realized_pnl']:.2f}")
        return position
        
    def get_position(self, pair: str) -> Optional[dict]:
        """
        Pozíció lekérdezése
        
        Args:
            pair: Trading pair
            
        Returns:
            dict: Pozíció adatai vagy None
        """
        return self.positions.get(pair)
        
    def get_all_positions(self) -> Dict[str, dict]:
        """Összes aktív pozíció lekérdezése"""
        return self.positions.copy()
        
    def calculate_pnl(self, pair: str, current_price: float) -> float:
        """
        P&L számítás egy pozícióra
        
        Args:
            pair: Trading pair
            current_price: Jelenlegi ár
            
        Returns:
            float: Profit/Loss érték
        """
        position = self.positions.get(pair)
        if not position:
            return 0.0
            
        entry_price = position["entry_price"]
        volume = position["volume"]
        side = position["side"]
        
        if side == "buy":
            pnl = (current_price - entry_price) * volume
        elif side == "sell":
            pnl = (entry_price - current_price) * volume
        else:
            pnl = 0.0
            
        # Frissítjük az unrealized P&L-t
        position["unrealized_pnl"] = pnl
        
        return pnl
        
    def calculate_total_pnl(self, current_prices: Dict[str, float]) -> dict:
        """
        Összes pozíció P&L számítása
        
        Args:
            current_prices: Aktuális árak dict formában {pair: price}
            
        Returns:
            dict: Összesített P&L adatok
        """
        total_unrealized = 0.0
        total_realized = sum(pos.get("realized_pnl", 0.0) for pos in self.position_history)
        
        for pair, position in self.positions.items():
            if pair in current_prices:
                pnl = self.calculate_pnl(pair, current_prices[pair])
                total_unrealized += pnl
                
        return {
            "total_unrealized": total_unrealized,
            "total_realized": total_realized,
            "total_pnl": total_unrealized + total_realized,
            "active_positions": len(self.positions),
            "closed_positions": len(self.position_history)
        }
        
    def update_stop_loss(self, pair: str, new_stop_loss: float) -> bool:
        """
        Stop loss frissítése
        
        Args:
            pair: Trading pair
            new_stop_loss: Új stop loss ár
            
        Returns:
            bool: Sikeres volt-e a frissítés
        """
        if pair not in self.positions:
            return False
            
        self.positions[pair]["stop_loss"] = float(new_stop_loss)
        print(f"[POSITION] Updated stop loss for {pair}: {new_stop_loss:.2f}")
        return True
        
    def update_take_profit(self, pair: str, new_take_profit: float) -> bool:
        """
        Take profit frissítése
        
        Args:
            pair: Trading pair
            new_take_profit: Új take profit ár
            
        Returns:
            bool: Sikeres volt-e a frissítés
        """
        if pair not in self.positions:
            return False
            
        self.positions[pair]["take_profit"] = float(new_take_profit)
        print(f"[POSITION] Updated take profit for {pair}: {new_take_profit:.2f}")
        return True
        
    def check_stop_loss_take_profit(self, current_prices: Dict[str, float]) -> List[str]:
        """
        Stop loss és take profit szintek ellenőrzése
        
        Args:
            current_prices: Aktuális árak
            
        Returns:
            List[str]: Zárásra jelölt pozíciók listája
        """
        positions_to_close = []
        
        for pair, position in self.positions.items():
            if pair not in current_prices:
                continue
                
            current_price = current_prices[pair]
            side = position["side"]
            stop_loss = position.get("stop_loss")
            take_profit = position.get("take_profit")
            
            should_close = False
            close_reason = ""
            
            if side == "buy":
                if stop_loss and current_price <= stop_loss:
                    should_close = True
                    close_reason = "STOP_LOSS"
                elif take_profit and current_price >= take_profit:
                    should_close = True
                    close_reason = "TAKE_PROFIT"
            elif side == "sell":
                if stop_loss and current_price >= stop_loss:
                    should_close = True
                    close_reason = "STOP_LOSS"
                elif take_profit and current_price <= take_profit:
                    should_close = True
                    close_reason = "TAKE_PROFIT"
                    
            if should_close:
                positions_to_close.append(pair)
                print(f"[POSITION] {pair} triggered {close_reason} at {current_price:.2f}")
                
        return positions_to_close
        
    def get_statistics(self) -> dict:
        """
        Pozíció statisztikák lekérdezése
        
        Returns:
            dict: Statisztikai adatok
        """
        if not self.position_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "average_pnl": 0.0,
                "max_profit": 0.0,
                "max_loss": 0.0,
                "average_hold_time": 0.0
            }
            
        closed_positions = [pos for pos in self.position_history if pos.get("realized_pnl") is not None]
        
        if not closed_positions:
            return {"total_trades": 0}
            
        pnls = [pos["realized_pnl"] for pos in closed_positions]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        hold_times = [pos.get("hold_duration", 0) for pos in closed_positions]
        
        return {
            "total_trades": len(closed_positions),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(closed_positions) * 100,
            "average_pnl": sum(pnls) / len(pnls),
            "max_profit": max(pnls) if pnls else 0.0,
            "max_loss": min(pnls) if pnls else 0.0,
            "average_hold_time": sum(hold_times) / len(hold_times) if hold_times else 0.0
        }
        
    def set_max_positions(self, max_positions: int):
        """Maximum pozíciók számának beállítása"""
        self.max_positions = max(1, max_positions)
        print(f"[POSITION] Max positions set to: {self.max_positions}")
        
    def clear_all_positions(self, current_prices: Optional[Dict[str, float]] = None):
        """Összes pozíció zárása"""
        for pair in list(self.positions.keys()):
            exit_price = current_prices.get(pair) if current_prices else None
            self.close_position(pair, exit_price)
        print("[POSITION] All positions cleared")
