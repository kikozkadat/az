# core/position_manager.py

import time
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class PositionManager:
    def __init__(self, max_positions: int = 5):
        self.positions: Dict[str, dict] = {}  # kulcs: pair, érték: pozíció adatok
        self.max_positions = max_positions
        self.position_history: List[dict] = []
        
    def open_position(self, pair: str, side: str, entry_price: float, volume: float, 
                     stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                     entry_time_unix: Optional[float] = None, reason: Optional[str] = None) -> bool:
        """
        Új pozíció nyitása
        
        Args:
            pair: Trading pair (pl. "BTCUSD")
            side: "buy" vagy "sell"
            entry_price: Belépési ár
            volume: Pozíció mérete
            stop_loss: Stop loss ár (opcionális)
            take_profit: Take profit ár (opcionális)
            entry_time_unix: Pozíció nyitásának unix timestamp-je (opcionális)
            reason: Pozíció nyitásának oka (opcionális)
            
        Returns:
            bool: Sikeres volt-e a pozíció nyitása
        """
        try:
            # Ellenőrzések
            if pair in self.positions:
                logger.warning(f"Position already exists for {pair}")
                return False
                
            if len(self.positions) >= self.max_positions:
                logger.warning(f"Maximum positions ({self.max_positions}) reached")
                return False
                
            if volume <= 0 or entry_price <= 0:
                logger.error(f"Invalid volume or price: {volume}, {entry_price}")
                return False
                
            # Entry time meghatározása
            if entry_time_unix is None:
                entry_time_unix = time.time()
            
            # Pozíció létrehozása
            position = {
                "pair": pair,
                "side": side.lower(),
                "entry_price": float(entry_price),
                "volume": float(volume),
                "stop_loss": float(stop_loss) if stop_loss else None,
                "take_profit": float(take_profit) if take_profit else None,
                "open_time": float(entry_time_unix),  # Unix timestamp használata
                "entry_time_unix": float(entry_time_unix),  # Kompatibilitás
                "unrealized_pnl": 0.0,
                "status": "open",
                "reason": reason or "MANUAL"  # Nyitás okának rögzítése
            }
            
            self.positions[pair] = position
            
            # Pozícióméret USD-ben számítás (ha szükséges)
            position_size_usd = volume * entry_price
            
            logger.info(f"[POSITION] Opened: {pair} {side.upper()} {volume:.6f} @ ${entry_price:.6f} (${position_size_usd:.2f}) - {reason or 'MANUAL'}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening position for {pair}: {e}", exc_info=True)
            return False
        
    def close_position(self, pair: str, exit_price: Optional[float] = None, 
                      reason: Optional[str] = None) -> Optional[dict]:
        """
        Pozíció zárása
        
        Args:
            pair: Trading pair
            exit_price: Kilépési ár (opcionális)
            reason: Zárás oka (opcionális)
            
        Returns:
            dict: Lezárt pozíció adatai vagy None
        """
        try:
            if pair not in self.positions:
                logger.warning(f"No position found for {pair}")
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
            position["close_reason"] = reason or "MANUAL"  # Zárás okának rögzítése
            
            # P&L számítás és státusz meghatározás
            pnl = position.get("realized_pnl", 0.0)
            if pnl > 0:
                position["result"] = "WIN"
            elif pnl < 0:
                position["result"] = "LOSS"
            else:
                position["result"] = "BREAKEVEN"
            
            # Pozíció eltávolítása az aktívakból és hozzáadása a históriához
            del self.positions[pair]
            self.position_history.append(position)
            
            # História méretének korlátozása
            if len(self.position_history) > 1000:
                self.position_history = self.position_history[-1000:]
                
            logger.info(f"[POSITION] Closed: {pair} P&L: ${pnl:.2f} ({reason or 'MANUAL'})")
            
            # Visszatérési érték a main_window számára
            return {
                "pair": pair,
                "side": position["side"],
                "entry_price": position["entry_price"],
                "exit_price": position.get("exit_price"),
                "volume": position["volume"],
                "pnl": pnl,
                "status": "closed",
                "reason": reason or "MANUAL",
                "result": position["result"]
            }
            
        except Exception as e:
            logger.error(f"Error closing position for {pair}: {e}", exc_info=True)
            return None
        
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
            float: Profit/Loss érték USD-ben
        """
        try:
            position = self.positions.get(pair)
            if not position:
                return 0.0
                
            entry_price = position["entry_price"]
            volume = position["volume"]
            side = position["side"]
            
            if side == "buy":
                # Long pozíció: profit ha az ár nő
                price_diff = current_price - entry_price
                pnl = price_diff * volume
            elif side == "sell":
                # Short pozíció: profit ha az ár csökken
                price_diff = entry_price - current_price
                pnl = price_diff * volume
            else:
                logger.warning(f"Unknown side '{side}' for position {pair}")
                pnl = 0.0
                
            # Frissítjük az unrealized P&L-t
            position["unrealized_pnl"] = pnl
            
            return pnl
            
        except Exception as e:
            logger.error(f"Error calculating P&L for {pair}: {e}", exc_info=True)
            return 0.0
        
    def calculate_total_pnl(self, current_prices: Dict[str, float]) -> dict:
        """
        Összes pozíció P&L számítása
        
        Args:
            current_prices: Aktuális árak dict formában {pair: price}
            
        Returns:
            dict: Összesített P&L adatok
        """
        try:
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
            
        except Exception as e:
            logger.error(f"Error calculating total P&L: {e}", exc_info=True)
            return {
                "total_unrealized": 0.0,
                "total_realized": 0.0,
                "total_pnl": 0.0,
                "active_positions": 0,
                "closed_positions": 0
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
        try:
            if pair not in self.positions:
                logger.warning(f"Position not found for stop loss update: {pair}")
                return False
                
            self.positions[pair]["stop_loss"] = float(new_stop_loss)
            logger.info(f"[POSITION] Updated stop loss for {pair}: ${new_stop_loss:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating stop loss for {pair}: {e}", exc_info=True)
            return False
        
    def update_take_profit(self, pair: str, new_take_profit: float) -> bool:
        """
        Take profit frissítése
        
        Args:
            pair: Trading pair
            new_take_profit: Új take profit ár
            
        Returns:
            bool: Sikeres volt-e a frissítés
        """
        try:
            if pair not in self.positions:
                logger.warning(f"Position not found for take profit update: {pair}")
                return False
                
            self.positions[pair]["take_profit"] = float(new_take_profit)
            logger.info(f"[POSITION] Updated take profit for {pair}: ${new_take_profit:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating take profit for {pair}: {e}", exc_info=True)
            return False
        
    def check_stop_loss_take_profit(self, current_prices: Dict[str, float]) -> List[str]:
        """
        Stop loss és take profit szintek ellenőrzése
        
        Args:
            current_prices: Aktuális árak
            
        Returns:
            List[str]: Zárásra jelölt pozíciók listája
        """
        positions_to_close = []
        
        try:
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
                    logger.info(f"[POSITION] {pair} triggered {close_reason} at ${current_price:.6f}")
                    
        except Exception as e:
            logger.error(f"Error checking SL/TP levels: {e}", exc_info=True)
                
        return positions_to_close
        
    def get_statistics(self) -> dict:
        """
        Pozíció statisztikák lekérdezése
        
        Returns:
            dict: Statisztikai adatok
        """
        try:
            if not self.position_history:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "total_profit": 0.0,
                    "average_pnl": 0.0,
                    "max_profit": 0.0,
                    "max_loss": 0.0,
                    "average_hold_time": 0.0
                }
                
            closed_positions = [pos for pos in self.position_history if pos.get("realized_pnl") is not None]
            
            if not closed_positions:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "total_profit": 0.0,
                    "average_pnl": 0.0,
                    "max_profit": 0.0,
                    "max_loss": 0.0,
                    "average_hold_time": 0.0
                }
                
            pnls = [pos["realized_pnl"] for pos in closed_positions]
            winning_trades = [pnl for pnl in pnls if pnl > 0]
            losing_trades = [pnl for pnl in pnls if pnl < 0]
            hold_times = [pos.get("hold_duration", 0) for pos in closed_positions]
            
            total_profit = sum(pnls)
            win_rate = len(winning_trades) / len(closed_positions) if closed_positions else 0.0
            
            return {
                "total_trades": len(closed_positions),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": win_rate,
                "total_profit": total_profit,
                "average_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
                "max_profit": max(pnls) if pnls else 0.0,
                "max_loss": min(pnls) if pnls else 0.0,
                "average_hold_time": sum(hold_times) / len(hold_times) if hold_times else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}", exc_info=True)
            return {"total_trades": 0, "win_rate": 0.0, "total_profit": 0.0}
        
    def set_max_positions(self, max_positions: int):
        """Maximum pozíciók számának beállítása"""
        try:
            self.max_positions = max(1, max_positions)
            logger.info(f"[POSITION] Max positions set to: {self.max_positions}")
        except Exception as e:
            logger.error(f"Error setting max positions: {e}", exc_info=True)
        
    def clear_all_positions(self, current_prices: Optional[Dict[str, float]] = None):
        """Összes pozíció zárása"""
        try:
            closed_count = 0
            for pair in list(self.positions.keys()):
                exit_price = current_prices.get(pair) if current_prices else None
                if self.close_position(pair, exit_price, reason="CLEAR_ALL"):
                    closed_count += 1
            logger.info(f"[POSITION] All positions cleared: {closed_count} positions closed")
        except Exception as e:
            logger.error(f"Error clearing all positions: {e}", exc_info=True)

    def close_all_positions_market(self, current_prices: Optional[Dict[str, float]] = None) -> List[dict]:
        """
        Összes pozíció azonnali piaci áron történő zárása (vészhelyzeti funkció)
        
        Args:
            current_prices: Aktuális árak dict formában
            
        Returns:
            List[dict]: Lezárt pozíciók adatai
        """
        try:
            closed_positions = []
            
            for pair in list(self.positions.keys()):
                exit_price = current_prices.get(pair) if current_prices else None
                closed_pos = self.close_position(pair, exit_price, reason="EMERGENCY_MARKET_CLOSE")
                if closed_pos:
                    closed_positions.append(closed_pos)
            
            logger.info(f"[POSITION] Emergency market close: {len(closed_positions)} positions closed")
            return closed_positions
            
        except Exception as e:
            logger.error(f"Error in emergency market close: {e}", exc_info=True)
            return []

    def get_position_summary(self) -> dict:
        """
        Pozíciók összesítő adatai
        
        Returns:
            dict: Összesítő adatok
        """
        try:
            active_count = len(self.positions)
            total_volume_usd = 0.0
            
            for pair, position in self.positions.items():
                volume_usd = position["volume"] * position["entry_price"]
                total_volume_usd += volume_usd
            
            return {
                "active_positions": active_count,
                "max_positions": self.max_positions,
                "total_volume_usd": total_volume_usd,
                "available_slots": self.max_positions - active_count,
                "position_history_count": len(self.position_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting position summary: {e}", exc_info=True)
            return {
                "active_positions": 0,
                "max_positions": self.max_positions,
                "total_volume_usd": 0.0,
                "available_slots": self.max_positions,
                "position_history_count": 0
            }
