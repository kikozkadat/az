# strategy/profit_calculator.py - Egyszerűsített verzió $50 mikro-tradinghez

from typing import Dict

class ProfitCalculator:
    """
    $50 pozíciókra optimalizált profit számítások
    Cél: $0.15-0.20 profit $50 befektetésből
    """
    
    def __init__(self):
        # Kraken fee struktúra (taker fees)
        self.default_fee_rate = 0.0026  # 0.26% alapértelmezett fee
        
        # 🎯 MIKRO-TRADING PROFIT CÉLOK ($50 pozíciókra)
        self.min_profit_targets = {
            25: 0.08,    # $25 -> $0.08 profit (0.32%)
            50: 0.15,    # $50 -> $0.15 profit (0.30%)
            75: 0.20,    # $75 -> $0.20 profit (0.27%)
            100: 0.25,   # $100 -> $0.25 profit (0.25%)
        }

    def calculate_micro_trading_params(self, investment_amount: float = 50.0) -> Dict:
        """
        $50 mikro-trading paraméterek számítása
        
        Args:
            investment_amount: Befektetett összeg (alapértelmezett: $50)
            
        Returns:
            dict: Mikro-trading paraméterek
        """
        try:
            # Fee számítások
            buy_fee = investment_amount * self.default_fee_rate  # ~$0.13
            
            # Target profit (0.30% a befektetésből)
            target_profit = 0.15  # $0.15
            
            # Szükséges gross sell value
            required_gross_sell = investment_amount + target_profit
            
            # Sell fee számítása
            required_sell_amount = required_gross_sell / (1 - self.default_fee_rate)
            sell_fee = required_sell_amount * self.default_fee_rate
            
            # Szükséges árváltozás
            price_change_required = (required_sell_amount - investment_amount) / investment_amount
            
            # Break-even pont
            total_fees = buy_fee + sell_fee
            break_even_change = total_fees / investment_amount
            
            return {
                'investment_amount': investment_amount,
                'target_profit_usd': target_profit,
                'target_profit_pct': (target_profit / investment_amount) * 100,
                'buy_fee': buy_fee,
                'sell_fee_estimated': sell_fee,
                'total_fees': total_fees,
                'required_price_change_pct': price_change_required * 100,
                'break_even_price_change_pct': break_even_change * 100,
                'min_profit_after_fees': target_profit,
                'fee_ratio': total_fees / investment_amount * 100,
                'effective_investment': investment_amount - buy_fee
            }
        except Exception as e:
            print(f"Error calculating micro trading params: {e}")
            return {'investment_amount': investment_amount, 'target_profit_usd': 0.15}

    def calculate_micro_sl_tp(self, entry_price: float, 
                             investment_amount: float = 50.0) -> Dict:
        """
        $50 pozícióra optimalizált SL/TP számítása
        
        Args:
            entry_price: Belépési ár
            investment_amount: Pozíció mérete ($50)
            
        Returns:
            dict: SL/TP szintek mikro-tradinghez
        """
        try:
            params = self.calculate_micro_trading_params(investment_amount)
            
            # Take Profit ár (0.8% árváltozás kell $0.15 profithoz)
            tp_price_change = params['required_price_change_pct'] / 100
            take_profit_price = entry_price * (1 + tp_price_change)
            
            # Stop Loss (tighter SL mikro-tradingnél)
            # $50-nél max $2-3 veszteség (4-6%)
            max_loss_pct = 0.04  # 4% max loss = $2
            stop_loss_price = entry_price * (1 - max_loss_pct)
            
            # Risk/Reward ratio
            potential_gain = take_profit_price - entry_price
            potential_loss = entry_price - stop_loss_price
            risk_reward_ratio = potential_gain / potential_loss if potential_loss > 0 else 0
            
            return {
                'entry_price': entry_price,
                'take_profit_price': take_profit_price,
                'stop_loss_price': stop_loss_price,
                'tp_price_change_pct': tp_price_change * 100,
                'sl_price_change_pct': -max_loss_pct * 100,
                'risk_reward_ratio': risk_reward_ratio,
                'target_profit_usd': params['target_profit_usd'],
                'max_loss_usd': investment_amount * max_loss_pct,
                'position_size_usd': investment_amount,
                'required_accuracy_pct': (1 / (1 + risk_reward_ratio)) * 100 if risk_reward_ratio > 0 else 50  # Win rate needed
            }
        except Exception as e:
            print(f"Error calculating micro SL/TP: {e}")
            # Fallback calculation
            return {
                'entry_price': entry_price,
                'take_profit_price': entry_price * 1.008,  # 0.8% TP
                'stop_loss_price': entry_price * 0.96,    # 4% SL
                'tp_price_change_pct': 0.8,
                'sl_price_change_pct': -4.0,
                'risk_reward_ratio': 0.2,
                'target_profit_usd': 0.15,
                'max_loss_usd': 2.0,
                'position_size_usd': investment_amount
            }

    def optimize_for_micro_trading(self, available_balance: float) -> Dict:
        """
        Mikro-trading optimalizálás
        
        Args:
            available_balance: Elérhető egyenleg
            
        Returns:
            dict: Optimalizált mikro-trading beállítások
        """
        try:
            # $50 pozíciók, max 5-10% egyenleg felhasználása
            max_position_size = min(50.0, available_balance * 0.10)
            
            # Pozíciók száma az egyenleg alapján
            if available_balance >= 1000:
                max_positions = 5
                position_size = 50.0
            elif available_balance >= 500:
                max_positions = 3
                position_size = 50.0
            elif available_balance >= 200:
                max_positions = 2
                position_size = min(50.0, available_balance * 0.15)
            else:
                max_positions = 1
                position_size = min(25.0, available_balance * 0.20)
            
            return {
                'recommended_position_size': position_size,
                'max_concurrent_positions': max_positions,
                'total_exposure_pct': (position_size * max_positions) / available_balance * 100,
                'single_position_risk_pct': position_size / available_balance * 100,
                'daily_profit_target': 0.15 * max_positions,  # $0.15 * pozíciók száma
                'daily_max_loss': position_size * max_positions * 0.04  # 4% max loss per position
            }
        except Exception as e:
            print(f"Error optimizing for micro trading: {e}")
            return {
                'recommended_position_size': 50.0,
                'max_concurrent_positions': 3,
                'total_exposure_pct': 15.0,
                'daily_profit_target': 0.45
            }

    def calculate_fees(self, trade_amount: float, fee_rate: float = None) -> Dict:
        """
        Trading fee számítás
        
        Args:
            trade_amount: Trade összeg
            fee_rate: Fee százalék (alapértelmezett: 0.26%)
            
        Returns:
            dict: Fee információk
        """
        try:
            if fee_rate is None:
                fee_rate = self.default_fee_rate
                
            fee_amount = trade_amount * fee_rate
            net_amount = trade_amount - fee_amount
            
            return {
                'gross_amount': trade_amount,
                'fee_amount': fee_amount,
                'net_amount': net_amount,
                'fee_rate_pct': fee_rate * 100
            }
        except Exception as e:
            print(f"Error calculating fees: {e}")
            return {'gross_amount': trade_amount, 'fee_amount': 0, 'net_amount': trade_amount}

    def calculate_break_even_price(self, entry_price: float, position_size: float) -> float:
        """
        Break-even ár számítása fee-kkel
        
        Args:
            entry_price: Belépési ár
            position_size: Pozíció méret USD-ben
            
        Returns:
            float: Break-even ár
        """
        try:
            # Total fees (buy + sell)
            total_fee_rate = self.default_fee_rate * 2
            
            # Break-even price needs to cover fees
            break_even_price = entry_price * (1 + total_fee_rate)
            
            return break_even_price
        except Exception as e:
            print(f"Error calculating break-even price: {e}")
            return entry_price * 1.0052  # ~0.52% for fees

    def validate_trade_params(self, entry_price: float, stop_loss: float, 
                            take_profit: float, position_size: float) -> Dict:
        """
        Trade paraméterek validálása
        
        Args:
            entry_price: Belépési ár
            stop_loss: Stop loss ár
            take_profit: Take profit ár
            position_size: Pozíció méret
            
        Returns:
            dict: Validációs eredmény
        """
        try:
            issues = []
            
            # Basic validations
            if stop_loss >= entry_price:
                issues.append("Stop loss must be below entry price")
            
            if take_profit <= entry_price:
                issues.append("Take profit must be above entry price")
            
            if position_size < 25 or position_size > 100:
                issues.append("Position size should be between $25-$100 for micro-trading")
            
            # Risk/reward validation
            potential_loss = entry_price - stop_loss
            potential_gain = take_profit - entry_price
            
            if potential_loss > 0:
                risk_reward = potential_gain / potential_loss
                if risk_reward < 0.1:  # Very poor R/R
                    issues.append("Risk/reward ratio too low (< 0.1)")
            
            # Loss percentage validation
            loss_pct = (potential_loss / entry_price) * 100
            if loss_pct > 5:  # More than 5% loss
                issues.append("Stop loss too wide (> 5%)")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'risk_reward_ratio': risk_reward if 'risk_reward' in locals() else 0,
                'max_loss_pct': loss_pct if 'loss_pct' in locals() else 0
            }
            
        except Exception as e:
            print(f"Error validating trade params: {e}")
            return {'valid': False, 'issues': [f"Validation error: {e}"]}

    def calculate_position_size_by_risk(self, account_balance: float, 
                                      risk_pct: float, entry_price: float, 
                                      stop_loss: float) -> float:
        """
        Pozíció méret számítása kockázat alapján
        
        Args:
            account_balance: Számla egyenleg
            risk_pct: Kockázat százalék (pl. 2.0 = 2%)
            entry_price: Belépési ár
            stop_loss: Stop loss ár
            
        Returns:
            float: Ajánlott pozíció méret USD-ben
        """
        try:
            # Maximum risk amount
            max_risk_amount = account_balance * (risk_pct / 100)
            
            # Price difference (risk per unit)
            price_diff = abs(entry_price - stop_loss)
            
            if price_diff == 0:
                return 50.0  # Default micro position size
            
            # Calculate position size
            volume = max_risk_amount / price_diff
            position_value = volume * entry_price
            
            # Clamp to micro-trading range
            position_value = max(25.0, min(75.0, position_value))
            
            return round(position_value, 2)
            
        except Exception as e:
            print(f"Error calculating position size by risk: {e}")
            return 50.0  # Default fallback
