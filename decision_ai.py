# strategy/decision_ai.py

class DecisionEngine:
    def __init__(self):
        self.weights = {
            "rsi": 0.2,
            "ema": 0.3,
            "bollinger": 0.2,
            "volatility": 0.1,
            "history_success": 0.2
        }

    def score_trade(self, indicators, history_stats):
        score = 0

        # RSI scoring: adds to score if RSI is within a neutral range (not overbought/oversold)
        # If RSI is between 30 and 70, it contributes fully to the score based on its weight.
        # Otherwise (RSI < 30 or RSI > 70), it contributes half of its potential score.
        if 30 < indicators.get("rsi", 50) < 70:  # Added .get() for safety
            score += self.weights["rsi"] * 1
        else:
            score += self.weights["rsi"] * 0.5

        # EMA cross scoring: adds to score if an EMA cross is detected
        # Assumes indicators["ema_cross"] is a boolean (True if cross occurred).
        if indicators.get("ema_cross", False): # Added .get() for safety
            score += self.weights["ema"] * 1

        # Bollinger Band breakout scoring: adds to score if a breakout is detected
        # Assumes indicators["bollinger_breakout"] is a boolean.
        if indicators.get("bollinger_breakout", False): # Added .get() for safety
            score += self.weights["bollinger"] * 1

        # Volatility (ATR) scoring: adds to score if ATR (Average True Range) is above a certain threshold
        # This suggests sufficient market movement for a potential trade.
        if indicators.get("atr", 0) > 0.005: # Added .get() for safety
            score += self.weights["volatility"] * 1

        # Historical success scoring: adds to score if historical win rate is good
        # Assumes history_stats["win_rate"] provides the win rate (e.g., 0.6 for 60%).
        if history_stats.get("win_rate", 0) > 0.6: # Added .get() for safety
            score += self.weights["history_success"] * 1

        return score

    def make_decision(self, indicators, history_stats, threshold=0.7):
        """
        Makes a trading decision (buy, sell, or hold) based on the calculated score.

        Args:
            indicators (dict): A dictionary containing various technical indicator values.
                               Expected keys: "rsi", "ema_cross", "bollinger_breakout", "atr".
            history_stats (dict): A dictionary containing historical performance statistics.
                                  Expected key: "win_rate".
            threshold (float): The score threshold above which a "buy" decision is made.

        Returns:
            str: The trading decision ("buy", "sell", or "hold").
        """
        # Calculate the overall score for the potential trade
        score = self.score_trade(indicators, history_stats)

        # Decision logic based on the score
        if score >= threshold:
            return "buy"  # If score meets or exceeds the buy threshold
        elif score <= 0.4: # Arbitrary threshold for sell, can be adjusted
            return "sell" # If score is below a certain sell threshold
        else:
            return "hold" # Otherwise, hold the current position or do nothing

