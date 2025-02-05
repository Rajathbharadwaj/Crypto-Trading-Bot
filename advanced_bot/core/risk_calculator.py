import logging
from typing import Dict
from datetime import datetime, timedelta

class RiskCalculator:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.starting_balance = 500
        self.current_balance = self.starting_balance
        self.max_risk_per_trade = 25  # $25 max risk per trade
        self.daily_loss_limit = 100  # $100 daily loss limit
        self.max_leverage = 20
        self.trades_today = []
        self.last_reset = datetime.now()

    def can_take_trade(self, symbol: str, risk_amount: float, setup_quality: float) -> bool:
        """Determine if a new trade can be taken based on risk parameters"""
        try:
            # Check if we've hit daily loss limit
            daily_pnl = sum(t['profit'] for t in self.trades_today)
            if daily_pnl <= -self.daily_loss_limit:
                self.logger.warning("Daily loss limit reached")
                return False

            # Check if risk per trade is too high
            if risk_amount > self.max_risk_per_trade:
                self.logger.warning(f"Risk amount ${risk_amount} exceeds max risk per trade ${self.max_risk_per_trade}")
                return False

            # Calculate current drawdown
            drawdown = (self.starting_balance - self.current_balance) / self.starting_balance
            if drawdown > 0.15:  # 15% max drawdown
                self.logger.warning(f"Maximum drawdown reached: {drawdown*100:.1f}%")
                return False

            # Adjust risk based on recent performance
            win_rate = self.calculate_win_rate()
            if win_rate < 0.4:  # Below 40% win rate
                if setup_quality < 0.9:  # Only take very high quality setups
                    self.logger.warning("Win rate too low for medium quality setups")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error in can_take_trade: {str(e)}")
            return False

    def calculate_position_size(self, entry_price: float, stop_loss: float, symbol: str) -> float:
        """Calculate safe position size based on risk parameters"""
        try:
            # Calculate base position size from risk
            price_distance = abs(entry_price - stop_loss)
            base_position = self.max_risk_per_trade / price_distance

            # Apply leverage
            leveraged_position = base_position * self.max_leverage

            # Additional safety checks
            account_risk = (self.max_risk_per_trade / self.current_balance) * 100
            if account_risk > 5:  # Don't risk more than 5% per trade
                leveraged_position *= (5 / account_risk)

            # Round to appropriate decimals
            if symbol == 'BTCUSD':
                return round(leveraged_position, 3)  # 0.001 BTC precision
            else:  # XAUUSD
                return round(leveraged_position, 2)  # 0.01 lot precision

        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0

    def record_trade(self, trade_data: Dict):
        """Record a completed trade"""
        try:
            self.trades_today.append(trade_data)
            self.current_balance += trade_data['profit']

            # Clean old trades
            self.clean_old_trades()

        except Exception as e:
            self.logger.error(f"Error recording trade: {str(e)}")

    def calculate_win_rate(self) -> float:
        """Calculate recent win rate"""
        try:
            if not self.trades_today:
                return 0.5  # Default to neutral if no trades

            winning_trades = sum(1 for t in self.trades_today if t['profit'] > 0)
            return winning_trades / len(self.trades_today)

        except Exception as e:
            self.logger.error(f"Error calculating win rate: {str(e)}")
            return 0

    def clean_old_trades(self):
        """Remove trades older than 24 hours"""
        try:
            cutoff = datetime.now() - timedelta(days=1)
            self.trades_today = [t for t in self.trades_today 
                               if t['timestamp'] > cutoff]

            # Reset daily stats if needed
            if datetime.now() - self.last_reset > timedelta(days=1):
                self.last_reset = datetime.now()
                self.trades_today = []

        except Exception as e:
            self.logger.error(f"Error cleaning old trades: {str(e)}")

    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        try:
            daily_pnl = sum(t['profit'] for t in self.trades_today)
            win_rate = self.calculate_win_rate()
            drawdown = (self.starting_balance - self.current_balance) / self.starting_balance

            return {
                'current_balance': self.current_balance,
                'daily_pnl': daily_pnl,
                'win_rate': win_rate,
                'drawdown': drawdown,
                'trades_today': len(self.trades_today)
            }

        except Exception as e:
            self.logger.error(f"Error getting risk metrics: {str(e)}")
            return {}
