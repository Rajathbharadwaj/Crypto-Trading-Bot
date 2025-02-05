import asyncio
from typing import Dict, List
import logging
from datetime import datetime, timedelta

class RiskManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.position_limits = {}
        self.daily_losses = {}
        self.trade_history = []
        self.last_reset = datetime.now()

    def can_open_position(self, strategy_name: str, size: float, price: float) -> bool:
        """Check if a new position can be opened based on risk parameters"""
        # Check strategy-specific limits
        if strategy_name not in self.position_limits:
            self.position_limits[strategy_name] = 0
        
        new_exposure = self.position_limits[strategy_name] + (size * price)
        
        # Check individual strategy limits
        if new_exposure > self.config.STRATEGY_LIMITS.get(strategy_name, float('inf')):
            self.logger.warning(f"{strategy_name} position limit reached")
            return False
            
        # Check total exposure
        total_exposure = sum(self.position_limits.values()) + (size * price)
        if total_exposure > self.config.MAX_TOTAL_EXPOSURE:
            self.logger.warning("Total exposure limit reached")
            return False
            
        # Check daily loss limit
        if self.daily_losses.get(strategy_name, 0) > self.config.MAX_DAILY_LOSS:
            self.logger.warning(f"{strategy_name} daily loss limit reached")
            return False
            
        return True

    def update_position(self, strategy_name: str, size: float, price: float):
        """Update position tracking for a strategy"""
        if strategy_name not in self.position_limits:
            self.position_limits[strategy_name] = 0
        self.position_limits[strategy_name] += size * price

    def record_trade(self, strategy_name: str, profit: float, trade_data: Dict):
        """Record a completed trade and update risk metrics"""
        # Update daily loss tracking
        if strategy_name not in self.daily_losses:
            self.daily_losses[strategy_name] = 0
        
        self.daily_losses[strategy_name] += profit if profit < 0 else 0
        
        # Record trade
        self.trade_history.append({
            'strategy': strategy_name,
            'profit': profit,
            'timestamp': datetime.now(),
            'data': trade_data
        })
        
        # Clean old trade history
        self.clean_old_trades()

    def get_risk_metrics(self) -> Dict:
        """Calculate current risk metrics"""
        return {
            'total_exposure': sum(self.position_limits.values()),
            'exposure_by_strategy': self.position_limits.copy(),
            'daily_losses': self.daily_losses.copy(),
            'trade_count': len(self.trade_history),
            'win_rate': self.calculate_win_rate(),
            'risk_score': self.calculate_risk_score()
        }

    def calculate_win_rate(self) -> float:
        """Calculate recent win rate"""
        if not self.trade_history:
            return 0.0
            
        recent_trades = [t for t in self.trade_history 
                        if t['timestamp'] > datetime.now() - timedelta(days=1)]
        if not recent_trades:
            return 0.0
            
        winning_trades = sum(1 for t in recent_trades if t['profit'] > 0)
        return winning_trades / len(recent_trades)

    def calculate_risk_score(self) -> float:
        """Calculate overall risk score (0-100)"""
        metrics = [
            # Exposure risk (40%)
            (sum(self.position_limits.values()) / self.config.MAX_TOTAL_EXPOSURE) * 40,
            
            # Loss risk (30%)
            (sum(self.daily_losses.values()) / self.config.MAX_DAILY_LOSS) * 30,
            
            # Win rate risk (30%)
            ((1 - self.calculate_win_rate()) * 30)
        ]
        
        return sum(metrics)

    def clean_old_trades(self):
        """Remove trades older than 7 days"""
        cutoff = datetime.now() - timedelta(days=7)
        self.trade_history = [t for t in self.trade_history 
                            if t['timestamp'] > cutoff]

    def reset_daily_metrics(self):
        """Reset daily tracking metrics"""
        self.daily_losses = {}
        self.last_reset = datetime.now()

    async def run_risk_monitoring(self):
        """Continuous risk monitoring loop"""
        while True:
            try:
                # Reset daily metrics if needed
                if datetime.now() - self.last_reset > timedelta(days=1):
                    self.reset_daily_metrics()
                
                # Calculate current risk score
                risk_score = self.calculate_risk_score()
                
                # Log warnings for high risk
                if risk_score > 80:
                    self.logger.warning(f"High risk score: {risk_score}")
                    
                # Sleep for a short time
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in risk monitoring: {str(e)}")
                await asyncio.sleep(5)
