import asyncio
from typing import Dict, List
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .risk_manager import RiskManager

class PerformanceAnalytics:
    def __init__(self, config: Dict, risk_manager: RiskManager):
        self.config = config
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(__name__)
        self.trade_history: List[Dict] = []
        self.hourly_stats: List[Dict] = []
        self.strategy_metrics = {}

    def record_trade(self, trade_data: Dict):
        """Record a completed trade"""
        self.trade_history.append({
            **trade_data,
            'timestamp': datetime.now()
        })
        self.update_strategy_metrics(trade_data)

    def update_strategy_metrics(self, trade_data: Dict):
        """Update metrics for a specific strategy"""
        strategy = trade_data['strategy']
        if strategy not in self.strategy_metrics:
            self.strategy_metrics[strategy] = {
                'total_profit': 0,
                'win_count': 0,
                'loss_count': 0,
                'total_trades': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }

        metrics = self.strategy_metrics[strategy]
        profit = trade_data['profit']

        metrics['total_profit'] += profit
        metrics['total_trades'] += 1

        if profit > 0:
            metrics['win_count'] += 1
            metrics['largest_win'] = max(metrics['largest_win'], profit)
            metrics['avg_profit'] = (metrics['avg_profit'] * (metrics['win_count'] - 1) + profit) / metrics['win_count']
        else:
            metrics['loss_count'] += 1
            metrics['largest_loss'] = min(metrics['largest_loss'], profit)
            metrics['avg_loss'] = (metrics['avg_loss'] * (metrics['loss_count'] - 1) + profit) / metrics['loss_count']

        if metrics['avg_loss'] != 0:
            metrics['profit_factor'] = abs(metrics['avg_profit'] * metrics['win_count'] / 
                                        (metrics['avg_loss'] * metrics['loss_count']))

    def calculate_hourly_stats(self):
        """Calculate performance statistics for the last hour"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Filter recent trades
        recent_trades = [t for t in self.trade_history if t['timestamp'] > hour_ago]
        
        if not recent_trades:
            return None
            
        stats = {
            'timestamp': now,
            'total_trades': len(recent_trades),
            'total_profit': sum(t['profit'] for t in recent_trades),
            'win_rate': sum(1 for t in recent_trades if t['profit'] > 0) / len(recent_trades),
            'avg_profit_per_trade': sum(t['profit'] for t in recent_trades) / len(recent_trades),
            'largest_profit': max(t['profit'] for t in recent_trades),
            'largest_loss': min(t['profit'] for t in recent_trades),
            'profit_by_strategy': self.get_profit_by_strategy(recent_trades)
        }
        
        self.hourly_stats.append(stats)
        return stats

    def get_profit_by_strategy(self, trades: List[Dict]) -> Dict:
        """Calculate profit for each strategy"""
        profits = {}
        for trade in trades:
            strategy = trade['strategy']
            if strategy not in profits:
                profits[strategy] = 0
            profits[strategy] += trade['profit']
        return profits

    def calculate_drawdown(self) -> Dict:
        """Calculate maximum drawdown and current drawdown"""
        if not self.trade_history:
            return {'max_drawdown': 0, 'current_drawdown': 0}

        # Calculate equity curve
        equity = pd.Series([t['profit'] for t in self.trade_history]).cumsum()
        
        # Calculate running maximum
        running_max = equity.expanding().max()
        
        # Calculate drawdowns
        drawdowns = equity - running_max
        max_drawdown = drawdowns.min()
        current_drawdown = drawdowns.iloc[-1]
        
        return {
            'max_drawdown': abs(max_drawdown),
            'current_drawdown': abs(current_drawdown)
        }

    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of trading performance"""
        if not self.trade_history:
            return 0

        # Calculate daily returns
        profits = pd.Series([t['profit'] for t in self.trade_history])
        daily_returns = profits.resample('D').sum()
        
        if len(daily_returns) < 2:
            return 0
            
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        return sharpe

    def get_strategy_performance(self, strategy: str) -> Dict:
        """Get detailed performance metrics for a strategy"""
        if strategy not in self.strategy_metrics:
            return None
            
        metrics = self.strategy_metrics[strategy]
        total_trades = metrics['total_trades']
        
        if total_trades == 0:
            return None
            
        return {
            'total_profit': metrics['total_profit'],
            'win_rate': metrics['win_count'] / total_trades if total_trades > 0 else 0,
            'profit_factor': metrics['profit_factor'],
            'avg_profit': metrics['avg_profit'],
            'avg_loss': metrics['avg_loss'],
            'largest_win': metrics['largest_win'],
            'largest_loss': metrics['largest_loss'],
            'total_trades': total_trades
        }

    def get_performance_summary(self) -> Dict:
        """Get overall performance summary"""
        if not self.trade_history:
            return None

        drawdown = self.calculate_drawdown()
        
        return {
            'total_profit': sum(t['profit'] for t in self.trade_history),
            'total_trades': len(self.trade_history),
            'win_rate': sum(1 for t in self.trade_history if t['profit'] > 0) / len(self.trade_history),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'max_drawdown': drawdown['max_drawdown'],
            'current_drawdown': drawdown['current_drawdown'],
            'strategy_metrics': self.strategy_metrics,
            'risk_metrics': self.risk_manager.get_risk_metrics()
        }

    async def run_analytics(self):
        """Continuous analytics calculation loop"""
        while True:
            try:
                # Calculate hourly stats
                self.calculate_hourly_stats()
                
                # Clean old trade history (keep last 7 days)
                cutoff = datetime.now() - timedelta(days=7)
                self.trade_history = [t for t in self.trade_history 
                                    if t['timestamp'] > cutoff]
                
                # Clean old hourly stats (keep last 24 hours)
                cutoff = datetime.now() - timedelta(hours=24)
                self.hourly_stats = [s for s in self.hourly_stats 
                                   if s['timestamp'] > cutoff]
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in analytics: {str(e)}")
                await asyncio.sleep(5)
