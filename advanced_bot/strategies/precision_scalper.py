import asyncio
from typing import Dict, List
import logging
import numpy as np
from datetime import datetime, timedelta
from ..core.exchange_manager import ExchangeManager

class PrecisionScalper:
    def __init__(self, exchange_manager: ExchangeManager, config: Dict):
        self.exchange_manager = exchange_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_trades: Dict[str, List[Dict]] = {'BTCUSD': [], 'XAUUSD': []}
        self.daily_stats = {
            'BTCUSD': {'profit': 0, 'trades': 0, 'wins': 0},
            'XAUUSD': {'profit': 0, 'trades': 0, 'wins': 0}
        }
        self.candle_history = {
            'BTCUSD': [],
            'XAUUSD': []
        }

    async def analyze_setup(self, symbol: str) -> Dict:
        """Analyze trading setup for a symbol"""
        try:
            # Get recent candles
            candles = await self.exchange_manager.get_recent_candles(symbol, '1m', 20)
            if not candles:
                return None

            # Calculate key metrics
            close_prices = np.array([c['close'] for c in candles])
            volumes = np.array([c['volume'] for c in candles])
            
            # Calculate EMAs
            ema9 = self.calculate_ema(close_prices, 9)
            ema20 = self.calculate_ema(close_prices, 20)
            
            # Calculate RSI
            rsi = self.calculate_rsi(close_prices, 14)
            
            # Calculate volume profile
            avg_volume = np.mean(volumes[-5:])
            current_volume = volumes[-1]
            
            # Detect potential setup
            setup = {
                'symbol': symbol,
                'current_price': close_prices[-1],
                'ema_trend': 'up' if ema9[-1] > ema20[-1] else 'down',
                'rsi': rsi[-1],
                'volume_ratio': current_volume / avg_volume,
                'momentum': self.calculate_momentum(close_prices),
                'volatility': self.calculate_volatility(close_prices)
            }

            return setup

        except Exception as e:
            self.logger.error(f"Error analyzing setup for {symbol}: {str(e)}")
            return None

    def calculate_ema(self, prices: np.array, period: int) -> np.array:
        """Calculate Exponential Moving Average"""
        multiplier = 2 / (period + 1)
        ema = [prices[0]]
        
        for price in prices[1:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])
            
        return np.array(ema)

    def calculate_rsi(self, prices: np.array, period: int) -> np.array:
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)

        return rsi

    def calculate_momentum(self, prices: np.array) -> float:
        """Calculate price momentum"""
        return (prices[-1] - prices[-5]) / prices[-5] * 100

    def calculate_volatility(self, prices: np.array) -> float:
        """Calculate recent price volatility"""
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) * np.sqrt(len(returns))

    async def find_entry_opportunity(self, symbol: str) -> Dict:
        """Find potential entry opportunity"""
        setup = await self.analyze_setup(symbol)
        if not setup:
            return None

        # Entry rules based on symbol
        if symbol == 'BTCUSD':
            return self.analyze_btc_setup(setup)
        else:  # XAUUSD
            return self.analyze_gold_setup(setup)

    def analyze_btc_setup(self, setup: Dict) -> Dict:
        """Analyze BTC-specific setup"""
        # Strong trend confirmation
        if setup['ema_trend'] == 'up' and setup['momentum'] > 0.2:
            if 40 <= setup['rsi'] <= 60:  # Not overbought/oversold
                if setup['volume_ratio'] > 1.5:  # Strong volume
                    return {
                        'type': 'long',
                        'entry_price': setup['current_price'],
                        'stop_loss': setup['current_price'] * 0.9975,  # 0.25% stop
                        'take_profit': setup['current_price'] * 1.005,  # 0.5% target
                        'confidence': min(setup['volume_ratio'] * 0.5, 0.9)
                    }
                    
        elif setup['ema_trend'] == 'down' and setup['momentum'] < -0.2:
            if 40 <= setup['rsi'] <= 60:
                if setup['volume_ratio'] > 1.5:
                    return {
                        'type': 'short',
                        'entry_price': setup['current_price'],
                        'stop_loss': setup['current_price'] * 1.0025,
                        'take_profit': setup['current_price'] * 0.995,
                        'confidence': min(setup['volume_ratio'] * 0.5, 0.9)
                    }

        return None

    def analyze_gold_setup(self, setup: Dict) -> Dict:
        """Analyze Gold-specific setup"""
        # Gold tends to move in smaller increments
        if setup['ema_trend'] == 'up' and setup['momentum'] > 0.1:
            if 45 <= setup['rsi'] <= 55:  # Tighter RSI range for gold
                if setup['volume_ratio'] > 1.3:  # Lower volume requirement
                    return {
                        'type': 'long',
                        'entry_price': setup['current_price'],
                        'stop_loss': setup['current_price'] * 0.9985,  # 0.15% stop
                        'take_profit': setup['current_price'] * 1.003,  # 0.3% target
                        'confidence': min(setup['volume_ratio'] * 0.6, 0.9)
                    }
                    
        elif setup['ema_trend'] == 'down' and setup['momentum'] < -0.1:
            if 45 <= setup['rsi'] <= 55:
                if setup['volume_ratio'] > 1.3:
                    return {
                        'type': 'short',
                        'entry_price': setup['current_price'],
                        'stop_loss': setup['current_price'] * 1.0015,
                        'take_profit': setup['current_price'] * 0.997,
                        'confidence': min(setup['volume_ratio'] * 0.6, 0.9)
                    }

        return None

    async def execute_trade(self, symbol: str, setup: Dict):
        """Execute trade based on setup"""
        try:
            # Calculate position size based on risk
            account_balance = 500  # Starting capital
            risk_per_trade = 25  # $25 risk per trade
            
            # Calculate position size based on stop loss distance
            stop_distance = abs(setup['entry_price'] - setup['stop_loss'])
            position_size = risk_per_trade / stop_distance
            
            # Apply leverage (20x)
            position_size *= 20
            
            # Execute entry
            entry_order = await self.exchange_manager.execute_trade(
                symbol,
                setup['type'],
                position_size,
                setup['entry_price']
            )

            if entry_order:
                # Place stop loss and take profit orders
                stop_order = await self.exchange_manager.place_stop_loss(
                    symbol,
                    'sell' if setup['type'] == 'long' else 'buy',
                    position_size,
                    setup['stop_loss']
                )

                tp_order = await self.exchange_manager.place_take_profit(
                    symbol,
                    'sell' if setup['type'] == 'long' else 'buy',
                    position_size,
                    setup['take_profit']
                )

                # Record trade
                self.active_trades[symbol].append({
                    'entry_order': entry_order,
                    'stop_order': stop_order,
                    'tp_order': tp_order,
                    'setup': setup,
                    'position_size': position_size,
                    'entry_time': datetime.now()
                })

        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")

    async def run(self):
        """Main trading loop"""
        while True:
            try:
                for symbol in ['BTCUSD', 'XAUUSD']:
                    # Check if we can take new trades
                    if len(self.active_trades[symbol]) < 2:  # Max 2 trades per symbol
                        if self.daily_stats[symbol]['profit'] > -100:  # Not hit daily loss limit
                            opportunity = await self.find_entry_opportunity(symbol)
                            if opportunity and opportunity['confidence'] > 0.8:  # High confidence setups only
                                await self.execute_trade(symbol, opportunity)

                    # Monitor active trades
                    await self.monitor_trades(symbol)

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(5)

    async def monitor_trades(self, symbol: str):
        """Monitor and manage active trades"""
        for trade in list(self.active_trades[symbol]):
            try:
                # Check if stop loss or take profit hit
                current_price = await self.exchange_manager.get_current_price(symbol)
                
                if trade['setup']['type'] == 'long':
                    if current_price <= trade['setup']['stop_loss'] or \
                       current_price >= trade['setup']['take_profit']:
                        await self.close_trade(symbol, trade)
                else:  # short
                    if current_price >= trade['setup']['stop_loss'] or \
                       current_price <= trade['setup']['take_profit']:
                        await self.close_trade(symbol, trade)

                # Check trade duration
                if datetime.now() - trade['entry_time'] > timedelta(minutes=5):
                    # Close trade if open more than 5 minutes
                    await self.close_trade(symbol, trade)

            except Exception as e:
                self.logger.error(f"Error monitoring trade: {str(e)}")

    async def close_trade(self, symbol: str, trade: Dict):
        """Close a trade and record results"""
        try:
            close_price = await self.exchange_manager.get_current_price(symbol)
            
            # Calculate profit
            entry_price = float(trade['entry_order']['price'])
            profit = (close_price - entry_price) * trade['position_size'] if trade['setup']['type'] == 'long' \
                    else (entry_price - close_price) * trade['position_size']

            # Update statistics
            self.daily_stats[symbol]['profit'] += profit
            self.daily_stats[symbol]['trades'] += 1
            if profit > 0:
                self.daily_stats[symbol]['wins'] += 1

            # Remove from active trades
            self.active_trades[symbol].remove(trade)

            self.logger.info(f"Closed {symbol} trade: ${profit:.2f}")

        except Exception as e:
            self.logger.error(f"Error closing trade: {str(e)}")

    def get_statistics(self) -> Dict:
        """Get current trading statistics"""
        return {
            'daily_stats': self.daily_stats,
            'active_trades': {
                'BTCUSD': len(self.active_trades['BTCUSD']),
                'XAUUSD': len(self.active_trades['XAUUSD'])
            },
            'win_rates': {
                'BTCUSD': self.daily_stats['BTCUSD']['wins'] / max(self.daily_stats['BTCUSD']['trades'], 1),
                'XAUUSD': self.daily_stats['XAUUSD']['wins'] / max(self.daily_stats['XAUUSD']['trades'], 1)
            }
        }
