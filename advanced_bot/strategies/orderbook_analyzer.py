import asyncio
from typing import Dict, List, Tuple
import logging
import numpy as np
from ..core.exchange_manager import ExchangeManager

class OrderBookAnalyzer:
    def __init__(self, exchange_manager: ExchangeManager, config: Dict):
        self.exchange_manager = exchange_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_trades: List[Dict] = []
        self.daily_profit = 0
        self.orderbook_history: List[Dict] = []
        self.detected_walls: Dict[str, Dict] = {}

    async def analyze_orderbook(self, symbol: str, exchange: str) -> Dict:
        """Analyze orderbook for trading opportunities"""
        try:
            orderbook = await self.exchange_manager.get_orderbook(exchange, symbol)
            if not orderbook:
                return None

            analysis = {
                'buy_walls': self.detect_walls(orderbook['bids']),
                'sell_walls': self.detect_walls(orderbook['asks']),
                'imbalance': self.calculate_imbalance(orderbook),
                'spread': orderbook['asks'][0][0] - orderbook['bids'][0][0],
                'timestamp': orderbook.get('timestamp', None)
            }

            self.orderbook_history.append(analysis)
            if len(self.orderbook_history) > 100:  # Keep last 100 analyses
                self.orderbook_history.pop(0)

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing orderbook: {str(e)}")
            return None

    def detect_walls(self, orders: List[List[float]], min_wall_size: float = None) -> List[Dict]:
        """Detect significant order walls"""
        if min_wall_size is None:
            min_wall_size = self.config.ORDER_BOOK['min_wall_size']

        walls = []
        cumulative_size = 0
        current_price = None
        current_size = 0

        for price, size in orders:
            if current_price is None:
                current_price = price
                current_size = size
            elif abs(price - current_price) < 0.0001:  # Orders at same price
                current_size += size
            else:
                if current_size >= min_wall_size:
                    walls.append({
                        'price': current_price,
                        'size': current_size
                    })
                current_price = price
                current_size = size
            cumulative_size += size

        # Check last wall
        if current_size >= min_wall_size:
            walls.append({
                'price': current_price,
                'size': current_size
            })

        return walls

    def calculate_imbalance(self, orderbook: Dict) -> float:
        """Calculate buy/sell imbalance in orderbook"""
        buy_volume = sum(size for _, size in orderbook['bids'][:10])  # Top 10 levels
        sell_volume = sum(size for _, size in orderbook['asks'][:10])
        
        if sell_volume == 0:
            return float('inf')
        return buy_volume / sell_volume

    async def find_trading_opportunities(self, symbol: str, exchange: str) -> List[Dict]:
        """Find trading opportunities based on orderbook analysis"""
        analysis = await self.analyze_orderbook(symbol, exchange)
        if not analysis:
            return []

        opportunities = []

        # Check for strong imbalances
        if analysis['imbalance'] > 3.0:  # Strong buy pressure
            opportunities.append({
                'type': 'imbalance_long',
                'strength': analysis['imbalance'],
                'entry_price': analysis['buy_walls'][0]['price'] if analysis['buy_walls'] else None
            })
        elif analysis['imbalance'] < 0.33:  # Strong sell pressure
            opportunities.append({
                'type': 'imbalance_short',
                'strength': 1/analysis['imbalance'],
                'entry_price': analysis['sell_walls'][0]['price'] if analysis['sell_walls'] else None
            })

        # Check for wall breakouts
        for wall in analysis['buy_walls']:
            if self.is_wall_breaking(wall, 'buy'):
                opportunities.append({
                    'type': 'wall_breakout_long',
                    'wall_price': wall['price'],
                    'wall_size': wall['size']
                })

        for wall in analysis['sell_walls']:
            if self.is_wall_breaking(wall, 'sell'):
                opportunities.append({
                    'type': 'wall_breakout_short',
                    'wall_price': wall['price'],
                    'wall_size': wall['size']
                })

        return opportunities

    def is_wall_breaking(self, wall: Dict, direction: str) -> bool:
        """Determine if an order wall is about to break"""
        if len(self.orderbook_history) < 10:
            return False

        # Get wall sizes over time at this price level
        wall_sizes = []
        price = wall['price']
        for history in self.orderbook_history[-10:]:
            walls = history['buy_walls'] if direction == 'buy' else history['sell_walls']
            matching_walls = [w for w in walls if abs(w['price'] - price) < 0.0001]
            wall_sizes.append(matching_walls[0]['size'] if matching_walls else 0)

        # Wall is breaking if size is consistently decreasing
        if len(wall_sizes) >= 3:
            recent_trend = np.polyfit(range(len(wall_sizes)), wall_sizes, 1)[0]
            return recent_trend < -self.config.ORDER_BOOK['wall_break_threshold']

        return False

    async def execute_wall_trade(self, opportunity: Dict, symbol: str, exchange: str):
        """Execute trade based on wall analysis"""
        try:
            direction = 'buy' if 'long' in opportunity['type'] else 'sell'
            
            # Calculate position size based on wall size
            position_size = min(
                opportunity.get('wall_size', 0) * 0.1,  # 10% of wall size
                self.config.ORDER_BOOK['max_position']
            )

            # Place the trade
            entry_order = await self.exchange_manager.execute_trade(
                exchange,
                symbol,
                direction,
                position_size,
                opportunity.get('entry_price')
            )

            if entry_order:
                # Set take profit and stop loss
                if direction == 'buy':
                    take_profit = opportunity['wall_price'] * (1 + self.config.ORDER_BOOK['profit_target'])
                    stop_loss = opportunity['wall_price'] * (1 - self.config.ORDER_BOOK['stop_loss'])
                else:
                    take_profit = opportunity['wall_price'] * (1 - self.config.ORDER_BOOK['profit_target'])
                    stop_loss = opportunity['wall_price'] * (1 + self.config.ORDER_BOOK['stop_loss'])

                self.active_trades.append({
                    'entry_order': entry_order,
                    'direction': direction,
                    'entry_price': opportunity['wall_price'],
                    'take_profit': take_profit,
                    'stop_loss': stop_loss,
                    'size': position_size,
                    'exchange': exchange
                })

        except Exception as e:
            self.logger.error(f"Error executing wall trade: {str(e)}")

    async def run(self, symbol: str, exchange: str):
        """Main loop for orderbook analysis strategy"""
        while True:
            try:
                opportunities = await self.find_trading_opportunities(symbol, exchange)
                
                for opportunity in opportunities:
                    if len(self.active_trades) < self.config.ORDER_BOOK['max_trades']:
                        await self.execute_wall_trade(opportunity, symbol, exchange)

                # Monitor active trades
                await self.monitor_trades(symbol, exchange)
                
                await asyncio.sleep(0.1)  # Small delay to prevent API rate limits

            except Exception as e:
                self.logger.error(f"Error in orderbook analysis main loop: {str(e)}")
                await asyncio.sleep(1)

    async def monitor_trades(self, symbol: str, exchange: str):
        """Monitor and manage active trades"""
        for trade in list(self.active_trades):
            try:
                ticker = await self.exchange_manager.exchanges[exchange].fetch_ticker(symbol)
                current_price = ticker['last']

                # Check for take profit or stop loss
                if (trade['direction'] == 'buy' and 
                    (current_price >= trade['take_profit'] or current_price <= trade['stop_loss'])) or \
                   (trade['direction'] == 'sell' and 
                    (current_price <= trade['take_profit'] or current_price >= trade['stop_loss'])):

                    # Close the position
                    close_order = await self.exchange_manager.execute_trade(
                        exchange,
                        symbol,
                        'sell' if trade['direction'] == 'buy' else 'buy',
                        trade['size']
                    )

                    if close_order:
                        profit = self.calculate_profit(trade, close_order)
                        self.daily_profit += profit
                        self.active_trades.remove(trade)
                        
                        self.logger.info(f"Closed wall trade. Profit: ${profit:.2f}")

            except Exception as e:
                self.logger.error(f"Error monitoring trade: {str(e)}")

    def calculate_profit(self, trade: Dict, close_order: Dict) -> float:
        """Calculate profit from a trade"""
        entry_price = float(trade['entry_price'])
        close_price = float(close_order['price'])
        size = float(trade['size'])

        if trade['direction'] == 'buy':
            return (close_price - entry_price) * size
        else:
            return (entry_price - close_price) * size

    def get_statistics(self) -> Dict:
        """Get current trading statistics"""
        return {
            'daily_profit': self.daily_profit,
            'active_trades': len(self.active_trades),
            'total_trades': len(self.active_trades),
            'detected_walls': len(self.detected_walls)
        }
