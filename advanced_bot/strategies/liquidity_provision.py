import asyncio
from typing import Dict, List
import logging
from ..core.exchange_manager import ExchangeManager

class LiquidityProvider:
    def __init__(self, exchange_manager: ExchangeManager, config: Dict):
        self.exchange_manager = exchange_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_orders: Dict[str, Dict] = {}
        self.daily_profit = 0
        self.filled_orders: List[Dict] = []

    async def place_orders(self, symbol: str, exchange: str):
        """Place buy and sell orders around the current price"""
        try:
            # Get current mid price
            ticker = await self.exchange_manager.exchanges[exchange].fetch_ticker(symbol)
            mid_price = (ticker['bid'] + ticker['ask']) / 2

            # Calculate order prices
            buy_price = mid_price * (1 - self.config.LIQUIDITY_PROVISION['spread'])
            sell_price = mid_price * (1 + self.config.LIQUIDITY_PROVISION['spread'])
            
            # Place buy order
            buy_order = await self.exchange_manager.execute_trade(
                exchange, symbol, 'buy',
                self.config.LIQUIDITY_PROVISION['order_size'],
                buy_price
            )
            
            # Place sell order
            sell_order = await self.exchange_manager.execute_trade(
                exchange, symbol, 'sell',
                self.config.LIQUIDITY_PROVISION['order_size'],
                sell_price
            )

            if buy_order and sell_order:
                self.active_orders[buy_order['id']] = {
                    'order': buy_order,
                    'type': 'buy',
                    'price': buy_price
                }
                self.active_orders[sell_order['id']] = {
                    'order': sell_order,
                    'type': 'sell',
                    'price': sell_price
                }

        except Exception as e:
            self.logger.error(f"Error placing orders: {str(e)}")

    async def monitor_orders(self, symbol: str, exchange: str):
        """Monitor and manage active orders"""
        try:
            for order_id, order_info in list(self.active_orders.items()):
                order = await self.exchange_manager.exchanges[exchange].fetch_order(order_id, symbol)
                
                if order['status'] == 'filled':
                    # Order was filled, place opposite order for profit
                    fill_price = float(order['price'])
                    
                    if order_info['type'] == 'buy':
                        # Place sell order higher
                        sell_price = fill_price * (1 + self.config.LIQUIDITY_PROVISION['spread'] * 2)
                        new_order = await self.exchange_manager.execute_trade(
                            exchange, symbol, 'sell',
                            float(order['filled']),
                            sell_price
                        )
                    else:
                        # Place buy order lower
                        buy_price = fill_price * (1 - self.config.LIQUIDITY_PROVISION['spread'] * 2)
                        new_order = await self.exchange_manager.execute_trade(
                            exchange, symbol, 'buy',
                            float(order['filled']),
                            buy_price
                        )

                    if new_order:
                        self.active_orders[new_order['id']] = {
                            'order': new_order,
                            'type': 'sell' if order_info['type'] == 'buy' else 'buy',
                            'price': sell_price if order_info['type'] == 'buy' else buy_price,
                            'paired_fill': order
                        }

                    # Calculate and record profit if this was a closing trade
                    if 'paired_fill' in order_info:
                        profit = self.calculate_profit(order_info['paired_fill'], order)
                        self.daily_profit += profit
                        self.filled_orders.append({
                            'open_order': order_info['paired_fill'],
                            'close_order': order,
                            'profit': profit
                        })

                    del self.active_orders[order_id]

        except Exception as e:
            self.logger.error(f"Error monitoring orders: {str(e)}")

    def calculate_profit(self, open_order: Dict, close_order: Dict) -> float:
        """Calculate profit from a pair of orders"""
        if open_order['side'] == 'buy':
            return (float(close_order['price']) - float(open_order['price'])) * float(open_order['filled'])
        else:
            return (float(open_order['price']) - float(close_order['price'])) * float(open_order['filled'])

    async def run(self, symbol: str, exchange: str):
        """Main loop for liquidity provision strategy"""
        while True:
            try:
                # Ensure we don't exceed max open orders
                if len(self.active_orders) < self.config.LIQUIDITY_PROVISION['max_open_orders']:
                    await self.place_orders(symbol, exchange)

                # Monitor existing orders
                await self.monitor_orders(symbol, exchange)

                # Small delay to prevent API rate limits
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in liquidity provision main loop: {str(e)}")
                await asyncio.sleep(1)

    def get_statistics(self) -> Dict:
        """Get current trading statistics"""
        return {
            'daily_profit': self.daily_profit,
            'active_orders': len(self.active_orders),
            'total_filled_orders': len(self.filled_orders),
            'success_rate': sum(1 for t in self.filled_orders if t['profit'] > 0) / max(len(self.filled_orders), 1)
        }
