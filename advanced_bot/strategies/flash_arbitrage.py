import asyncio
from typing import Dict, List
import logging
from ..core.exchange_manager import ExchangeManager

class FlashArbitrage:
    def __init__(self, exchange_manager: ExchangeManager, config: Dict):
        self.exchange_manager = exchange_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_trades: List[Dict] = []
        self.daily_profit = 0

    async def find_opportunities(self, symbol: str) -> List[Dict]:
        """Find arbitrage opportunities across exchanges"""
        differences = await self.exchange_manager.get_price_differences(symbol)
        return [d for d in differences if d['spread'] > self.config.FLASH_ARBITRAGE['min_spread']]

    async def execute_arbitrage(self, opportunity: Dict, symbol: str):
        """Execute arbitrage trades"""
        try:
            amount = min(
                self.config.FLASH_ARBITRAGE['max_position'],
                opportunity['sell_price'] * self.config.MAX_POSITION_SIZE
            )

            # Execute buy order
            buy_order = await self.exchange_manager.execute_trade(
                opportunity['buy_exchange'],
                symbol,
                'buy',
                amount,
                opportunity['buy_price']
            )

            if not buy_order:
                self.logger.error("Buy order failed")
                return

            # Execute sell order
            sell_order = await self.exchange_manager.execute_trade(
                opportunity['sell_exchange'],
                symbol,
                'sell',
                amount,
                opportunity['sell_price']
            )

            if not sell_order:
                # Cancel buy order if sell fails
                await self.exchange_manager.exchanges[opportunity['buy_exchange']].cancel_order(
                    buy_order['id'], symbol
                )
                self.logger.error("Sell order failed, cancelled buy order")
                return

            # Calculate profit
            profit = (opportunity['sell_price'] - opportunity['buy_price']) * amount
            self.daily_profit += profit

            self.logger.info(f"Arbitrage executed: Profit ${profit:.2f}")
            
            # Add to active trades
            self.active_trades.append({
                'buy_order': buy_order,
                'sell_order': sell_order,
                'profit': profit,
                'timestamp': buy_order['timestamp']
            })

        except Exception as e:
            self.logger.error(f"Error executing arbitrage: {str(e)}")

    async def monitor_and_execute(self, symbol: str):
        """Main loop for monitoring and executing arbitrage opportunities"""
        while True:
            try:
                opportunities = await self.find_opportunities(symbol)
                
                for opportunity in opportunities:
                    if self.daily_profit < self.config.MAX_DAILY_LOSS:  # Still within daily limit
                        asyncio.create_task(self.execute_arbitrage(opportunity, symbol))
                    
                await asyncio.sleep(0.1)  # Small delay to prevent API rate limits
                
            except Exception as e:
                self.logger.error(f"Error in monitor_and_execute: {str(e)}")
                await asyncio.sleep(1)  # Longer delay on error

    def get_statistics(self) -> Dict:
        """Get current trading statistics"""
        return {
            'daily_profit': self.daily_profit,
            'active_trades': len(self.active_trades),
            'total_trades': len(self.active_trades),
            'success_rate': sum(1 for t in self.active_trades if t['profit'] > 0) / max(len(self.active_trades), 1)
        }
