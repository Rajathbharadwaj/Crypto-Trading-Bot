import asyncio
from typing import Dict, List
import logging
from ..core.exchange_manager import ExchangeManager
from ..core.news_monitor import NewsMonitor

class NewsTrader:
    def __init__(self, exchange_manager: ExchangeManager, news_monitor: NewsMonitor, config: Dict):
        self.exchange_manager = exchange_manager
        self.news_monitor = news_monitor
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_trades: List[Dict] = []
        self.daily_profit = 0

        # Register callback for news events
        self.news_monitor.register_callback(self.on_news_event)

    async def on_news_event(self, news_data: Dict):
        """Handle incoming news events"""
        try:
            # Analyze news sentiment and importance
            if self.is_significant_news(news_data):
                # Determine trade direction based on news
                direction = self.analyze_news_sentiment(news_data)
                
                if direction:
                    await self.execute_news_trade(direction, news_data)

        except Exception as e:
            self.logger.error(f"Error processing news event: {str(e)}")

    def is_significant_news(self, news_data: Dict) -> bool:
        """Determine if news is significant enough to trade"""
        keywords = self.config.NEWS_TRADING['keywords']
        
        # Check for high-impact keywords
        high_impact = any(
            keyword in news_data['content'].lower() 
            for keyword in ['listing', 'partnership', 'acquisition']
        )

        # Check source credibility
        credible_source = news_data['source'] in ['binance', 'coinbase', 'twitter']
        
        return high_impact and credible_source

    def analyze_news_sentiment(self, news_data: Dict) -> str:
        """Analyze news sentiment to determine trade direction"""
        positive_keywords = ['listing', 'partnership', 'upgrade', 'launch']
        negative_keywords = ['hack', 'security', 'vulnerability', 'delay']

        content = news_data['content'].lower()
        
        positive_score = sum(1 for word in positive_keywords if word in content)
        negative_score = sum(1 for word in negative_keywords if word in content)

        if positive_score > negative_score:
            return 'buy'
        elif negative_score > positive_score:
            return 'sell'
        return None

    async def execute_news_trade(self, direction: str, news_data: Dict):
        """Execute trade based on news"""
        try:
            # Use multiple exchanges for better fill probability
            for exchange_name in self.config.FLASH_ARBITRAGE['exchanges']:
                symbol = 'BTC/USDT'  # Default to BTC
                
                # Calculate position size
                ticker = await self.exchange_manager.exchanges[exchange_name].fetch_ticker(symbol)
                current_price = ticker['last']
                
                # Use smaller position size for news trades
                position_size = min(
                    self.config.NEWS_TRADING['max_position'],
                    (self.config.MAX_POSITION_SIZE * current_price) / len(self.config.FLASH_ARBITRAGE['exchanges'])
                )

                # Place market order
                order = await self.exchange_manager.execute_trade(
                    exchange_name,
                    symbol,
                    direction,
                    position_size
                )

                if order:
                    # Set take profit and stop loss
                    if direction == 'buy':
                        take_profit = current_price * 1.02  # 2% profit target
                        stop_loss = current_price * 0.995   # 0.5% stop loss
                    else:
                        take_profit = current_price * 0.98
                        stop_loss = current_price * 1.005

                    # Place take profit order
                    tp_order = await self.exchange_manager.execute_trade(
                        exchange_name,
                        symbol,
                        'sell' if direction == 'buy' else 'buy',
                        position_size,
                        take_profit
                    )

                    self.active_trades.append({
                        'entry_order': order,
                        'tp_order': tp_order,
                        'direction': direction,
                        'news_data': news_data,
                        'entry_price': current_price,
                        'take_profit': take_profit,
                        'stop_loss': stop_loss,
                        'exchange': exchange_name
                    })

                    self.logger.info(f"Executed news trade: {direction} on {exchange_name} at {current_price}")

        except Exception as e:
            self.logger.error(f"Error executing news trade: {str(e)}")

    async def monitor_active_trades(self):
        """Monitor and manage active news trades"""
        while True:
            try:
                for trade in list(self.active_trades):
                    exchange = trade['exchange']
                    symbol = 'BTC/USDT'
                    
                    # Check current price
                    ticker = await self.exchange_manager.exchanges[exchange].fetch_ticker(symbol)
                    current_price = ticker['last']

                    # Check if stop loss hit
                    if (trade['direction'] == 'buy' and current_price <= trade['stop_loss']) or \
                       (trade['direction'] == 'sell' and current_price >= trade['stop_loss']):
                        
                        # Close position at market
                        close_order = await self.exchange_manager.execute_trade(
                            exchange,
                            symbol,
                            'sell' if trade['direction'] == 'buy' else 'buy',
                            float(trade['entry_order']['filled'])
                        )

                        if close_order:
                            profit = self.calculate_profit(trade, close_order)
                            self.daily_profit += profit
                            self.active_trades.remove(trade)
                            
                            self.logger.info(f"Closed news trade at stop loss. Profit: ${profit:.2f}")

                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Error monitoring news trades: {str(e)}")
                await asyncio.sleep(5)

    def calculate_profit(self, trade: Dict, close_order: Dict) -> float:
        """Calculate profit from a trade"""
        entry_price = float(trade['entry_order']['price'])
        close_price = float(close_order['price'])
        size = float(trade['entry_order']['filled'])

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
            'success_rate': sum(1 for t in self.active_trades if self.calculate_profit(t, t['tp_order']) > 0) / max(len(self.active_trades), 1)
        }
