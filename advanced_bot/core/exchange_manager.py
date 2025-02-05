import ccxt.async_support as ccxt
import asyncio
from typing import Dict, List
import logging

class ExchangeManager:
    def __init__(self, config):
        self.config = config
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.orderbooks: Dict[str, Dict] = {}
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize connections to all configured exchanges"""
        exchange_configs = {
            'binance': {
                'apiKey': self.config.BINANCE_API_KEY,
                'secret': self.config.BINANCE_SECRET_KEY,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            },
            'coinbase': {
                'apiKey': self.config.COINBASE_API_KEY,
                'secret': self.config.COINBASE_SECRET_KEY,
                'enableRateLimit': True
            },
            'bybit': {
                'apiKey': self.config.BYBIT_API_KEY,
                'secret': self.config.BYBIT_SECRET_KEY,
                'enableRateLimit': True
            }
        }

        for name, config in exchange_configs.items():
            try:
                exchange_class = getattr(ccxt, name)
                self.exchanges[name] = exchange_class(config)
                await self.exchanges[name].load_markets()
                self.logger.info(f"Initialized {name} exchange")
            except Exception as e:
                self.logger.error(f"Failed to initialize {name}: {str(e)}")

    async def get_orderbook(self, exchange: str, symbol: str) -> Dict:
        """Get real-time orderbook data"""
        try:
            orderbook = await self.exchanges[exchange].fetch_order_book(symbol)
            self.orderbooks[exchange] = orderbook
            return orderbook
        except Exception as e:
            self.logger.error(f"Error fetching orderbook from {exchange}: {str(e)}")
            return None

    async def execute_trade(self, exchange: str, symbol: str, side: str, amount: float, price: float = None) -> Dict:
        """Execute a trade on specified exchange"""
        try:
            if price:
                order = await self.exchanges[exchange].create_limit_order(
                    symbol, side, amount, price
                )
            else:
                order = await self.exchanges[exchange].create_market_order(
                    symbol, side, amount
                )
            return order
        except Exception as e:
            self.logger.error(f"Trade execution failed on {exchange}: {str(e)}")
            return None

    async def get_price_differences(self, symbol: str) -> List[Dict]:
        """Calculate price differences across exchanges"""
        prices = {}
        differences = []
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                ticker = await exchange.fetch_ticker(symbol)
                prices[exchange_name] = {
                    'bid': ticker['bid'],
                    'ask': ticker['ask']
                }
            except Exception as e:
                self.logger.error(f"Error fetching price from {exchange_name}: {str(e)}")

        # Calculate differences between exchanges
        exchanges = list(prices.keys())
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                ex1, ex2 = exchanges[i], exchanges[j]
                if ex1 in prices and ex2 in prices:
                    spread = (prices[ex2]['bid'] - prices[ex1]['ask']) / prices[ex1]['ask']
                    if spread > self.config.FLASH_ARBITRAGE['min_spread']:
                        differences.append({
                            'buy_exchange': ex1,
                            'sell_exchange': ex2,
                            'spread': spread,
                            'buy_price': prices[ex1]['ask'],
                            'sell_price': prices[ex2]['bid']
                        })

        return differences

    async def cleanup(self):
        """Close all exchange connections"""
        for exchange in self.exchanges.values():
            try:
                await exchange.close()
            except Exception as e:
                self.logger.error(f"Error closing exchange connection: {str(e)}")
