import asyncio
import logging
from typing import Dict, List
from datetime import datetime, timedelta
import json

from .core.exchange_manager import ExchangeManager
from .core.news_monitor import NewsMonitor
from .strategies.flash_arbitrage import FlashArbitrage
from .strategies.liquidity_provision import LiquidityProvider
from .strategies.news_trader import NewsTrader
from .strategies.orderbook_analyzer import OrderBookAnalyzer

class BotOrchestrator:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.exchange_manager = ExchangeManager(config)
        self.news_monitor = NewsMonitor(config)
        
        # Initialize strategies
        self.flash_arbitrage = FlashArbitrage(self.exchange_manager, config)
        self.liquidity_provider = LiquidityProvider(self.exchange_manager, config)
        self.news_trader = NewsTrader(self.exchange_manager, self.news_monitor, config)
        self.orderbook_analyzer = OrderBookAnalyzer(self.exchange_manager, config)
        
        # Trading state
        self.is_trading = False
        self.daily_stats = {
            'total_profit': 0,
            'trades_executed': 0,
            'successful_trades': 0
        }
        self.last_reset = datetime.now()

    async def initialize(self):
        """Initialize all components and connections"""
        try:
            await self.exchange_manager.initialize()
            self.logger.info("Exchange connections initialized")
            
            # Load previous state if exists
            await self.load_state()
            
            self.is_trading = True
            self.logger.info("Bot initialization complete")
            
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}")
            raise

    async def start(self, symbol: str = 'BTC/USDT'):
        """Start all trading strategies"""
        try:
            await self.initialize()
            
            # Start all strategy tasks
            tasks = [
                self.run_flash_arbitrage(symbol),
                self.run_liquidity_provision(symbol),
                self.run_news_trading(symbol),
                self.run_orderbook_analysis(symbol),
                self.monitor_overall_risk(),
                self.save_state_periodically()
            ]
            
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
            await self.shutdown()

    async def run_flash_arbitrage(self, symbol: str):
        """Run flash arbitrage strategy"""
        while self.is_trading:
            try:
                await self.flash_arbitrage.monitor_and_execute(symbol)
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error in flash arbitrage: {str(e)}")
                await asyncio.sleep(1)

    async def run_liquidity_provision(self, symbol: str):
        """Run liquidity provision strategy"""
        while self.is_trading:
            try:
                for exchange in self.config.FLASH_ARBITRAGE['exchanges']:
                    await self.liquidity_provider.run(symbol, exchange)
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error in liquidity provision: {str(e)}")
                await asyncio.sleep(1)

    async def run_news_trading(self, symbol: str):
        """Run news trading strategy"""
        while self.is_trading:
            try:
                await self.news_trader.monitor_active_trades()
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error in news trading: {str(e)}")
                await asyncio.sleep(1)

    async def run_orderbook_analysis(self, symbol: str):
        """Run orderbook analysis strategy"""
        while self.is_trading:
            try:
                for exchange in self.config.FLASH_ARBITRAGE['exchanges']:
                    await self.orderbook_analyzer.run(symbol, exchange)
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error in orderbook analysis: {str(e)}")
                await asyncio.sleep(1)

    async def monitor_overall_risk(self):
        """Monitor and manage overall risk across all strategies"""
        while self.is_trading:
            try:
                total_position_value = 0
                total_daily_profit = 0
                
                # Calculate total position value and profit
                strategies = [self.flash_arbitrage, self.liquidity_provider, 
                            self.news_trader, self.orderbook_analyzer]
                
                for strategy in strategies:
                    stats = strategy.get_statistics()
                    total_daily_profit += stats['daily_profit']
                    
                    # Sum up position values from active trades
                    for trade in getattr(strategy, 'active_trades', []):
                        position_size = float(trade.get('size', 0))
                        price = float(trade.get('entry_price', 0))
                        total_position_value += position_size * price

                # Check if we need to reduce risk
                if total_position_value > self.config.MAX_POSITION_SIZE:
                    await self.reduce_risk()
                
                # Update daily statistics
                self.daily_stats['total_profit'] = total_daily_profit
                
                # Reset daily stats if needed
                if datetime.now() - self.last_reset > timedelta(days=1):
                    await self.reset_daily_stats()
                
                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Error in risk monitoring: {str(e)}")
                await asyncio.sleep(5)

    async def reduce_risk(self):
        """Reduce risk by closing some positions"""
        try:
            strategies = [self.flash_arbitrage, self.liquidity_provider, 
                         self.news_trader, self.orderbook_analyzer]
            
            for strategy in strategies:
                # Close oldest trades first
                active_trades = getattr(strategy, 'active_trades', [])
                if active_trades:
                    # Sort by entry time and close oldest
                    sorted_trades = sorted(active_trades, 
                                        key=lambda x: x['entry_order']['timestamp'])
                    if sorted_trades:
                        oldest_trade = sorted_trades[0]
                        await strategy.close_position(oldest_trade)
                        
            self.logger.info("Reduced overall position risk")
            
        except Exception as e:
            self.logger.error(f"Error reducing risk: {str(e)}")

    async def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_stats = {
            'total_profit': 0,
            'trades_executed': 0,
            'successful_trades': 0
        }
        self.last_reset = datetime.now()
        
        # Save state after reset
        await self.save_state()

    async def save_state(self):
        """Save current state to file"""
        try:
            state = {
                'daily_stats': self.daily_stats,
                'last_reset': self.last_reset.isoformat(),
                'flash_arbitrage_stats': self.flash_arbitrage.get_statistics(),
                'liquidity_provider_stats': self.liquidity_provider.get_statistics(),
                'news_trader_stats': self.news_trader.get_statistics(),
                'orderbook_analyzer_stats': self.orderbook_analyzer.get_statistics()
            }
            
            with open('bot_state.json', 'w') as f:
                json.dump(state, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")

    async def load_state(self):
        """Load state from file"""
        try:
            with open('bot_state.json', 'r') as f:
                state = json.load(f)
                
            self.daily_stats = state['daily_stats']
            self.last_reset = datetime.fromisoformat(state['last_reset'])
            
        except FileNotFoundError:
            self.logger.info("No previous state found, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")

    async def save_state_periodically(self):
        """Periodically save state"""
        while self.is_trading:
            await self.save_state()
            await asyncio.sleep(300)  # Save every 5 minutes

    async def shutdown(self):
        """Gracefully shutdown the bot"""
        self.is_trading = False
        
        try:
            # Close all positions
            await self.reduce_risk()
            
            # Save final state
            await self.save_state()
            
            # Close exchange connections
            await self.exchange_manager.cleanup()
            
            self.logger.info("Bot shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

    def get_overall_statistics(self) -> Dict:
        """Get overall trading statistics"""
        return {
            'daily_stats': self.daily_stats,
            'flash_arbitrage': self.flash_arbitrage.get_statistics(),
            'liquidity_provider': self.liquidity_provider.get_statistics(),
            'news_trader': self.news_trader.get_statistics(),
            'orderbook_analyzer': self.orderbook_analyzer.get_statistics(),
            'total_profit': self.daily_stats['total_profit'],
            'uptime': (datetime.now() - self.last_reset).total_seconds() / 3600  # hours
        }
