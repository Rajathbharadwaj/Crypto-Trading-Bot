import threading
from new_trading_bot import GoldTradingBot
import logging
from rich.console import Console
import time
import MetaTrader5 as mt5

class MultiTimeframeBot:
    def __init__(self):
        self.console = Console()
        self.bots = {}
        self.threads = {}
        
    def create_bot(self, timeframe, name):
        """Create a new bot instance for a specific timeframe"""
        bot = GoldTradingBot(timeframe=timeframe, name=f"Gold_{name}")
        self.bots[name] = bot
        return bot
        
    def start_bot(self, name):
        """Start a bot in its own thread"""
        if name not in self.bots:
            self.console.print(f"[red]Bot {name} not found![/red]")
            return
            
        bot = self.bots[name]
        thread = threading.Thread(target=bot.run, name=f"Thread_{name}", daemon=True)
        self.threads[name] = thread
        thread.start()
        self.console.print(f"[green]Started bot: {name}[/green]")
        
    def start_all(self):
        """Start all bots"""
        for name in self.bots:
            self.start_bot(name)
            
    def stop_bot(self, name):
        """Stop a specific bot"""
        if name in self.bots:
            self.bots[name].running = False
            if name in self.threads:
                self.threads[name].join(timeout=5)
                self.console.print(f"[yellow]Stopped bot: {name}[/yellow]")
                
    def stop_all(self):
        """Stop all bots"""
        for name in list(self.bots.keys()):
            self.stop_bot(name)

# Usage example
if __name__ == "__main__":
    # Initialize multi-timeframe manager
    mtf = MultiTimeframeBot()
    
    # Create bots for different timeframes
    mtf.create_bot(mt5.TIMEFRAME_M5, "M5")
    mtf.create_bot(mt5.TIMEFRAME_M1, "M1")
    
    try:
        # Start all bots
        mtf.start_all()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping all bots...")
        mtf.stop_all() 