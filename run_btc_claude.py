import MetaTrader5 as mt5
from datetime import datetime
import pytz
from btc_bot import BTCTradingBot
from rich.console import Console

def main():
    console = Console()
    
    # Display startup message
    console.print("\n=== BTC Trading Bot Startup ===", style="bold blue")
    console.print("Initializing...", style="yellow")
    
    # Create bot instance
    try:
        bot = BTCTradingBot()
        
        # Display configuration
        console.print("\nBot Configuration:", style="bold green")
        console.print(f"Symbol: BTCUSD")
        console.print(f"Timeframe: M1")
        console.print(f"Maximum Stop Loss: 30 pips")
        console.print("[green]Trading 24/7 - Crypto market never sleeps![/green]")
        
        # Start the bot
        console.print("\n[bold green]Starting bot...[/bold green]")
        console.print("Press Ctrl+C to stop the bot\n")
        
        bot.run()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Bot shutdown requested by user...[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
    finally:
        # Cleanup
        if mt5.initialize():
            mt5.shutdown()
        console.print("\n[bold blue]Bot shutdown complete[/bold blue]")

if __name__ == "__main__":
    main() 