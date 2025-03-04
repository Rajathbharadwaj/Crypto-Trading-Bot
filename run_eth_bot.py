import MetaTrader5 as mt5
from datetime import datetime
import pytz
from eth_bot import ETHTradingBot
from rich.console import Console

def main():
    console = Console()
    
    # Display startup message
    console.print("\n=== ETH Trading Bot Startup ===", style="bold blue")
    console.print("Initializing...", style="yellow")
    
    # Create bot instance
    try:
        bot = ETHTradingBot()
        
        # Display configuration
        console.print("\nBot Configuration:", style="bold green")
        console.print(f"Symbol: ETHUSD")
        console.print(f"Timeframe: M15")
        console.print(f"Trading Days: All days of the week")
        console.print(f"Initial Stop Loss: 0.25%")
        console.print(f"Lot Size: 1.0")
        console.print(f"Partial Exits: 0.50 lots at 0.25%, 0.20 lots at 0.50%, 0.20 lots at 0.75%, 0.10 lots with trailing stop")
        
        # Start the bot
        console.print("\n[bold green]Starting bot...[/bold green]")
        console.print("Press Ctrl+C to stop the bot\n")
        
        bot.run()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Bot shutdown requested by user...[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
    finally:
        # Cleanup
        if mt5.initialize():
            mt5.shutdown()
        console.print("\n[bold blue]Bot shutdown complete[/bold blue]")

if __name__ == "__main__":
    main() 