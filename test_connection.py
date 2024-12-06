from trading_bot import GoldTradingBot
from rich.console import Console

if __name__ == "__main__":
    console = Console()
    bot = GoldTradingBot()
    
    try:
        # Test connection
        if bot.check_connection():
            console.print("[green]Connection test successful![/green]")
            
            # Get a single live tick
            data = bot.get_live_data()
            if data is not None:
                console.print("\nLive Data Sample:")
                console.print(data)
        else:
            console.print("[red]Connection test failed![/red]")
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]") 