# from trading_bot import GoldTradingBot
# from rich.console import Console

# if __name__ == "__main__":
#     console = Console()
    
#     try:
#         bot = GoldTradingBot()
#         bot.run()
#     except KeyboardInterrupt:
#         console.print("\n[yellow]Shutting down...[/yellow]")
#     except Exception as e:
#         console.print(f"[bold red]Error: {str(e)}[/bold red]") 


from new_trading_bot import GoldTradingBot
from rich.console import Console

def main():
    console = Console()
    
    try:
        bot = GoldTradingBot()
        bot.run()
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
    # finally:
    #     mt5.shutdown()
    #     console.print("[yellow]Bot shutdown complete[/yellow]")

if __name__ == "__main__":
    main()