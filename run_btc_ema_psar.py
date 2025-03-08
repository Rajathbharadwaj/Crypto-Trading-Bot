import MetaTrader5 as mt5
from datetime import datetime
import pytz
from btc_ema_psar_bot import BTCEmaPsarBot
from rich.console import Console
from rich.table import Table
from rich import box
from rich.align import Align
import time
import traceback
import sys
import threading
import signal
import os
import logging

# Set up logging to file
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, f"btc_bot_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global variables for watchdog
watchdog_active = True
last_heartbeat = 0

def log_exception(e, message="Exception occurred"):
    """Log exception details to both console and file"""
    console = Console()
    error_str = f"{message}: {str(e)}\n{traceback.format_exc()}"
    console.print(f"[red]{error_str}[/red]")
    logging.error(error_str)

def watchdog_timer():
    """Watchdog timer thread to detect if main process stops heartbeating"""
    console = Console()
    global last_heartbeat, watchdog_active
    
    while watchdog_active:
        # Check if heartbeat has been updated in the last 60 seconds
        current_time = time.time()
        if last_heartbeat > 0 and current_time - last_heartbeat > 60:
            message = "WATCHDOG ALERT: Bot appears to be frozen (no heartbeat for 60+ seconds)"
            console.print(f"[bold red]{message}[/bold red]")
            logging.critical(message)
            console.print("[yellow]Sending termination signal to allow auto-restart...[/yellow]")
            # Send SIGTERM to the main process to trigger restart
            try:
                os.kill(os.getpid(), signal.SIGTERM)
            except Exception as e:
                log_exception(e, "Failed to send termination signal")
            return
        
        # Sleep for a while before checking again
        time.sleep(5)

def update_heartbeat():
    """Update the heartbeat timestamp"""
    global last_heartbeat
    last_heartbeat = time.time()
    logging.debug(f"Heartbeat updated at {datetime.fromtimestamp(last_heartbeat)}")

def main():
    console = Console()
    global watchdog_active, last_heartbeat
    
    logging.info("BTC Bot starting")
    
    # Set up the watchdog
    watchdog_thread = threading.Thread(target=watchdog_timer, daemon=True)
    watchdog_thread.start()
    
    # Display startup message with fancy header
    console.print("\n" + "="*50, style="blue")
    console.print(Align("BTC EMA-PSAR Trading Bot", align="center"), style="bold blue")
    console.print("="*50 + "\n", style="blue")
    
    # Create configuration table
    config_table = Table(title="Bot Configuration", box=box.ROUNDED)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_params = {
        "Symbol": "BTCUSD",
        "Timeframe": "M15",
        "EMA Fast": "9",
        "EMA Slow": "20",
        "PSAR Step": "0.02",
        "PSAR Max": "0.2",
        "Price Movement": "0.15%",
        "Lot Size": "0.1",
        "Risk Management": "Dynamic SL/TP",
        "Exit Strategy": "Multi-level with PSAR"
    }
    
    for param, value in config_params.items():
        config_table.add_row(param, value)
    
    # Set up for recovery mechanism
    max_restarts = 5
    restart_count = 0
    
    while restart_count <= max_restarts:
        try:
            # Update heartbeat to show we're alive
            update_heartbeat()
            
            # Initialize bot
            console.print(f"\nInitializing bot (attempt {restart_count + 1}/{max_restarts + 1})...", style="yellow")
            logging.info(f"Initializing bot (attempt {restart_count + 1}/{max_restarts + 1})")
            
            # Create the bot with exception handling
            try:
                bot = BTCEmaPsarBot()
                console.print("[green]Bot instance created successfully[/green]")
                logging.info("Bot instance created successfully")
            except Exception as bot_error:
                log_exception(bot_error, "Error creating bot instance")
                raise
            
            # Display configuration if this is the first start
            if restart_count == 0:
                console.print(config_table)
                
                # Display controls
                console.print("\nControls:", style="bold cyan")
                console.print("• Press Ctrl+C to stop the bot")
                console.print("• Bot will automatically handle trades based on EMA-PSAR strategy")
                console.print("• Check terminal for real-time signals and trades\n")
                logging.info("Initial configuration displayed")
            else:
                console.print("\n[bold yellow]Bot restarted after error - resuming operations...[/bold yellow]")
                logging.info(f"Bot restarted after error - attempt {restart_count}")
            
            # Start bot with additional safeguards
            console.print("[bold green]Bot is now running...[/bold green]\n")
            
            # Track when the bot starts running
            start_time = datetime.now()
            console.print(f"[dim]Bot started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
            logging.info(f"Bot started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Create a heartbeat updater thread
            def heartbeat_updater():
                while watchdog_active:
                    update_heartbeat()
                    time.sleep(5)
            
            heartbeat_thread = threading.Thread(target=heartbeat_updater, daemon=True)
            heartbeat_thread.start()
            logging.info("Heartbeat thread started")
            
            # Start the bot with extensive exception capture
            try:
                logging.info("Starting bot.run()")
                # Install a global exception hook to catch unhandled exceptions
                def global_exception_hook(exctype, value, tb):
                    error_msg = f"UNHANDLED EXCEPTION: {exctype.__name__}: {value}"
                    console.print(f"[bold red]{error_msg}[/bold red]")
                    logging.critical(error_msg)
                    logging.critical("".join(traceback.format_tb(tb)))
                    raise value  # Re-raise the exception to let the try/except handle it
                
                # Set the exception hook
                sys.excepthook = global_exception_hook
                
                # Run the bot
                bot.run()
                
                # If we get here, the bot exited normally
                console.print("[yellow]Bot.run() completed normally[/yellow]")
                logging.info("Bot.run() completed normally")
            except Exception as run_error:
                log_exception(run_error, "Error in bot.run()")
                raise
            
            # If we reach here, the bot exited normally without an exception
            # Calculate uptime
            end_time = datetime.now()
            uptime = end_time - start_time
            uptime_msg = f"Bot ran for {uptime.total_seconds()/60:.1f} minutes"
            console.print(f"[dim]{uptime_msg}[/dim]")
            logging.info(uptime_msg)
            
            # If we get here, the bot exited normally
            console.print("\n[bold green]Bot completed its run normally[/bold green]")
            logging.info("Bot completed its run normally")
            break
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Bot shutdown requested by user...[/yellow]")
            logging.info("Bot shutdown requested by user")
            break
            
        except Exception as e:
            restart_count += 1
            log_exception(e, f"General error (restart {restart_count}/{max_restarts})")
            
            if restart_count <= max_restarts:
                wait_time = min(300, 30 * restart_count)  # Exponential backoff, max 5 minutes
                console.print(f"[yellow]Bot will restart in {wait_time} seconds (attempt {restart_count}/{max_restarts})...[/yellow]")
                logging.info(f"Bot will restart in {wait_time} seconds (attempt {restart_count}/{max_restarts})")
                
                # Make sure MT5 is shutdown cleanly before restart
                try:
                    if mt5.initialize():
                        mt5.shutdown()
                        console.print("[dim]MT5 connection closed successfully[/dim]")
                        logging.info("MT5 connection closed successfully before restart")
                except Exception as mt5_error:
                    log_exception(mt5_error, "Error shutting down MT5")
                    
                # Wait before restart, checking heartbeat
                console.print(f"[dim]Waiting {wait_time} seconds before restarting...[/dim]")
                
                # Sleep in small intervals to respond to KeyboardInterrupt quickly
                try:
                    for _ in range(int(wait_time)):
                        time.sleep(1)
                        update_heartbeat()  # Update heartbeat during wait
                except KeyboardInterrupt:
                    console.print("\n[yellow]Restart interrupted by user[/yellow]")
                    logging.info("Restart interrupted by user")
                    watchdog_active = False  # Stop the watchdog
                    break
            else:
                console.print("[bold red]Maximum restart attempts reached. Please check the logs and fix any issues.[/bold red]")
                logging.critical("Maximum restart attempts reached")
                break
    
    # Signal the watchdog to stop
    watchdog_active = False
    logging.info("Watchdog deactivated")
    
    # Final cleanup
    try:
        if mt5.initialize():
            mt5.shutdown()
            console.print("\n[blue]MT5 connection closed successfully[/blue]")
            logging.info("MT5 connection closed successfully during final cleanup")
    except Exception as cleanup_error:
        log_exception(cleanup_error, "Error during final cleanup")
        
    console.print("\n[bold blue]Bot shutdown complete[/bold blue]")
    console.print("="*50 + "\n", style="blue")
    logging.info("Bot shutdown complete")

if __name__ == "__main__":
    try:
        main()
    except Exception as fatal_error:
        # Last resort error handler
        console = Console()
        error_message = f"FATAL ERROR IN MAIN FUNCTION: {str(fatal_error)}"
        console.print(f"\n[bold red]{error_message}[/bold red]")
        console.print(f"[red]{traceback.format_exc()}[/red]")
        console.print("[yellow]The application will now exit. Please check the logs and fix any issues.[/yellow]")
        
        # Log to file
        logging.critical(error_message)
        logging.critical(traceback.format_exc())
        
        sys.exit(1) 