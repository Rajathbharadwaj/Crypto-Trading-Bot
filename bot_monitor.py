import subprocess
import time
import sys
import logging
from datetime import datetime
from rich.console import Console
import platform
import signal
import os

class BotWatchdog:
    def __init__(self):
        self.console = Console()
        self.setup_logging()
        self.system = platform.system()
        self.process = None
        self.running = True
        
    def setup_logging(self):
        logging.basicConfig(
            filename='watchdog.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('Watchdog')
        
    def activate_conda_and_run(self):
        """Activate conda environment and run the bot"""
        try:
            if self.system == 'Windows':
                cmd = 'conda activate mt5 && python main.py'
                self.process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    universal_newlines=True
                )
            else:  # For Linux/Mac
                cmd = 'source activate mt5 && python main.py'
                self.process = subprocess.Popen(
                    cmd,
                    shell=True,
                    executable='/bin/bash',
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    universal_newlines=True
                )
            return self.process
            
        except Exception as e:
            self.logger.error(f"Error activating conda: {str(e)}")
            self.console.print(f"[bold red]Error activating conda: {str(e)}[/bold red]")
            return None
            
    def cleanup(self):
        """Clean up the process"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()
                
    def run_bot(self):
        while self.running:
            try:
                self.console.print(f"\n[green]Starting trading bot at {datetime.now()}[/green]")
                self.logger.info("Starting trading bot")
                
                # Activate conda and run bot
                process = self.activate_conda_and_run()
                output = process.stdout.readline()
                error = process.stderr.readline()
                    
                # Process has finished
                if output == '' and error == '' and process.poll() is not None:
                    break
                
                if process is None:
                    self.console.print("[bold red]Failed to start bot, retrying in 30 seconds...[/bold red]")
                    time.sleep(30)
                    continue
                
                # Wait for process to complete
                process.wait()
                
                if not self.running:
                    break
                    
                self.console.print(f"[yellow]Bot stopped, restarting in 5 seconds...[/yellow]")
                time.sleep(5)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Shutting down...[/yellow]")
                self.running = False
                self.cleanup()
                break
            except Exception as e:
                self.console.print(f"[bold red]Error: {str(e)}[/bold red]")
                time.sleep(5)

if __name__ == "__main__":
    watchdog = BotWatchdog()
    try:
        watchdog.run_bot()
    except KeyboardInterrupt:
        watchdog.running = False
        watchdog.cleanup()
    finally:
        print("\nWatchdog shutdown complete")