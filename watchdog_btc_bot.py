#!/usr/bin/env python
import subprocess
import time
import os
import signal
import sys
from datetime import datetime

def log_message(message):
    """Log a message with timestamp to console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def run_watchdog():
    """Run the BTC bot with a watchdog that will restart it if it crashes"""
    log_message("Starting BTC Bot Watchdog")
    
    max_restarts = 20
    restart_count = 0
    
    while restart_count <= max_restarts:
        restart_count += 1
        
        # Log restart attempt
        log_message(f"Starting BTC bot (attempt {restart_count}/{max_restarts+1})")
        
        # Start the bot process
        try:
            # Start the bot as a subprocess without piping output so it shows directly in console
            process = subprocess.Popen([sys.executable, "run_btc_ema_psar.py"])
            
            log_message(f"BTC bot started with PID {process.pid}")
            
            # Wait for the process to complete
            process.wait()
            
            # Process has exited, check return code
            exit_code = process.returncode
            log_message(f"BTC bot process exited with code {exit_code}")
            
            # If exit code is 0, it was a clean exit
            if exit_code == 0:
                log_message("BTC bot exited cleanly. Not restarting.")
                break
            
            # Otherwise, it crashed
            log_message("BTC bot crashed. Preparing to restart...")
            
        except KeyboardInterrupt:
            log_message("Shutdown requested by user. Terminating bot process...")
            if process and process.poll() is None:
                try:
                    process.terminate()
                    # Give it a moment to terminate gracefully
                    time.sleep(5)
                    # If it's still running, force kill
                    if process.poll() is None:
                        process.kill()
                except:
                    pass
            log_message("Watchdog exiting.")
            break
            
        except Exception as e:
            log_message(f"Error in watchdog: {str(e)}")
            
        # Wait before restart (exponential backoff with a small maximum wait time)
        if restart_count <= max_restarts:
            # Use exponential backoff but cap at 5 seconds to avoid leaving trades unmanaged
            wait_time = min(5, 1 * (2 ** min(restart_count - 1, 2)))
            log_message(f"Waiting {wait_time} seconds before restarting...")
            
            try:
                time.sleep(wait_time)
            except KeyboardInterrupt:
                log_message("Restart interrupted by user. Exiting watchdog.")
                break
        else:
            log_message("Maximum restart attempts reached. Watchdog exiting.")
            break
    
    log_message("Watchdog terminated.")

if __name__ == "__main__":
    try:
        run_watchdog()
    except KeyboardInterrupt:
        print("\nWatchdog terminated by user.")
    except Exception as e:
        print(f"Fatal error in watchdog: {str(e)}")
    finally:
        print("Watchdog exited.") 