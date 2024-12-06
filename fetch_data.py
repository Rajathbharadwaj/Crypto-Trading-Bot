import MetaTrader5 as mt5
from datetime import datetime
import time

def setup_mt5():
    # First, check if MT5 is already running
    if mt5.initialize():
        print("MT5 is already initialized")
        
        # Get the current account info
        account_info = mt5.account_info()
        if account_info is not None:
            print("Already logged in as:", account_info.login)
            return True
        
    # If not initialized or no account info, try to initialize
    print("Attempting to initialize MT5...")
    if not mt5.initialize():
        print("MT5 initialization failed")
        print(f"Error code: {mt5.last_error()}")
        return False

    print("MT5 initialized successfully")
    
    # Print terminal info
    terminal_info = mt5.terminal_info()
    if terminal_info is not None:
        print("Terminal connected:", terminal_info.connected)
        print("Terminal trade_allowed:", terminal_info.trade_allowed)
        print("Terminal path:", terminal_info.path)
    
    # Try to get current account info before login
    account_info = mt5.account_info()
    if account_info is not None:
        print("Current account:", account_info.login)
    else:
        print("No account currently logged in")

    # Now try to login
    print("\nAttempting to login...")
    authorized = mt5.login(
        login=182277035,
        password="QAZwsx456!",
        server="Exness-MT5Trial6"
    )

    if not authorized:
        print(f"Failed to connect to account, error code: {mt5.last_error()}")
        mt5.shutdown()
        return False

    print("Connected to MT5 successfully!")
    return True

if __name__ == "__main__":
    try:
        setup_mt5()
    finally:
        print("\nShutting down MT5 connection...")
        mt5.shutdown()