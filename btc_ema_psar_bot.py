import traceback
import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta, timezone
from rich.console import Console
from rich.table import Table
import time
import os
import json
import gc  # Add garbage collection module
import math
import random

# Define a custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

class BTCEmaPsarBot:
    def __init__(self):
        """Initialize the bot"""
        # MT5 connection setup
        if not mt5.initialize():
            print("MT5 initialization failed")
            quit()
        
        # Set up logging with Rich
        self.console = Console()
        self.console.print("[bold green]BTC EMA-PSAR Bot Started[/bold green]")
        
        # Define timeframe mapping
        self.TIMEFRAMES = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        
        # Symbol and timeframe settings
        self.symbol = "BTCUSD"
        self.timeframe = self.TIMEFRAMES["M15"]  # Default to M15
        self.running = True
        
        # Set lot size - fix for volume error
        # Using 0.1 lots instead of 1 (adjust based on your broker's minimum)
        self.lot_size = 0.1
        
        # Initialize tracking dict for partial exits
        self.partial_exits_taken = {}
        
        # Add file paths for state and logs
        self.log_dir = "logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Fix missing state_file attribute
        self.state_file = os.path.join(self.log_dir, f"btc_ema_psar_state.json")
        self.trade_log_file = os.path.join(self.log_dir, f"btc_ema_psar_trades.json")
        
        # Initialize crossover tracking variables
        self.crossover_detected = False
        self.crossover_type = None
        self.crossover_price = None
        self.crossover_time = None
        
        # Add PSAR direction tracking (from ETH bot)
        self.last_psar_direction = None
        self.psar_direction_changed = False
        self.psar_direction_change_time = None
        
        # Add counter-trend tracking
        self.counter_trend_positions = {}  # Dict to track counter-trend positions by ticket
        self.counter_trend_active = False  # Flag to track if we have active counter-trend trades
        self.last_counter_trend_check = 0  # Timestamp of last counter-trend check
        
        # Load previous state if exists
        self.load_state()
        
        # Indicator parameters
        self.ema_fast = 9
        self.ema_slow = 20
        self.psar_step = 0.02
        self.psar_max = 0.2
        
        # Trading parameters
        self.price_movement_threshold = 0.15  # 0.15% movement required
        self.psar_distance_multiplier = 0.8   # PSAR should be at least 80% of average distance
        
        # Initialize MT5
        if not self.initialize_mt5():
            raise Exception("Failed to initialize MT5")

        # Store active symbol - the one we actually use for trading
        self.active_symbol = None
        self.alternative_symbols = ["BTCUSDT", "BTC/USD", "BTC"]

    def initialize_mt5(self):
        """Initialize connection to MetaTrader 5"""
        self.console.print("[yellow]Initializing connection to MetaTrader 5...[/yellow]")
        
        # Initialize MT5 if not already
        if not mt5.initialize():
            self.console.print("[red]MT5 initialization failed![/red]")
            error_code = mt5.last_error()
            self.console.print(f"[red]MT5 error code: {error_code}[/red]")
            return False
        
        self.console.print("[green]MT5 initialized successfully[/green]")
        
        # Login to MT5
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                self.console.print(f"[yellow]MT5 login attempt {attempt}/{max_attempts}...[/yellow]")
                if mt5.login(login=182992485, password="QAZwsx456!", server="Exness-MT5Trial6"):
                    self.console.print("[green]Successfully logged in to MT5[/green]")
                    
                    # Create log directory if it doesn't exist
                    if not os.path.exists(self.log_dir):
                        os.makedirs(self.log_dir)
                    
                    # Find a valid BTC symbol to use
                    self._find_valid_btc_symbol()
                    
                    return True
                else:
                    error_code = mt5.last_error()
                    self.console.print(f"[red]MT5 login failed: Error code {error_code}[/red]")
            except Exception as e:
                self.console.print(f"[red]Exception during MT5 login: {str(e)}[/red]")
            
            # Wait before next attempt (with increasing delay)
            if attempt < max_attempts:
                wait_time = 5 * attempt
                self.console.print(f"[yellow]Waiting {wait_time} seconds before next attempt...[/yellow]")
                time.sleep(wait_time)
        
        self.console.print("[red]All MT5 login attempts failed![/red]")
        return False

    def _find_valid_btc_symbol(self):
        """Find a valid BTC symbol that can be used with this broker"""
        try:
            # First, check the default symbol
            self.console.print(f"[dim]Checking if {self.symbol} is available...[/dim]")
            symbols = mt5.symbols_get()
            
            if symbols is None:
                error_code = mt5.last_error()
                self.console.print(f"[red]Failed to get symbols. MT5 error code: {error_code}[/red]")
                return False
            
            # Extract symbol names
            symbol_names = [symbol.name for symbol in symbols]
            
            # Add more logging to show available BTC symbols
            btc_symbols = [name for name in symbol_names if "BTC" in name]
            self.console.print(f"Available BTC symbols: {btc_symbols}")
            
            # First, try the default symbol
            if self.symbol in symbol_names:
                # Test that we can get data for this symbol
                test_data = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 10)
                if test_data is not None:
                    self.active_symbol = self.symbol
                    self.console.print(f"[green]Default symbol {self.symbol} is valid[/green]")
                    return True
                else:
                    self.console.print(f"[yellow]Default symbol {self.symbol} exists but cannot get data[/yellow]")
            
            # Try alternative symbols
            for alt_symbol in self.alternative_symbols:
                if alt_symbol in symbol_names:
                    # Test that we can get data for this symbol
                    test_data = mt5.copy_rates_from_pos(alt_symbol, self.timeframe, 0, 10)
                    if test_data is not None:
                        self.active_symbol = alt_symbol
                        self.console.print(f"[green]Alternative symbol {alt_symbol} is valid[/green]")
                        return True
                    else:
                        self.console.print(f"[yellow]Alternative symbol {alt_symbol} exists but cannot get data[/yellow]")
            
            self.console.print("[red]Could not find any working BTC symbol![/red]")
            return False
        except Exception as e:
            self.console.print(f"[red]Error in _find_valid_btc_symbol: {str(e)}[/red]")
            return False

    def calculate_indicators(self, df):
        """Calculate EMA, PSAR, and distance metrics"""
        try:
            self.console.print("[dim]Calculating technical indicators...[/dim]")
            
            # Make a copy of the dataframe to avoid pandas warnings
            # Use a more efficient copying method
            df = df.copy(deep=False)  # Changed from deep=True to reduce memory usage
            
            # Force garbage collection before heavy calculations
            gc.collect()
            
            # Calculate EMAs
            try:
                # Calculate EMAs safely
                df['ema_9'] = ta.ema(df['close'], length=self.ema_fast)
                df['ema_20'] = ta.ema(df['close'], length=self.ema_slow)
                self.console.print("[dim]EMA calculation successful[/dim]")
            except Exception as ema_error:
                self.console.print(f"[red]Error calculating EMAs: {str(ema_error)}[/red]")
                import traceback
                self.console.print(f"[red]{traceback.format_exc()}[/red]")
                return None
            
            # Force garbage collection between calculations
            gc.collect()
            
            # Calculate PSAR with better error handling
            try:
                # IMPORTANT: ta.psar requires pandas Series, not NumPy arrays
                # DO NOT convert to numpy arrays for this calculation
                psar = ta.psar(
                    high=df['high'],  # Keep as pandas Series
                    low=df['low'],    # Keep as pandas Series
                    close=df['close'], # Keep as pandas Series
                    af0=self.psar_step, 
                    af=self.psar_step, 
                    max_af=self.psar_max
                )
                
                # Get column names for long and short PSAR
                psar_l_col = f'PSARl_{self.psar_step}_{self.psar_max}'
                psar_s_col = f'PSARs_{self.psar_step}_{self.psar_max}'
                
                # Check that we have the expected columns
                if psar_l_col not in psar.columns or psar_s_col not in psar.columns:
                    available_cols = list(psar.columns)
                    self.console.print(f"[yellow]Warning: Expected PSAR columns not found. Available: {available_cols}[/yellow]")
                    
                    # Try to find the right columns based on prefix
                    psar_l_candidates = [col for col in psar.columns if col.startswith('PSARl_')]
                    psar_s_candidates = [col for col in psar.columns if col.startswith('PSARs_')]
                    
                    if psar_l_candidates and psar_s_candidates:
                        psar_l_col = psar_l_candidates[0]
                        psar_s_col = psar_s_candidates[0]
                        self.console.print(f"[yellow]Using alternative PSAR columns: {psar_l_col} and {psar_s_col}[/yellow]")
                    else:
                        self.console.print("[red]Could not find suitable PSAR columns[/red]")
                        return None
                
                # Initialize PSAR column
                df['psar'] = np.nan
                
                # Fill with long PSAR values (safer method)
                mask_l = ~psar[psar_l_col].isna()
                if mask_l.any():
                    df.loc[mask_l, 'psar'] = psar.loc[mask_l, psar_l_col]
                
                # Fill with short PSAR values
                mask_s = ~psar[psar_s_col].isna()
                if mask_s.any():
                    df.loc[mask_s, 'psar'] = psar.loc[mask_s, psar_s_col]
                
                self.console.print("[dim]PSAR calculation successful[/dim]")
            except Exception as psar_error:
                self.console.print(f"[red]Error calculating PSAR: {str(psar_error)}[/red]")
                import traceback
                self.console.print(f"[red]{traceback.format_exc()}[/red]")
                # Try to free memory before returning
                gc.collect()
                return None
            
            # Force garbage collection between calculations
            gc.collect()
            
            # Determine PSAR direction
            try:
                # Calculate direction using pandas operations
                df['psar_direction'] = np.where(df['psar'] < df['low'], 1, -1)  # 1 for bullish, -1 for bearish
                
                # Calculate PSAR distance as percentage
                df['psar_distance'] = np.where(
                    df['psar_direction'] == 1,
                    (df['low'] - df['psar']) / df['psar'] * 100,  # When PSAR is below
                    (df['psar'] - df['high']) / df['psar'] * 100  # When PSAR is above
                )
                
                # Calculate average PSAR distance
                min_periods = min(20, len(df) - 1)
                window_size = min(96, len(df) - 1)
                df['avg_psar_distance'] = df['psar_distance'].rolling(
                    window=window_size, 
                    min_periods=min_periods
                ).mean()
                
                # Fill any remaining NaN values
                df['avg_psar_distance'] = df['avg_psar_distance'].fillna(df['psar_distance'])
                
                self.console.print("[dim]PSAR direction and distance calculated[/dim]")
            except Exception as direction_error:
                self.console.print(f"[red]Error calculating PSAR direction and distance: {str(direction_error)}[/red]")
                import traceback
                self.console.print(f"[red]{traceback.format_exc()}[/red]")
                # Try to free memory before returning
                gc.collect()
                return None
            
            # Check for PSAR direction changes
            if len(df) > 1:
                try:
                    current_direction = df.iloc[-1]['psar_direction']
                    previous_direction = df.iloc[-2]['psar_direction']
                    
                    # If direction has changed and we haven't logged it yet
                    if current_direction != previous_direction and not self.psar_direction_changed:
                        self.psar_direction_changed = True
                        self.psar_direction_change_time = time.time()
                        self.save_state()
                        
                        # Log the direction change
                        direction_text = "Bullish (below price)" if current_direction == 1 else "Bearish (above price)"
                        self.console.print(f"[bold yellow]PSAR direction changed to {direction_text}[/bold yellow]")
                except Exception as change_error:
                    self.console.print(f"[yellow]Warning: Error checking PSAR direction change: {str(change_error)}[/yellow]")
                    # Continue anyway, this is not critical
            
            # Calculate EMA gap and rate of change metrics for counter-trend trading
            try:
                # Calculate the gap between EMAs (absolute value)
                df['ema_gap'] = abs(df['ema_9'] - df['ema_20'])
                
                # Calculate the rate of change (ROC) of the gap over different periods
                df['ema_gap_roc_1'] = df['ema_gap'].pct_change(1) * 100  # 1-period ROC
                df['ema_gap_roc_3'] = df['ema_gap'].pct_change(3) * 100  # 3-period ROC
                
                # Calculate the acceleration (ROC of the ROC)
                df['ema_gap_accel'] = df['ema_gap_roc_1'].pct_change(1) * 100
                
                # Determine if gap is widening or narrowing (boolean)
                df['gap_widening'] = df['ema_gap'] > df['ema_gap'].shift(1)
                
                # Fill any NaN values that result from calculations
                for col in ['ema_gap_roc_1', 'ema_gap_roc_3', 'ema_gap_accel']:
                    df[col] = df[col].fillna(0)
                
                self.console.print("[dim]EMA gap and rate of change metrics calculated[/dim]")
            except Exception as gap_error:
                self.console.print(f"[yellow]Warning: Error calculating EMA gap metrics: {str(gap_error)}[/yellow]")
                # Continue anyway, this is not critical
            
            # Force garbage collection after heavy calculations
            gc.collect()
            
            self.console.print("[dim]Indicator calculations complete[/dim]")
            return df
            
        except Exception as e:
            self.console.print(f"[red]Error in calculate_indicators: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
            # Try to free memory before returning
            gc.collect()
            return None

    def get_trading_signal(self, df):
        """Generate trading signals based on EMA crossover and PSAR direction changes"""
        try:
            if df is None or len(df) < 2:
                return None, None
            
            # Get the latest data
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            # Current prices
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                return None, None
            
            current_ask = symbol_info.ask  # Price to buy at
            current_bid = symbol_info.bid  # Price to sell at
            
            # Current indicators
            ema_fast = last_row['ema_9']
            ema_slow = last_row['ema_20']
            psar = last_row['psar']
            current_psar_direction = last_row['psar_direction']
            psar_distance = last_row['psar_distance']
            
            # Check if avg_psar_distance exists, use psar_distance as fallback
            try:
                avg_psar_distance = last_row['avg_psar_distance']
            except (KeyError, Exception) as e:
                self.console.print(f"[yellow]Warning: avg_psar_distance not found, using current distance as fallback: {e}[/yellow]")
                avg_psar_distance = psar_distance
            
            # Display signal analysis
            self.console.print("\n=== Signal Analysis ===")
            self.console.print(f"Current Ask Price (Buy at): {current_ask}")
            self.console.print(f"Current Bid Price (Sell at): {current_bid}")
            self.console.print(f"EMA9: {ema_fast:.2f}")
            self.console.print(f"EMA20: {ema_slow:.2f}")
            self.console.print(f"PSAR: {psar:.2f}")
            self.console.print(f"PSAR Direction: {'Bullish' if current_psar_direction == 1 else 'Bearish'}")
            self.console.print(f"PSAR Distance: {psar_distance:.2f}%")
            self.console.print(f"Average PSAR Distance: {avg_psar_distance:.2f}%")
            
            # No signals if not enough data
            if pd.isna(ema_fast) or pd.isna(ema_slow) or pd.isna(psar):
                self.console.print("[yellow]Some indicators are NaN, waiting for more data...[/yellow]")
                return None, "Indicators contain NaN values"
            
            # Ensure psar_direction_change_time is a float timestamp
            if self.psar_direction_changed:
                # Convert psar_direction_change_time to float if it's a string (ISO format)
                if isinstance(self.psar_direction_change_time, str):
                    try:
                        # If it's a timestamp stored as string, convert directly
                        if self.psar_direction_change_time.replace('.', '').isdigit():
                            self.psar_direction_change_time = float(self.psar_direction_change_time)
                        # If it's an ISO format timestamp, convert to datetime first then to timestamp
                        else:
                            dt = datetime.fromisoformat(self.psar_direction_change_time)
                            self.psar_direction_change_time = dt.timestamp()
                    except (ValueError, TypeError, AttributeError):
                        # If conversion fails, reset to current time
                        self.console.print("[yellow]Warning: Invalid timestamp format, resetting to current time[/yellow]")
                self.psar_direction_change_time = time.time()
                        
                # Now calculate time difference (will be float - float)
                try:
                    time_since_direction_change = time.time() - self.psar_direction_change_time
                    self.console.print(f"Time since PSAR direction change: {time_since_direction_change:.1f}s")
                    
                    # Reset direction change after 60 minutes if not acted upon
                    if time_since_direction_change > 3600:  # 1 hour
                        self.psar_direction_changed = False
                        self.save_state()
                        self.console.print("[yellow]PSAR direction change expired[/yellow]")
                except Exception as e:
                    self.console.print(f"[red]Error calculating time difference: {str(e)}[/red]")
                    # Reset to avoid future errors
                    self.psar_direction_change_time = time.time()
                    self.save_state()
            
            # Check EMA alignment
            emas_bullish = ema_fast > ema_slow
            emas_bearish = ema_fast < ema_slow
            
            # Fix: Check if PSAR is below/above the Average PSAR range using scalar values, not Series
            # Use the last_row values which are already scalar values from the DataFrame
            psar_below_avg = (psar_distance > avg_psar_distance) and (current_psar_direction == 1)
            psar_above_avg = (psar_distance > avg_psar_distance) and (current_psar_direction == -1)
            
            # Display indicators alignment
            self.console.print(f"EMAs Bullish Aligned: {emas_bullish}")
            self.console.print(f"EMAs Bearish Aligned: {emas_bearish}")
            self.console.print(f"PSAR Direction Changed: {self.psar_direction_changed}")
            self.console.print(f"PSAR Below Avg Range: {psar_below_avg}")
            self.console.print(f"PSAR Above Avg Range: {psar_above_avg}")
            
            # EXPLAIN WHY NO SIGNAL (NEW CODE)
            self.console.print("\n=== Trade Conditions ===")
            
            # Explain BUY conditions
            self.console.print("[bold]For BUY signal:[/bold]")
            self.console.print(f"✓ PSAR Direction Changed: {'✅' if self.psar_direction_changed else '❌'}")
            self.console.print(f"✓ PSAR Direction is Bullish: {'✅' if current_psar_direction == 1 else '❌'}")
            self.console.print(f"✓ EMAs Bullish Aligned: {'✅' if emas_bullish else '❌'}")
            self.console.print(f"✓ PSAR Below Avg Range: {'✅' if psar_below_avg else '❌'}")
            
            # Explain SELL conditions
            self.console.print("[bold]For SELL signal:[/bold]")
            self.console.print(f"✓ PSAR Direction Changed: {'✅' if self.psar_direction_changed else '❌'}")
            self.console.print(f"✓ PSAR Direction is Bearish: {'✅' if current_psar_direction == -1 else '❌'}")
            self.console.print(f"✓ EMAs Bearish Aligned: {'✅' if emas_bearish else '❌'}")
            self.console.print(f"✓ PSAR Above Avg Range: {'✅' if psar_above_avg else '❌'}")
            
            if self.psar_direction_changed:
                # Ensure psar_direction_change_time is a float timestamp
                if isinstance(self.psar_direction_change_time, str):
                    try:
                        # If it's a timestamp stored as string, convert directly
                        if self.psar_direction_change_time.replace('.', '').isdigit():
                            self.psar_direction_change_time = float(self.psar_direction_change_time)
                        # If it's an ISO format timestamp, convert to datetime first then to timestamp
                        else:
                            dt = datetime.fromisoformat(self.psar_direction_change_time)
                            self.psar_direction_change_time = dt.timestamp()
                    except (ValueError, TypeError, AttributeError):
                        # If conversion fails, reset to current time
                        self.console.print("[yellow]Warning: Invalid timestamp format, resetting to current time[/yellow]")
                        self.psar_direction_change_time = time.time()
                        
                # Now calculate time difference (will be float - float)
                try:
                    time_since_direction_change = time.time() - self.psar_direction_change_time
                    self.console.print(f"Time since PSAR direction change: {time_since_direction_change:.1f}s")
                    
                    # Reset direction change after 60 minutes if not acted upon
                    if time_since_direction_change > 3600:  # 1 hour
                        self.psar_direction_changed = False
                        self.save_state()
                        self.console.print("[yellow]PSAR direction change expired[/yellow]")
                except Exception as e:
                    self.console.print(f"[red]Error calculating time difference: {str(e)}[/red]")
                    # Reset to avoid future errors
                    self.psar_direction_change_time = time.time()
                    self.save_state()
            
            # Generate signal when PSAR direction changes and aligns with EMAs
            signal = None
            reason = None
            
            # For bullish setup: PSAR turned bullish (below price) and EMAs are bullish (EMA9 > EMA20)
            if self.psar_direction_changed and current_psar_direction == 1 and emas_bullish:
                # Calculate price movement since PSAR direction change
                price_movement = None
                
                if hasattr(self, 'psar_direction_change_time') and self.psar_direction_change_time:
                    # Get a few candles ago to check price movement
                    candles_ago = min(5, len(df) - 1)  # At most 5 candles back
                    past_price = df.iloc[-candles_ago]['close']
                    price_movement = ((current_bid / past_price) - 1) * 100
                    self.console.print(f"Price movement since direction change: {price_movement:.2f}%")
                
                # Ensure price is moving in the right direction
                if price_movement is None or price_movement > 0:  # Price moving up or undetermined
                    signal = "BUY"
                    reason = "PSAR turned bullish with EMAs aligned and price moving up"
                    
                    self.console.print(f"[bold green]✓ BUY SIGNAL TRIGGERED (PSAR Direction Change)![/bold green]")
                    self.console.print(f"[green]Will buy at: {current_ask}[/green]")
                    self.console.print(f"[yellow]Reason: {reason}[/yellow]")
                    
                    # Reset PSAR direction change flag to prevent multiple signals
                    self.psar_direction_changed = False
                    self.save_state()
                    
                    # Store reason for logging
                    self.last_signal_reason = reason
                    
                    return signal, reason
            
            # For bearish setup: PSAR turned bearish (above price) and EMAs are bearish (EMA9 < EMA20)
            elif self.psar_direction_changed and current_psar_direction == -1 and emas_bearish:
                # Calculate price movement since PSAR direction change
                price_movement = None
                
                if hasattr(self, 'psar_direction_change_time') and self.psar_direction_change_time:
                    # Get a few candles ago to check price movement
                    candles_ago = min(5, len(df) - 1)  # At most 5 candles back
                    past_price = df.iloc[-candles_ago]['close']
                    price_movement = ((current_ask / past_price) - 1) * 100
                    self.console.print(f"Price movement since direction change: {price_movement:.2f}%")
                
                # Ensure price is moving in the right direction
                if price_movement is None or price_movement < 0:  # Price moving down or undetermined
                    signal = "SELL"
                    reason = "PSAR turned bearish with EMAs aligned and price moving down"
                    
                    self.console.print(f"[bold red]✓ SELL SIGNAL TRIGGERED (PSAR Direction Change)![/bold red]")
                    self.console.print(f"[green]Will sell at: {current_bid}[/green]")
                    self.console.print(f"[yellow]Reason: {reason}[/yellow]")
                    
                    # Reset PSAR direction change flag to prevent multiple signals
                    self.psar_direction_changed = False
                    self.save_state()
                    
                    # Store reason for logging
                    self.last_signal_reason = reason
                    
                    return signal, reason
            
            # NEW CONDITION: Generate signal when PSAR is below Average PSAR range, even without EMA crossover
            # For bullish setup based on PSAR below Average PSAR range
            elif current_psar_direction == 1 and psar_below_avg:
                # Calculate price movement for short-term trend confirmation
                candles_ago = min(3, len(df) - 1)  # Check last 3 candles
                past_price = df.iloc[-candles_ago]['close']
                price_movement = ((current_bid / past_price) - 1) * 100
                
                self.console.print(f"Short-term price movement: {price_movement:.2f}%")
                
                # Ensure price is showing some upward momentum
                if price_movement > 0:  # Price moving up
                    signal = "BUY"
                    reason = "PSAR below Average PSAR range, showing strong bullish signal"
                    
                    self.console.print(f"[bold green]✓ BUY SIGNAL TRIGGERED (PSAR Below Avg Range)![/bold green]")
                    self.console.print(f"[green]Will buy at: {current_ask}[/green]")
                    self.console.print(f"[yellow]Reason: {reason}[/yellow]")
                    
                    # Store reason for logging
                    self.last_signal_reason = reason
                    
                    return signal, reason
            
            # For bearish setup based on PSAR above Average PSAR range
            elif current_psar_direction == -1 and psar_above_avg:
                # Calculate price movement for short-term trend confirmation
                candles_ago = min(3, len(df) - 1)  # Check last 3 candles
                past_price = df.iloc[-candles_ago]['close']
                price_movement = ((current_ask / past_price) - 1) * 100
                
                self.console.print(f"Short-term price movement: {price_movement:.2f}%")
                
                # Ensure price is showing some downward momentum
                if price_movement < 0:  # Price moving down
                    signal = "SELL"
                    reason = "PSAR above Average PSAR range, showing strong bearish signal"
                    
                    self.console.print(f"[bold red]✓ SELL SIGNAL TRIGGERED (PSAR Above Avg Range)![/bold red]")
                    self.console.print(f"[green]Will sell at: {current_bid}[/green]")
                    self.console.print(f"[yellow]Reason: {reason}[/yellow]")
                    
                    # Store reason for logging
                    self.last_signal_reason = reason
                    
                    return signal, reason
            
            # Check for classic EMA crossover signals (from original implementation)
            # Check for crossover - this is the original BTC bot logic
            bullish_cross = prev_row['ema_9'] <= prev_row['ema_20'] and last_row['ema_9'] > last_row['ema_20']
            bearish_cross = prev_row['ema_9'] >= prev_row['ema_20'] and last_row['ema_9'] < last_row['ema_20']
            
            if bullish_cross and not self.crossover_detected:
                self.crossover_detected = True
                self.crossover_type = "BUY"
                self.crossover_price = last_row['close']
                self.crossover_time = time.time()
                self.save_state()
                self.console.print("[yellow]Bullish EMA crossover detected - monitoring for confirmation[/yellow]")
            
            elif bearish_cross and not self.crossover_detected:
                self.crossover_detected = True
                self.crossover_type = "SELL"
                self.crossover_price = last_row['close']
                self.crossover_time = time.time()
                self.save_state()
                self.console.print("[yellow]Bearish EMA crossover detected - monitoring for confirmation[/yellow]")
            
            # If we have a crossover, check conditions
            if self.crossover_detected:
                # Critical fix: Convert crossover_time to float if it's a string
                if isinstance(self.crossover_time, str):
                    try:
                        self.crossover_time = float(self.crossover_time)
                    except (ValueError, TypeError):
                        self.crossover_time = time.time()
                        self.console.print("[yellow]Fixed invalid crossover time format[/yellow]")
                
                # Calculate time since crossover (now with the proper float type)
                time_since_crossover = time.time() - (self.crossover_time or 0)
                
                # If crossover is too old (3 hours), reset it
                if time_since_crossover > 10800:  # 3 hours
                    self.console.print("[yellow]Crossover expired - resetting[/yellow]")
                    self.crossover_detected = False
                    self.save_state()
                    return None, None
                
                price_movement = ((last_row['close'] - self.crossover_price) / self.crossover_price) * 100
                min_psar_distance = last_row['avg_psar_distance'] * self.psar_distance_multiplier
                
                self.console.print(f"Price Movement since crossover: {price_movement:.2f}%")
                self.console.print(f"Time since crossover: {time_since_crossover:.1f}s")
                
                # BUY conditions
                if self.crossover_type == "BUY":
                    if (last_row['psar_direction'] == 1 and  # PSAR is bullish
                        price_movement >= self.price_movement_threshold and  # Price moved up enough
                        psar_distance >= min_psar_distance):  # PSAR not too close
                        
                        self.crossover_detected = False
                        self.save_state()
                        self.console.print("[bold green]✓ BUY SIGNAL CONFIRMED![/bold green]")
                        return "BUY", "EMA cross with PSAR confirmation"
                
                # SELL conditions
                elif self.crossover_type == "SELL":
                    if (last_row['psar_direction'] == -1 and  # PSAR is bearish
                        price_movement <= -self.price_movement_threshold and  # Price moved down enough
                        psar_distance >= min_psar_distance):  # PSAR not too close
                        
                        self.crossover_detected = False
                        self.save_state()
                        self.console.print("[bold red]✓ SELL SIGNAL CONFIRMED![/bold red]")
                        return "SELL", "EMA cross with PSAR confirmation"
            
            return None, None
            
        except Exception as e:
            self.console.print(f"[red]Error getting trading signal: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
            
            # If the error is related to crossover_time, fix it and save
            if "crossover_time" in str(e):
                self.crossover_time = time.time()
                self.save_state()
                self.console.print("[yellow]Fixed crossover_time and saved state[/yellow]")
            
            return None, None

    def run(self):
        """Main bot loop"""
        self.console.print("[bold green]Starting BTC EMA-PSAR Bot...[/bold green]")
        
        # Initialize for partial exits tracking
        self.partial_exits_taken = {}
        self.last_position_update = 0  # Track when we last updated position info
        self.last_mt5_check = time.time()  # Track when we last checked MT5 connection
        self.consecutive_errors = 0  # Track consecutive errors for backoff
        
        # Make sure we have a valid active symbol
        if not self._find_valid_btc_symbol():
            self.console.print("[yellow]Could not find main BTC symbol, attempting to use a placeholder...[/yellow]")
            
            # If we can't find a working symbol, set a placeholder but continue
            self.active_symbol = self.symbol
            self.console.print(f"[yellow]Using placeholder symbol: {self.active_symbol}[/yellow]")
            self.console.print("[yellow]Will attempt to work with symbol data even if unavailable in broker[/yellow]")
        else:
            self.console.print(f"[green]Using symbol: {self.active_symbol}[/green]")
        
        # For debugging: add a main loop counter
        loop_count = 0
        
        # Force garbage collection at start
        gc.collect()
        
        while self.running:
            try:
                # For debugging: track how many times we've gone through the main loop
                loop_count += 1
                if loop_count % 10 == 0:  # Log every 10 iterations
                    self.console.print(f"[cyan]Main loop iteration #{loop_count}[/cyan]")
                    # Force garbage collection every 10 iterations
                    gc.collect()
                
                # Check MT5 connection periodically (every 5 minutes)
                current_time = time.time()
                if current_time - self.last_mt5_check > 300:  # 5 minutes
                    self.last_mt5_check = current_time
                    
                    # Check if MT5 is still connected
                    if not mt5.terminal_info():
                        self.console.print("[yellow]MT5 connection lost, attempting to reconnect...[/yellow]")
                        mt5.shutdown()
                        time.sleep(3)
                        success = self.initialize_mt5()
                        if success:
                            self.console.print("[green]Successfully reconnected to MT5[/green]")
                        else:
                            self.console.print("[red]Failed to reconnect to MT5, will retry...[/red]")
                            time.sleep(30)  # Wait longer before next attempt
                            continue
                
                # Get market data
                self.console.print("[dim]Getting market data from MT5...[/dim]")
                df = self.get_market_data()
                if df is None:
                    self.console.print("[yellow]Failed to get market data, waiting before retry...[/yellow]")
                    time.sleep(5)
                    continue
                
                # Calculate indicators
                self.console.print("[dim]Calculating indicators...[/dim]")
                df = self.calculate_indicators(df)
                if df is None:
                    self.console.print("[yellow]Failed to calculate indicators, waiting before retry...[/yellow]")
                    time.sleep(5)
                    continue
                
                # Check for open positions first
                self.console.print("[dim]Checking for open positions...[/dim]")
                
                # Add extra error handling for positions_get
                try:
                    positions = mt5.positions_get(symbol=self.symbol)
                    if positions is None:
                        error_code = mt5.last_error()
                        self.console.print(f"[yellow]Error getting positions: {error_code}, will retry...[/yellow]")
                        time.sleep(5)
                        continue
                except Exception as pos_error:
                    self.console.print(f"[red]Exception in positions_get: {str(pos_error)}[/red]")
                    import traceback
                    self.console.print(f"[red]{traceback.format_exc()}[/red]")
                    time.sleep(5)
                    continue
                    
                current_time = time.time()
                
                # Display detailed position information every 15 seconds
                if positions and (current_time - self.last_position_update > 15):
                    self.last_position_update = current_time
                    
                    for position in positions:
                        self.display_position_status(position, df)
                
                # For debugging: log whether we have positions
                if positions:
                    self.console.print(f"[dim]Found {len(positions)} open positions. Processing...[/dim]")
                
                if positions:
                    # We have open positions, check exit conditions
                    for position in positions:
                        try:
                            self.console.print(f"[dim]Checking exit conditions for position #{position.ticket}...[/dim]")
                            
                            # Check exit conditions but don't terminate the process
                            # check_exit_conditions now returns tuple: (action_type, exit_portion, reason)
                            exit_result = self.check_exit_conditions(df, position)
                            
                            # Log the result of checking exit conditions
                            self.console.print(f"[dim]Exit check result: exit_result = {exit_result}[/dim]")
                            
                            action_taken = False
                            if exit_result[0] is not None:
                                action_type, exit_portion, reason = exit_result
                                
                                if action_type == "PARTIAL_EXIT":
                                    self.console.print(f"[yellow]Executing partial exit ({exit_portion*100}%) due to: {reason}[/yellow]")
                                    exit_success = self.execute_partial_exit(position, exit_portion, reason)
                                    if exit_success:
                                        self.console.print("[green]Partial exit executed successfully[/green]")
                                        
                                        # Track this partial exit in our internal state
                                        pos_id = position.ticket
                                        if pos_id not in self.partial_exits_taken:
                                            self.partial_exits_taken[pos_id] = []
                                        
                                        self.partial_exits_taken[pos_id].append({
                                            "exit_percentage": exit_portion,
                                            "time": datetime.now(),
                                            "reason": reason
                                        })
                                        
                                        self.save_state()  # Save state to persist between restarts
                                        action_taken = True
                                    else:
                                        self.console.print("[red]Partial exit execution failed![/red]")
                                
                                elif action_type == "EXIT":
                                    self.console.print(f"[yellow]Executing full exit due to: {reason}[/yellow]")
                                    exit_success = self.close_position(position, reason)
                                    if exit_success:
                                        self.console.print("[green]Full exit executed successfully[/green]")
                                        action_taken = True
                                    else:
                                        self.console.print("[red]Full exit execution failed![/red]")
                            
                            # If no action was taken for the main position, check for counter-trend opportunities
                            if not action_taken:
                                counter_signal = self.check_for_counter_trend_opportunity(df, position)
                                
                                if counter_signal:
                                    self.console.print(f"[bold]Detected counter-trend {counter_signal} opportunity against position #{position.ticket}[/bold]")
                                    counter_result = self.execute_counter_trend_trade(counter_signal, df, position)
                                    
                                    if counter_result:
                                        self.console.print(f"[green]Successfully executed counter-trend {counter_signal} trade[/green]")
                                    else:
                                        self.console.print(f"[yellow]Failed to execute counter-trend {counter_signal} trade[/yellow]")
                            
                            # Log any actions that were taken
                            if action_taken:
                                self.console.print("[green]Exit condition checked and handled[/green]")
                                
                                # Refresh positions after an action to get the latest state
                                self.console.print("[dim]Refreshing positions after action...[/dim]")
                                try:
                                    updated_positions = mt5.positions_get(symbol=self.symbol)
                                    
                                    # Log the updated positions count
                                    if updated_positions is None:
                                        self.console.print("[yellow]Failed to get updated positions, will continue...[/yellow]")
                                    else:
                                        self.console.print(f"[dim]After action: Found {len(updated_positions)} open positions[/dim]")
                                    
                                    # If no positions left and we took an action, this means a full close occurred
                                    if (updated_positions is None or len(updated_positions) == 0) and len(positions) > 0:
                                        self.console.print("[bold green]All positions closed![/bold green]")
                                        # Don't exit the loop - keep looking for new opportunities
                                except Exception as refresh_error:
                                    self.console.print(f"[red]Error refreshing positions: {str(refresh_error)}[/red]")
                                
                        except Exception as pos_e:
                            # Handle errors for a specific position without stopping the entire loop
                            self.console.print(f"[red]Error checking exit for position {position.ticket}: {str(pos_e)}[/red]")
                            import traceback
                            self.console.print(f"[red]{traceback.format_exc()}[/red]")
                else:
                    # No open positions, look for new signals
                    self.console.print("[dim]No open positions. Looking for new signals...[/dim]")
                    try:
                        signal, reason = self.get_trading_signal(df)
                        
                        # Execute trade if signal exists
                        if signal:
                            self.execute_trade(signal, df)
                    except Exception as sig_e:
                        # Handle errors in signal generation without stopping the entire loop
                        self.console.print(f"[red]Error in signal generation: {str(sig_e)}[/red]")
                        import traceback
                        self.console.print(f"[red]{traceback.format_exc()}[/red]")
                
                # Reset consecutive errors counter on successful execution
                self.consecutive_errors = 0
                
                # Check and update counter-trend positions status
                if self.counter_trend_active:
                    self.check_counter_trend_positions()
                
                # Sleep for a short time to avoid hammering the system
                self.console.print("[dim]Sleeping before next iteration...[/dim]")
                
                # Clear dataframe reference and force garbage collection
                df = None
                gc.collect()
                
                time.sleep(1)
                
            except KeyboardInterrupt:
                self.console.print("[yellow]Bot shutdown requested via KeyboardInterrupt...[/yellow]")
                self.running = False
                self.save_state()  # Save state before shutdown
                
            except Exception as e:
                # Handle general errors in the main loop
                self.consecutive_errors += 1
                self.console.print(f"[red]Error in main loop: {str(e)}[/red]")
                import traceback
                self.console.print(f"[red]{traceback.format_exc()}[/red]")
                
                # Use exponential backoff for repeated errors
                wait_time = min(60, 5 * (2 ** min(self.consecutive_errors - 1, 5)))
                self.console.print(f"[yellow]Waiting {wait_time} seconds before retry (error #{self.consecutive_errors})[/yellow]")
                
                # Try to save state even if there's an error
                try:
                    self.save_state()
                except:
                    pass
                    
                time.sleep(wait_time)
                
                # If persistent errors, try to reconnect to MT5
                if self.consecutive_errors >= 5:
                    self.console.print("[yellow]Multiple consecutive errors, attempting MT5 reconnection...[/yellow]")
                    try:
                        mt5.shutdown()
                        time.sleep(3)
                        self.initialize_mt5()
                    except:
                        self.console.print("[red]Failed to reinitialize MT5[/red]")
                
                # Force garbage collection
                gc.collect()
        
        self.console.print("[yellow]Bot run loop has exited. This should only happen on intentional shutdown.[/yellow]")

    def get_market_data(self):
        """Get market data from MT5"""
        try:
            self.console.print("[dim]Requesting market data from MT5...[/dim]")
            
            # Force garbage collection before fetching data
            gc.collect()
            
            # Make sure we're using the active symbol that works
            if self.active_symbol is None:
                if not self._find_valid_btc_symbol():
                    # If we still can't find a working symbol, use the default symbol but warn
                    self.active_symbol = self.symbol
                    self.console.print(f"[yellow]Using default symbol {self.active_symbol} despite issues[/yellow]")
                    
            symbol_to_use = self.active_symbol
            self.console.print(f"[dim]Using symbol: {symbol_to_use}[/dim]")
            
            # Request a reduced amount of data to prevent memory issues
            # 2000 candles should be more than enough for PSAR and EMA calculations
            rates = mt5.copy_rates_from_pos(symbol_to_use, self.timeframe, 0, 2000)
            
            if rates is None:
                error_code = mt5.last_error()
                self.console.print(f"[red]Failed to get market data. MT5 error code: {error_code}[/red]")
                
                # If we can't get data for the active symbol, try a test symbol to check connection
                test_symbol = "EURUSD"  # This is almost always available
                test_rates = mt5.copy_rates_from_pos(test_symbol, self.timeframe, 0, 10)
                
                if test_rates is not None:
                    self.console.print(f"[yellow]Connection works with {test_symbol} but {symbol_to_use} is unavailable[/yellow]")
                    
                    # Create a synthetic empty dataframe with minimal required columns for testing
                    self.console.print("[yellow]Creating synthetic data for testing purposes[/yellow]")
                    
                    # Create a minimal synthetic dataframe for testing purposes
                    now = datetime.now()
                    synthetic_data = []
                    
                    for i in range(200):  # Create 200 candles
                        candle_time = now - timedelta(minutes=15 * i)
                        # Simple oscillation around 65000 for testing
                        close_price = 65000 + 500 * math.sin(i / 10)
                        # Create some variation in high/low
                        high_price = close_price + random.uniform(50, 150)
                        low_price = close_price - random.uniform(50, 150)
                        synthetic_data.append({
                            'time': candle_time.timestamp(),
                            'open': close_price - random.uniform(-100, 100),
                            'high': high_price,
                            'low': low_price,
                            'close': close_price,
                            'tick_volume': random.randint(100, 1000)
                        })
                    
                    # Convert to dataframe
                    df = pd.DataFrame(synthetic_data)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    
                    self.console.print("[yellow]Created synthetic data for testing[/yellow]")
                    return df
                
                return None
            
            self.console.print(f"[dim]Received {len(rates)} candles from MT5[/dim]")
                
                # Check if we have enough data
            if len(rates) < 100:  # At least 100 candles needed for reliable indicators
                self.console.print(f"[yellow]Warning: Only {len(rates)} candles received - may not be enough for accurate indicators[/yellow]")
                
                # Convert to dataframe more efficiently
            try:
                    # Create dataframe with only the columns we need
                df = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
                df['time'] = pd.to_datetime(df['time'], unit='s')
            except Exception as df_error:
                self.console.print(f"[red]Error creating dataframe: {str(df_error)}[/red]")
                import traceback
                self.console.print(f"[red]{traceback.format_exc()}[/red]")
                return None
                
                # Force garbage collection to prevent memory leaks
            gc.collect()
            
            return df
        except Exception as e:
            self.console.print(f"[red]Error in get_market_data: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
            return None

    def log_trade(self, trade_data):
        """Log trade information to file"""
        try:
            # Add timestamp to trade data
            trade_data["timestamp"] = datetime.now().isoformat()
            
            # Create an array in the log file, or append to it if it exists
            if os.path.exists(self.trade_log_file):
                try:
                    with open(self.trade_log_file, 'r') as f:
                        logs = json.load(f)
                        if not isinstance(logs, list):
                            logs = [logs]  # Convert to list if not already
                except json.JSONDecodeError:
                    logs = []  # Start fresh if file is corrupted
            else:
                logs = []
            
            # Append new trade data
            logs.append(trade_data)
            
            # Write updated logs back to file
            with open(self.trade_log_file, 'w') as f:
                json.dump(logs, f, indent=2, cls=NumpyEncoder)
                
            self.console.print(f"[green]Trade logged successfully[/green]")
        except Exception as e:
            self.console.print(f"[red]Error logging trade: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")

    def execute_trade(self, signal, df):
        """Execute trade based on signal"""
        try:
            last_row = df.iloc[-1]
            
            # Get current price for entry
            symbol_to_use = self.active_symbol or self.symbol
            tick_info = mt5.symbol_info_tick(symbol_to_use)
            
            if tick_info is None:
                self.console.print(f"[red]Could not get tick information for {symbol_to_use}[/red]")
                return False
                
            price = tick_info.ask if signal == "BUY" else tick_info.bid
            
            # Calculate PSAR-based stop loss distance (1.5x the PSAR distance for safety)
            sl_distance_pct = last_row['psar_distance'] * 1.5
            
            # Ensure minimum SL distance is at least 1%
            sl_distance_pct = max(sl_distance_pct, 1.0)
            
            # Make sure SL distance is positive
            sl_distance_pct = abs(sl_distance_pct)
            
            # Calculate stop loss price - ALWAYS place SL in the correct direction
            if signal == "BUY":
                # For BUY orders, SL must be BELOW entry price
                sl_price = price * (1 - sl_distance_pct/100)
            else:  # SELL
                # For SELL orders, SL must be ABOVE entry price
                sl_price = price * (1 + sl_distance_pct/100)
                
            sl_price = round(sl_price, 2)  # Round to 2 decimal places
            
            # Calculate multiple exit points (percentage-based)
            exit1_pct = 0.3  # First exit at 0.3%
            exit2_pct = 0.5  # Second exit at 0.5% 
            exit3_pct = 0.7  # Third exit at 0.7%
            
            # Calculate take profit levels for future use in our tracking
            tp1_price = price * (1 + exit1_pct/100) if signal == "BUY" else price * (1 - exit1_pct/100)
            tp2_price = price * (1 + exit2_pct/100) if signal == "BUY" else price * (1 - exit2_pct/100)
            tp3_price = price * (1 + exit3_pct/100) if signal == "BUY" else price * (1 - exit3_pct/100)
            
            tp1_price = round(tp1_price, 2)
            tp2_price = round(tp2_price, 2)
            tp3_price = round(tp3_price, 2)
            
            # Double-check that SL is in the correct direction relative to entry price
            if signal == "BUY" and sl_price >= price:
                self.console.print(f"[red]Error: Stop loss ({sl_price}) must be below entry price ({price}) for BUY orders[/red]")
                # Correct the stop loss to be below entry price
                sl_price = price * 0.99  # Set SL 1% below entry price
                self.console.print(f"[yellow]Corrected stop loss to {sl_price:.2f}[/yellow]")
            elif signal == "SELL" and sl_price <= price:
                self.console.print(f"[red]Error: Stop loss ({sl_price}) must be above entry price ({price}) for SELL orders[/red]")
                # Correct the stop loss to be above entry price
                sl_price = price * 1.01  # Set SL 1% above entry price
                self.console.print(f"[yellow]Corrected stop loss to {sl_price:.2f}[/yellow]")
            
            # Display the trade setup
            self.console.print("[bold cyan]Trade Setup:[/bold cyan]")
            self.console.print(f"Signal: [bold]{'BUY' if signal == 'BUY' else 'SELL'}[/bold]")
            self.console.print(f"Entry Price: {price:.2f}")
            self.console.print(f"Stop Loss: {sl_price:.2f} ({sl_distance_pct:.2f}%)")
            self.console.print(f"Take Profit 1 (30%): {tp1_price:.2f} ({exit1_pct}%)")
            self.console.print(f"Take Profit 2 (30%): {tp2_price:.2f} ({exit2_pct}%)")
            self.console.print(f"Take Profit 3 (40%): {tp3_price:.2f} ({exit3_pct}%)")
            
            # Record the crossover price and time
            self.crossover_detected = True
            self.crossover_price = price
            self.crossover_time = time.time()
            self.save_state()
            
            # Determine order type
            order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
            
            # Create the trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol_to_use,
                "volume": self.lot_size,  # Using the smaller lot size from __init__
                "type": order_type,
                "price": price,
                "sl": sl_price,  # Set the stop loss
                "deviation": 20,
                "magic": 234000,
                "comment": "BTC-BOT",  # Simplified comment
                "type_filling": mt5.ORDER_FILLING_IOC,
                "type_time": mt5.ORDER_TIME_GTC
            }
            
            # Log the trade request
            self.console.print(f"[yellow]Sending order: {request}[/yellow]")
            
            # Execute the trade
            result = mt5.order_send(request)
            
            # Handle potential None result
            if result is None:
                error_code = mt5.last_error()
                error_message = f"MT5 order_send returned None. Error code: {error_code}"
                self.console.print(f"[red]{error_message}[/red]")
                
                # Log the failed trade attempt
                trade_data = {
                    "action": "ENTRY_FAILED",
                    "signal": signal,
                    "attemptedPrice": price,
                    "error": error_message,
                    "errorCode": error_code,
                    "timestamp": datetime.now().isoformat()
                }
                self.log_trade(trade_data)
                return False
            
            # Check if trade was executed successfully
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_message = f"Order failed: {result.comment} (code {result.retcode})"
                self.console.print(f"[red]{error_message}[/red]")
                
                # Log the failed trade
                trade_data = {
                    "action": "ENTRY_FAILED",
                    "signal": signal,
                    "attemptedPrice": price,
                    "error": error_message,
                    "errorCode": result.retcode,
                    "timestamp": datetime.now().isoformat()
                }
                self.log_trade(trade_data)
                return False
            else:
                # Get the new position's ticket (ID)
                deal_ticket = result.order
                
                self.console.print(f"[green]Trade executed successfully! Ticket: {deal_ticket}[/green]")
                
                # Log the successful trade
                trade_data = {
                    "action": "ENTRY",
                    "signal": signal,
                    "price": price,
                    "stopLoss": sl_price,
                    "riskPercent": sl_distance_pct,
                    "ticket": deal_ticket,
                    "tp1": tp1_price,
                    "tp2": tp2_price,
                    "tp3": tp3_price,
                    "lotSize": self.lot_size,
                    "timestamp": datetime.now().isoformat(),
                    "psarDistance": f"{last_row['psar_distance']:.2f}%",
                    "psarValue": float(last_row['psar']),
                    "emaFast": float(last_row['ema_9']),
                    "emaSlow": float(last_row['ema_20'])
                }
                self.log_trade(trade_data)
                
                return True
            
        except Exception as e:
            self.console.print(f"[red]Error executing trade: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
            return False

    def check_exit_conditions(self, df, position):
        """Check for exit conditions"""
        try:
            # Use the active symbol for exit conditions
            symbol_to_use = self.active_symbol if self.active_symbol else self.symbol
            
            # Get current price data
            current_row = df.iloc[-1]
            
            # Get MT5 current price (more accurate than OHLC)
            tick = mt5.symbol_info_tick(symbol_to_use)
            
            if tick is None:
                # Use dataframe price if we can't get tick data (for testing)
                self.console.print(f"[yellow]Couldn't get current price for {symbol_to_use}, using dataframe price[/yellow]")
                
                if position.type == 0:  # BUY position
                    current_price = current_row['close']
                else:  # SELL position
                    current_price = current_row['close']
            else:
                # Use actual bid/ask price from MT5
                if position.type == 0:  # BUY position
                    current_price = tick.bid  # We sell at bid price when closing buy positions
                else:  # SELL position
                    current_price = tick.ask  # We buy at ask price when closing sell positions
            
            # Get PSAR direction and value
            psar = current_row['psar']
            psar_direction = current_row['psar_direction']
            
            # Extract original position open price
            open_price = position.price_open
            
            # Calculate profit percentage
            if position.type == 0:  # BUY position
                profit_percent = ((current_price / open_price) - 1) * 100
            else:  # SELL position
                profit_percent = ((open_price / current_price) - 1) * 100
            
            # Print status with more information
            position_type = "LONG" if position.type == 0 else "SHORT"
            self.console.print("\n[bold]Position Status[/bold]")
            self.console.print(f"Type: {position_type}")
            self.console.print(f"Open Price: {open_price:.2f}")
            self.console.print(f"Current Price: {current_price:.2f}")
            self.console.print(f"Profit/Loss: {profit_percent:.2f}%")
            self.console.print(f"PSAR: {psar:.2f}")
            self.console.print(f"PSAR Direction: {psar_direction}")
            
            # Get position ticket for tracking partial exits
            position_ticket = position.ticket
            
            # Check if we've already taken partial exits for this position
            taken_exits = []
            if hasattr(self, 'partial_exits_taken') and position_ticket in self.partial_exits_taken:
                taken_exits = [exit_data["exit_percentage"] for exit_data in self.partial_exits_taken[position_ticket]]
                
                # Show which partial exits were already taken
                self.console.print(f"Partial exits already taken: {[f'{exit*100:.1f}%' for exit in taken_exits]}")
            
            # For backward compatibility with older state files
            position_key = str(position.ticket)
            if hasattr(self, 'partial_exits_taken') and isinstance(self.partial_exits_taken, dict) and \
               position_key in self.partial_exits_taken and isinstance(self.partial_exits_taken[position_key], set):
                # Legacy format using 'exit1', 'exit2' strings
                legacy_exits = self.partial_exits_taken[position_key]
                if 'exit1' in legacy_exits:
                    taken_exits.append(0.3)
                if 'exit2' in legacy_exits:
                    taken_exits.append(0.5)
                if 'exit3' in legacy_exits:
                    taken_exits.append(0.7)
                if 'exit_full' in legacy_exits:  # Complete exit marker
                    taken_exits.append(1.0)
                
                # Convert to new format
                self.partial_exits_taken[position_ticket] = [
                    {"exit_percentage": exit_value, "time": datetime.now(), "reason": "Migrated from legacy format"}
                    for exit_value in taken_exits
                ]
                
                # Clean up old format
                if position_key != position_ticket:
                    del self.partial_exits_taken[position_key]
                
                self.console.print("[yellow]Migrated partial exit tracking to new format[/yellow]")
            
            # Logic for LONG position exit
            if position.type == 0:  # LONG position
                
                # Check for partial exit at 0.3% profit if not already taken
                if profit_percent >= 0.3 and 0.3 not in taken_exits:
                    self.console.print("[yellow]Exit condition met: 0.3% partial profit target reached[/yellow]")
                    # Take 30% partial profit
                    return "PARTIAL_EXIT", 0.3, "0.3% Partial Take Profit"
                
                # Check for partial exit at 0.5% profit if not already taken
                if profit_percent >= 0.5 and 0.5 not in taken_exits:
                    self.console.print("[yellow]Exit condition met: 0.5% partial profit target reached[/yellow]")
                    # Take 50% partial profit (of the remaining position)
                    return "PARTIAL_EXIT", 0.5, "0.5% Partial Take Profit"
                
                # Technical exit condition: PSAR above the close price (trend reversal)
                # Only if we've already taken partial exits
                if psar_direction == "Bearish" and taken_exits:
                    self.console.print("[yellow]Exit condition met: PSAR turned bearish after taking partial exits[/yellow]")
                    return "EXIT", 1.0, "PSAR Trend Reversal Exit"
            
            # Logic for SHORT position exit
            else:  # SHORT position
                
                # Check for partial exit at 0.3% profit if not already taken
                if profit_percent >= 0.3 and 0.3 not in taken_exits:
                    self.console.print("[yellow]Exit condition met: 0.3% partial profit target reached[/yellow]")
                    # Take 30% partial profit
                    return "PARTIAL_EXIT", 0.3, "0.3% Partial Take Profit"
                
                # Check for partial exit at 0.5% profit if not already taken
                if profit_percent >= 0.5 and 0.5 not in taken_exits:
                    self.console.print("[yellow]Exit condition met: 0.5% partial profit target reached[/yellow]")
                    # Take 50% partial profit (of the remaining position)
                    return "PARTIAL_EXIT", 0.5, "0.5% Partial Take Profit"
                
                # Technical exit condition: PSAR below the close price (trend reversal)
                # Only if we've already taken partial exits
                if psar_direction == "Bullish" and taken_exits:
                    self.console.print("[yellow]Exit condition met: PSAR turned bullish after taking partial exits[/yellow]")
                    return "EXIT", 1.0, "PSAR Trend Reversal Exit"
            
            # No exit conditions met
            self.console.print("[green]No exit conditions met. Holding position.[/green]")
            return None, 0, None
        
        except Exception as e:
            self.console.print(f"[red]Error in check_exit_conditions: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
            return None, 0, None

    def execute_partial_exit(self, position, exit_portion, reason):
        """Execute a partial position exit"""
        try:
            self.console.print(f"[bold yellow]Attempting to execute partial exit ({exit_portion*100}%) for position #{position.ticket}[/bold yellow]")
            
            # Make sure we're using the active symbol
            symbol_to_use = self.active_symbol or self.symbol
            
            # First, get the full details of the current position
            current_positions = mt5.positions_get(ticket=position.ticket)
            if not current_positions:
                self.console.print("[yellow]Position not found. It may have been closed already.[/yellow]")
                return False
                
            # Get original position details
            original_position = current_positions[0]
            original_volume = original_position.volume
            self.console.print(f"[dim]Original position size: {original_volume} lots[/dim]")
            
            # Calculate volume to close
            volume_to_close = original_volume * exit_portion
            
            # Get symbol info to check minimum and step constraints
            symbol_info = mt5.symbol_info(symbol_to_use)
            if symbol_info is None:
                self.console.print("[red]Failed to get symbol info, cannot determine volume constraints[/red]")
                # Try anyway with default settings
            else:
                # Check broker's minimum and step constraints
                min_volume = symbol_info.volume_min
                volume_step = symbol_info.volume_step
                
                self.console.print(f"[dim]Broker constraints: Min volume={min_volume}, Volume step={volume_step}[/dim]")
                
                # Ensure volume meets broker's requirements
                if volume_step > 0:
                    # Round to the nearest valid step
                    volume_to_close = round(volume_to_close / volume_step) * volume_step
                    
                # Check if calculated volume is below minimum
                if min_volume is not None and volume_to_close < min_volume:
                    volume_to_close = min_volume
                    self.console.print(f"[yellow]⚠️ Volume adjusted to broker minimum: {volume_to_close} lots[/yellow]")
                    
                    # Check if minimum lot size would effectively close the whole position
                    remaining_after_exit = original_volume - volume_to_close
                    if remaining_after_exit < min_volume:
                        self.console.print(f"[yellow]⚠️ Remaining size ({remaining_after_exit} lots) would be below minimum ({min_volume} lots)[/yellow]")
                        self.console.print("[yellow]This will effectively close the entire position[/yellow]")
                
                # Ensure volume doesn't exceed position size
                if volume_to_close > original_volume:
                    volume_to_close = original_volume
                    self.console.print("[yellow]⚠️ Volume adjusted to match full position size[/yellow]")
            
            # Get current price for the exit
            tick = mt5.symbol_info_tick(symbol_to_use)
            if tick is None:
                self.console.print("[red]Failed to get symbol tick info, cannot execute partial exit[/red]")
                return False
                
            price = tick.bid if original_position.type == mt5.ORDER_TYPE_BUY else tick.ask
            
            self.console.print(f"[yellow]Executing partial exit: {volume_to_close} lots of {original_volume} lots ({(volume_to_close/original_volume)*100:.1f}%) at {price}...[/yellow]")
            
            # Create the order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol_to_use,
                "volume": volume_to_close,
                "type": mt5.ORDER_TYPE_SELL if original_position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": original_position.ticket,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "PartialExit",  # Simplified comment - MT5 may have restrictions on comment format
                "type_filling": mt5.ORDER_FILLING_IOC,
                "type_time": mt5.ORDER_TIME_GTC
            }
            
            self.console.print(f"[dim]Sending partial exit request: {request}[/dim]")
            
            # Send the order request
            result = mt5.order_send(request)
            
            # Handle case when result is None
            if result is None:
                error_code = mt5.last_error()
                self.console.print(f"[red]MT5 order_send returned None. Error code: {error_code}[/red]")
                
                # Try to get updated position info
                updated_positions = mt5.positions_get(ticket=position.ticket)
                if not updated_positions:
                    self.console.print("[yellow]Position no longer exists - may have been closed by another process[/yellow]")
                    return True
                    
                self.console.print("[red]Failed to execute partial exit[/red]")
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.console.print(f"[red]Partial exit failed: {result.comment} (code {result.retcode})[/red]")
                
                # Additional diagnostics for volume errors
                if result.retcode == 10014:  # Invalid volume error
                    # Get updated symbol info
                    symbol_info = mt5.symbol_info(symbol_to_use)
                    if symbol_info is not None:
                        self.console.print(f"[yellow]Broker requires: Min volume: {symbol_info.volume_min}, Step: {symbol_info.volume_step}[/yellow]")
                        self.console.print(f"[yellow]We tried: {volume_to_close} lots[/yellow]")
                        
                        # Check if minimum lot size equals or exceeds position size
                        if symbol_info.volume_min >= original_volume:
                            self.console.print("[yellow]⚠️ Current position size is already at broker minimum. Cannot execute partial exit.[/yellow]")
                            self.console.print("[yellow]To exit, the position must be closed completely.[/yellow]")
                            
                            # Ask user for confirmation to close completely
                            self.console.print("[bold yellow]Would you like to close the entire position instead? (Automatically proceeding in 10 seconds)[/bold yellow]")
                            
                            # No interactivity - just proceed with full close
                            time.sleep(10)
                            return self.close_position(original_position, reason + " (adjusted to full exit due to broker minimum lot size)")
                
                return False
            else:
                # Get updated position info to verify the partial exit worked
                time.sleep(1)  # Wait a moment for the broker to process
                updated_positions = mt5.positions_get(ticket=position.ticket)
                
                if not updated_positions:
                    # Position closed completely, even though we requested partial
                    self.console.print("[yellow]Position was closed completely, even though partial exit was requested[/yellow]")
                    
                    # Log the complete exit
                    exit_data = {
                        "action": "FULL_EXIT",
                        "ticket": original_position.ticket,
                        "volume": original_volume,
                        "entryPrice": original_position.price_open,
                        "exitPrice": price,
                        "profit": original_position.profit,
                        "exitReason": reason + " (became full exit)",
                        "status": "FULL_CLOSE"
                    }
                    self.log_trade(exit_data)
                    
                    position_key = str(original_position.ticket)
                    if position_key in self.partial_exits_taken:
                        self.partial_exits_taken[position_key].add("exit_full")
                    
                    self.save_state()
                    return True
                
                new_position = updated_positions[0]
                new_volume = new_position.volume
                volume_change = original_volume - new_volume
                
                if abs(volume_change) < 0.0001:  # No actual change in position size
                    self.console.print("[red]⚠️ MT5 reported success, but position size did not change![/red]")
                    self.console.print(f"[yellow]Original: {original_volume} lots, Current: {new_volume} lots[/yellow]")
                    self.console.print("[yellow]The partial exit may have failed due to broker-specific rules[/yellow]")
                    return False
                
                # Successful partial exit
                self.console.print(f"[green]✓ Partial exit successful: {volume_change} lots at {price} ({reason})[/green]")
                self.console.print(f"[green]Position size reduced from {original_volume} to {new_volume} lots[/green]")
                self.console.print(f"[green]Profit for this portion: ${original_position.profit * (volume_change/original_volume):.2f}[/green]")
                
                # Log partial exit
                exit_data = {
                    "action": "PARTIAL_EXIT",
                    "ticket": original_position.ticket,
                    "exitPortion": exit_portion,
                    "volume": volume_change,
                    "entryPrice": original_position.price_open,
                    "exitPrice": price,
                    "profit": original_position.profit * (volume_change/original_volume),
                    "exitReason": reason,
                    "status": "PARTIAL_CLOSE"
                }
                self.log_trade(exit_data)
                
                # Save state to ensure we don't lose track
                self.save_state()
                
                return True
                
        except Exception as e:
            self.console.print(f"[red]Error executing partial exit: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
            return False

    def close_position(self, position, reason):
        """Close an entire position"""
        try:
            # Get current price for closing
            price = mt5.symbol_info_tick(self.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask
            
            self.console.print(f"[yellow]Attempting to close position {position.ticket} at {price}...[/yellow]")
            
            # Sanitize the comment - this was causing the error
            # Use a simple comment without special characters
            comment = "Python Exit"
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": comment,  # Using simplified comment
                "type_filling": mt5.ORDER_FILLING_IOC,
                "type_time": mt5.ORDER_TIME_GTC
            }
            
            # Send the request and handle potential None result
            result = mt5.order_send(request)
            
            if result is None:
                error_code = mt5.last_error()
                self.console.print(f"[red]MT5 order_send returned None. Error code: {error_code}[/red]")
                
                # Check if position still exists
                positions = mt5.positions_get(ticket=position.ticket)
                if not positions:
                    self.console.print("[yellow]Position no longer exists - may have been closed already[/yellow]")
                    return True
                
                self.console.print("[red]Position still exists but could not be closed[/red]")
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.console.print(f"[red]Position close failed: {result.comment} (code {result.retcode})[/red]")
                return False
            else:
                current_price = mt5.symbol_info_tick(self.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask
                self.console.print(f"[green]Position closed: {position.volume} lots at {current_price}[/green]")
                self.console.print(f"[green]Reason: {reason}[/green]")
                self.console.print(f"[green]Profit: ${position.profit:.2f}[/green]")
                
                # Log full exit
                exit_data = {
                    "action": "FULL_EXIT",
                    "ticket": position.ticket,
                    "volume": position.volume,
                    "entryPrice": position.price_open,
                    "soldAt" if position.type == mt5.ORDER_TYPE_BUY else "boughtAt": current_price,
                    "profit": position.profit,
                    "exitReason": reason,
                    "status": "CLOSED"
                }
                self.log_trade(exit_data)
                return True
                
        except Exception as e:
            self.console.print(f"[red]Error closing position: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
            return False

    def display_position_status(self, position, df):
        """Display detailed information about an open position"""
        try:
            # Get current price
            current_price = mt5.symbol_info_tick(self.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask
            
            # Calculate profit
            entry_price = position.price_open
            profit_pips = (current_price - entry_price) if position.type == mt5.ORDER_TYPE_BUY else (entry_price - current_price)
            profit_percentage = (profit_pips / entry_price) * 100
            
            # Determine which partial exits have been taken
            position_key = str(position.ticket)
            if position_key not in self.partial_exits_taken:
                self.partial_exits_taken[position_key] = set()
            
            exits_taken = self.partial_exits_taken[position_key]
            
            # Calculate exit target prices
            if position.type == mt5.ORDER_TYPE_BUY:
                exit1_price = entry_price * (1 + 0.3/100)
                exit2_price = entry_price * (1 + 0.5/100)
                exit3_price = entry_price * (1 + 0.7/100)
            else:
                exit1_price = entry_price * (1 - 0.3/100)
                exit2_price = entry_price * (1 - 0.5/100)
                exit3_price = entry_price * (1 - 0.7/100)
            
            # Time elapsed since position opened
            time_opened = position.time
            time_elapsed = time.time() - time_opened
            hours = int(time_elapsed / 3600)
            minutes = int((time_elapsed % 3600) / 60)
            seconds = int(time_elapsed % 60)
            
            # Format detailed position status
            position_type = "LONG" if position.type == mt5.ORDER_TYPE_BUY else "SHORT"
            
            self.console.print("\n[bold cyan]════════ ACTIVE POSITION STATUS ════════[/bold cyan]")
            self.console.print(f"Symbol: [bold]{self.symbol}[/bold]")
            self.console.print(f"Position: [bold]{position_type}[/bold] | Ticket: {position.ticket}")
            self.console.print(f"Volume: [bold]{position.volume} lots[/bold]")
            self.console.print(f"Entry Price: [bold]{entry_price:.2f}[/bold] | Current Price: [bold]{current_price:.2f}[/bold]")
            
            # Color-coded profit display
            profit_color = "green" if profit_percentage > 0 else "red"
            self.console.print(f"Profit: [{profit_color}]{profit_percentage:.2f}%[/{profit_color}] (${position.profit:.2f})")
            
            # Show exit targets with status indicators
            self.console.print("\n[bold]Exit Targets:[/bold]")
            
            # Exit 1 status
            exit1_status = "[bold green]✓ TAKEN[/bold green]" if 1 in exits_taken else \
                          "[bold yellow]PENDING[/bold yellow]" if profit_percentage < 0.3 else \
                          "[bold cyan]READY[/bold cyan]"
            exit1_progress = min(profit_percentage / 0.3 * 100, 100) if profit_percentage > 0 else 0
            self.console.print(f"Exit 1 (30%): {exit1_price:.2f} | Status: {exit1_status}")
            self.console.print(f"Progress: {exit1_progress:.1f}% {'▓' * int(exit1_progress/5)}{'░' * (20-int(exit1_progress/5))}")
            
            # Exit 2 status
            exit2_status = "[bold green]✓ TAKEN[/bold green]" if 2 in exits_taken else \
                          "[bold yellow]PENDING[/bold yellow]" if profit_percentage < 0.5 else \
                          "[bold cyan]READY[/bold cyan]"
            exit2_progress = min(profit_percentage / 0.5 * 100, 100) if profit_percentage > 0 else 0
            self.console.print(f"Exit 2 (30%): {exit2_price:.2f} | Status: {exit2_status}")
            self.console.print(f"Progress: {exit2_progress:.1f}% {'▓' * int(exit2_progress/5)}{'░' * (20-int(exit2_progress/5))}")
            
            # Exit 3 status
            exit3_status = "[bold green]✓ TAKEN[/bold green]" if 3 in exits_taken else \
                          "[bold yellow]PENDING[/bold yellow]" if profit_percentage < 0.7 else \
                          "[bold cyan]READY[/bold cyan]"
            exit3_progress = min(profit_percentage / 0.7 * 100, 100) if profit_percentage > 0 else 0
            self.console.print(f"Exit 3 (40%): {exit3_price:.2f} | Status: {exit3_status}")
            self.console.print(f"Progress: {exit3_progress:.1f}% {'▓' * int(exit3_progress/5)}{'░' * (20-int(exit3_progress/5))}")
            
            # Position time 
            self.console.print(f"\nPosition Age: [bold]{hours:02d}:{minutes:02d}:{seconds:02d}[/bold]")
            self.console.print(f"Stop Loss: {position.sl:.2f}")
            self.console.print("[bold cyan]═════════════════════════════════════[/bold cyan]\n")
            
        except Exception as e:
            self.console.print(f"[red]Error displaying position status: {str(e)}[/red]")

    def load_state(self):
        """Load the bot's state from a file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.crossover_detected = state.get('crossover_detected', False)
                    self.crossover_type = state.get('crossover_type', None)
                    self.crossover_price = state.get('crossover_price', None)
                    self.crossover_time = state.get('crossover_time', None)
                    self.partial_exits_taken = state.get('partial_exits_taken', {})
                    self.last_psar_direction = state.get('last_psar_direction', None)
                    self.psar_direction_changed = state.get('psar_direction_changed', False)
                    self.psar_direction_change_time = state.get('psar_direction_change_time', None)
                    
                    # Load counter-trend tracking
                    self.counter_trend_positions = state.get('counter_trend_positions', {})
                    self.counter_trend_active = state.get('counter_trend_active', False)
                    self.last_counter_trend_check = state.get('last_counter_trend_check', 0)
                    
                    # Convert string keys back to proper format for partial_exits_taken
                    if isinstance(self.partial_exits_taken, dict):
                        for position_id, exits in self.partial_exits_taken.items():
                            if isinstance(exits, list):
                                self.partial_exits_taken[position_id] = set(exits)
                
                self.console.print("[green]State loaded successfully[/green]")
        except Exception as e:
            self.console.print(f"[yellow]No previous state loaded: {str(e)}[/yellow]")
            # Initialize with default values
            self.crossover_detected = False
            self.crossover_type = None
            self.crossover_price = None
            self.crossover_time = None
            self.partial_exits_taken = {}
            self.last_psar_direction = None
            self.psar_direction_changed = False
            self.psar_direction_change_time = None
            self.counter_trend_positions = {}
            self.counter_trend_active = False
            self.last_counter_trend_check = 0

    def save_state(self):
        """Save the bot's state to a file"""
        try:
            # Convert sets to lists for JSON serialization
            partial_exits_json = {}
            for position_id, exits in self.partial_exits_taken.items():
                if isinstance(exits, set):
                    partial_exits_json[position_id] = list(exits)
                else:
                    partial_exits_json[position_id] = exits
            
            state = {
                'crossover_detected': self.crossover_detected,
                'crossover_type': self.crossover_type,
                'crossover_price': self.crossover_price,
                'crossover_time': self.crossover_time,
                'partial_exits_taken': partial_exits_json,
                'last_psar_direction': self.last_psar_direction,
                'psar_direction_changed': self.psar_direction_changed,
                'psar_direction_change_time': self.psar_direction_change_time,
                'counter_trend_positions': self.counter_trend_positions,
                'counter_trend_active': self.counter_trend_active,
                'last_counter_trend_check': self.last_counter_trend_check
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, cls=NumpyEncoder)
            
            self.console.print("[green]State saved successfully[/green]")
        except Exception as e:
            self.console.print(f"[red]Error saving state: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")

    def check_for_counter_trend_opportunity(self, df, position):
        """Check for counter-trend trading opportunities based on PSAR flips and EMA gap ROC.
        
        This method identifies counter-trend trading opportunities when:
        1. There's an existing profitable position
        2. PSAR direction is against the current position (not just flipped)
        3. EMA gap has been widening (strong trend) but is now narrowing (potential reversal)
        4. Price is showing momentum in the counter direction
        
        Args:
            df: DataFrame with price and indicator data
            position: The current open position
            
        Returns:
            str: Counter-trend signal ("BUY", "SELL", or None)
        """
        try:
            if df is None or len(df) < 4:  # Need at least 4 bars for ROC calculations
                return None
            
            # Get the latest data
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            # Current indicators
            current_price = last_row['close']
            prev_price = prev_row['close']
            ema_fast = last_row['ema_9']
            ema_slow = last_row['ema_20']
            psar = last_row['psar']
            psar_direction = last_row['psar_direction']
            prev_psar_direction = prev_row['psar_direction']
            
            # Check if we have the gap metrics
            if 'ema_gap' not in last_row or 'ema_gap_roc_3' not in last_row:
                self.console.print("[yellow]Missing EMA gap metrics, can't check for counter-trend opportunities[/yellow]")
                return None
            
            # Get the EMA gap metrics
            ema_gap = last_row['ema_gap']
            ema_gap_roc_3 = last_row['ema_gap_roc_3']
            gap_widening = last_row['gap_widening']
            
            # Calculate current profit percentage for the position
            entry_price = position.price_open
            current_position_price = position.price_current
            profit_pct = ((current_position_price / entry_price) - 1) * 100
            if position.type == mt5.POSITION_TYPE_SELL:
                profit_pct = -profit_pct
            
            # Initialize counter-trend signal to None
            counter_signal = None
            
            # Only consider counter-trend if the current position is profitable
            if profit_pct < 0.15:  # Minimum profit threshold
                self.console.print("[dim]Position not profitable enough for counter-trend strategy[/dim]")
                return None
            
            # Display counter-trend analysis
            self.console.print("\n=== Counter-Trend Analysis ===")
            self.console.print(f"Current Position: {'BUY' if position.type == mt5.POSITION_TYPE_BUY else 'SELL'}")
            self.console.print(f"Current Profit: {profit_pct:.2f}%")
            self.console.print(f"PSAR Direction: {'Bullish' if psar_direction == 1 else 'Bearish'}")
            self.console.print(f"PSAR Direction Changed: {'Yes' if psar_direction != prev_psar_direction else 'No'}")
            self.console.print(f"EMA Gap: {ema_gap:.4f}")
            self.console.print(f"EMA Gap ROC (3-period): {ema_gap_roc_3:.2f}%")
            self.console.print(f"Gap Widening: {'Yes' if gap_widening else 'No'}")
            
            # Check if PSAR has just flipped direction (from previous bar to current)
            psar_flipped = psar_direction != prev_psar_direction
            
            # For a BUY position, check for counter-trend SELL opportunity
            if position.type == mt5.POSITION_TYPE_BUY:
                # Check if PSAR is bearish (above price) or just flipped bearish
                psar_is_bearish = psar_direction == -1
                psar_flipped_bearish = psar_flipped and psar_is_bearish
                
                # Check price movement is downward
                price_moving_down = current_price < prev_price
                
                # Check if price is near EMA fast line but not crossed below slow line
                price_near_ema9 = current_price < ema_fast * 1.005  # Within 0.5% of EMA9
                
                # Check for significant previous trend strength
                strong_trend = ema_gap_roc_3 > 5.0  # Gap was widening rapidly
                
                # Check for trend reversal signs - now using psar_is_bearish instead of requiring a flip
                reversal_signal = not gap_widening and psar_is_bearish
                
                # Log conditions - show both if PSAR just flipped and if it's currently in the right direction
                self.console.print("\n[bold]Counter-trend SELL conditions:[/bold]")
                self.console.print(f"✓ PSAR Just Flipped Bearish: {'✅' if psar_flipped_bearish else '❌'}")
                self.console.print(f"✓ PSAR Is Bearish: {'✅' if psar_is_bearish else '❌'}")
                self.console.print(f"✓ Price Moving Down: {'✅' if price_moving_down else '❌'}")
                self.console.print(f"✓ Price Near EMA9: {'✅' if price_near_ema9 else '❌'}")
                self.console.print(f"✓ Previous Strong Trend (EMA Gap ROC): {'✅' if strong_trend else '❌'}")
                self.console.print(f"✓ Reversal Signal (Gap Narrowing): {'✅' if not gap_widening else '❌'}")
                
                # Consider counter-trend SELL if conditions are met (now using psar_is_bearish instead of requiring a flip)
                if psar_is_bearish and price_moving_down and price_near_ema9:
                    counter_signal = "SELL"
                    self.console.print("[bold red]Counter-trend SELL opportunity detected![/bold red]")
            
            # For a SELL position, check for counter-trend BUY opportunity
            elif position.type == mt5.POSITION_TYPE_SELL:
                # Check if PSAR is bullish (below price) or just flipped bullish
                psar_is_bullish = psar_direction == 1
                psar_flipped_bullish = psar_flipped and psar_is_bullish
                
                # Check price movement is upward
                price_moving_up = current_price > prev_price
                
                # Check if price is near EMA fast line but not crossed above slow line
                price_near_ema9 = current_price > ema_fast * 0.995  # Within 0.5% of EMA9
                
                # Check for significant previous trend strength
                strong_trend = ema_gap_roc_3 > 5.0  # Gap was widening rapidly
                
                # Check for trend reversal signs - now using psar_is_bullish instead of requiring a flip
                reversal_signal = not gap_widening and psar_is_bullish
                
                # Log conditions - show both if PSAR just flipped and if it's currently in the right direction
                self.console.print("\n[bold]Counter-trend BUY conditions:[/bold]")
                self.console.print(f"✓ PSAR Just Flipped Bullish: {'✅' if psar_flipped_bullish else '❌'}")
                self.console.print(f"✓ PSAR Is Bullish: {'✅' if psar_is_bullish else '❌'}")
                self.console.print(f"✓ Price Moving Up: {'✅' if price_moving_up else '❌'}")
                self.console.print(f"✓ Price Near EMA9: {'✅' if price_near_ema9 else '❌'}")
                self.console.print(f"✓ Previous Strong Trend (EMA Gap ROC): {'✅' if strong_trend else '❌'}")
                self.console.print(f"✓ Reversal Signal (Gap Narrowing): {'✅' if not gap_widening else '❌'}")
                
                # Consider counter-trend BUY if conditions are met (now using psar_is_bullish instead of requiring a flip)
                if psar_is_bullish and price_moving_up and price_near_ema9:
                    counter_signal = "BUY"
                    self.console.print("[bold green]Counter-trend BUY opportunity detected![/bold green]")
            
            return counter_signal
            
        except Exception as e:
            self.console.print(f"[red]Error in check_for_counter_trend_opportunity: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
            return None

    def execute_counter_trend_trade(self, signal, df, original_position):
        """Execute a counter-trend trade to capture retracements
        
        This method is similar to execute_trade but specifically for counter-trend opportunities
        while a position is already open. The counter-trend position size is smaller (50% of regular size)
        and has tighter stop-loss and take-profit targets.
        
        Args:
            signal: The trading signal ("BUY" or "SELL")
            df: DataFrame with price and indicator data
            original_position: The original position we're counter-trading against
            
        Returns:
            bool: True if trade execution was successful, False otherwise
        """
        try:
            last_row = df.iloc[-1]
            
            # Get current price for entry
            price = mt5.symbol_info_tick(self.symbol).ask if signal == "BUY" else mt5.symbol_info_tick(self.symbol).bid
            
            # Get key target/indicator prices
            ema9 = last_row['ema_9']
            ema20 = last_row['ema_20']
            psar = last_row['psar']
            
            # Use smaller lot size for counter-trend trades (50% of regular)
            counter_trend_lot_size = self.lot_size * 0.5  
            
            # Calculate profit target based on distance to EMA20
            if signal == "BUY":
                # For counter-trend BUY, target is near EMA20 (resistance)
                target_price = ema20
                # Stop loss is just below recent PSAR
                stop_loss = psar * 0.995  # 0.5% below PSAR
            else:  # SELL
                # For counter-trend SELL, target is near EMA20 (support)
                target_price = ema20
                # Stop loss is just above recent PSAR
                stop_loss = psar * 1.005  # 0.5% above PSAR
            
            # Calculate potential R:R ratio
            potential_gain = abs(target_price - price)
            potential_loss = abs(price - stop_loss)
            risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0
            
            # Only proceed if R:R is acceptable (at least 1.5:1)
            if risk_reward < 1.5:
                self.console.print(f"[yellow]Counter-trend trade rejected: Risk-reward ratio too low ({risk_reward:.2f})[/yellow]")
                return False
            
            # Create unique identifier for this counter-trend trade
            counter_trend_id = f"counter_{original_position.ticket}_{int(time.time())}"
            
            # Prepare trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": counter_trend_lot_size,
                "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": stop_loss,
                "tp": target_price,
                "deviation": 20,  # Allow slight price deviation
                "magic": 123457,  # Different magic number to identify counter-trend trades
                "comment": f"Counter-trend {signal} against position {original_position.ticket}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Display trade details
            self.console.print("\n=== Counter-Trend Trade Execution ===")
            self.console.print(f"Signal: {signal}")
            self.console.print(f"Entry Price: {price}")
            self.console.print(f"Stop Loss: {stop_loss}")
            self.console.print(f"Take Profit: {target_price}")
            self.console.print(f"Risk-Reward Ratio: {risk_reward:.2f}")
            self.console.print(f"Position Size: {counter_trend_lot_size} lots")
            
            # Execute the trade
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.console.print(f"[bold green]Counter-trend {signal} executed successfully![/bold green]")
                self.console.print(f"Order ID: {result.order}")
                
                # Log the trade
                trade_data = {
                    "type": "COUNTER_" + signal,
                    "original_position": original_position.ticket,
                    "ticket": result.order,
                    "entry_price": price,
                    "stop_loss": stop_loss,
                    "take_profit": target_price,
                    "lot_size": counter_trend_lot_size,
                    "time": datetime.now().isoformat(),
                    "risk_reward": float(risk_reward),
                    "ema9_at_entry": float(ema9),
                    "ema20_at_entry": float(ema20),
                    "psar_at_entry": float(psar)
                }
                self.log_trade(trade_data)
                
                # Track this counter-trend position in our state
                self.counter_trend_positions[str(result.order)] = {
                    "original_position": original_position.ticket,
                    "signal": signal,
                    "entry_time": time.time(),
                    "entry_price": price
                }
                self.counter_trend_active = True
                self.last_counter_trend_check = time.time()
                self.save_state()
                
                # Adjust the stop loss of the original position to break-even or small profit
                # This helps protect our profits in the original position
                if original_position.type == mt5.POSITION_TYPE_BUY:
                    # For BUY positions, set SL to entry price + small buffer
                    new_sl = original_position.price_open * 1.001  # 0.1% above entry
                    
                    # Only update if new SL is higher than current SL
                    if original_position.sl is None or new_sl > original_position.sl:
                        sl_request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "symbol": self.symbol,
                            "position": original_position.ticket,
                            "sl": new_sl,
                            "tp": original_position.tp  # Keep TP unchanged
                        }
                        
                        sl_result = mt5.order_send(sl_request)
                        if sl_result.retcode == mt5.TRADE_RETCODE_DONE:
                            self.console.print(f"[green]Updated original position stop loss to break-even+: {new_sl:.2f}[/green]")
                        else:
                            self.console.print(f"[yellow]Failed to update original position stop loss: {sl_result.comment}[/yellow]")
                
                elif original_position.type == mt5.POSITION_TYPE_SELL:
                    # For SELL positions, set SL to entry price - small buffer
                    new_sl = original_position.price_open * 0.999  # 0.1% below entry
                    
                    # Only update if new SL is lower than current SL
                    if original_position.sl is None or new_sl < original_position.sl:
                        sl_request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "symbol": self.symbol,
                            "position": original_position.ticket,
                            "sl": new_sl,
                            "tp": original_position.tp  # Keep TP unchanged
                        }
                        
                        sl_result = mt5.order_send(sl_request)
                        if sl_result.retcode == mt5.TRADE_RETCODE_DONE:
                            self.console.print(f"[green]Updated original position stop loss to break-even+: {new_sl:.2f}[/green]")
                        else:
                            self.console.print(f"[yellow]Failed to update original position stop loss: {sl_result.comment}[/yellow]")
                
                return True
            else:
                self.console.print(f"[bold red]Counter-trend trade execution failed![/bold red]")
                self.console.print(f"Error Code: {result.retcode}")
                self.console.print(f"Error Description: {result.comment}")
                return False
        
        except Exception as e:
            self.console.print(f"[red]Error executing counter-trend trade: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
            return False

    def check_counter_trend_positions(self):
        """Check status of counter-trend positions and clean up expired entries"""
        if not self.counter_trend_active or not self.counter_trend_positions:
            return
        
        self.console.print("[dim]Checking counter-trend positions...[/dim]")
        
        try:
            # Get all current positions
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                self.console.print("[yellow]Failed to get positions for counter-trend check[/yellow]")
                return
            
            active_tickets = [str(pos.ticket) for pos in positions]
            counter_tickets = list(self.counter_trend_positions.keys())
            
            # Check which counter-trend positions are no longer active
            for ticket in counter_tickets:
                if ticket not in active_tickets:
                    # This counter-trend position has been closed
                    position_info = self.counter_trend_positions[ticket]
                    self.console.print(f"[green]Counter-trend position {ticket} has been closed[/green]")
                    
                    # Remove from tracking
                    self.counter_trend_positions.pop(ticket, None)
            
            # Update counter_trend_active flag if no positions left
            if not self.counter_trend_positions:
                self.counter_trend_active = False
                self.console.print("[green]No more active counter-trend positions[/green]")
            
            # Clean up any counter-trend positions older than 24 hours
            # (This is a safety measure to prevent tracking old positions forever)
            current_time = time.time()
            expired_tickets = []
            
            for ticket, info in self.counter_trend_positions.items():
                if current_time - info["entry_time"] > 86400:  # 24 hours
                    expired_tickets.append(ticket)
            
            for ticket in expired_tickets:
                self.console.print(f"[yellow]Removing expired counter-trend position tracking: {ticket}[/yellow]")
                self.counter_trend_positions.pop(ticket, None)
            
            # Save state after any changes
            if expired_tickets or counter_tickets != list(self.counter_trend_positions.keys()):
                self.save_state()
                
        except Exception as e:
            self.console.print(f"[red]Error checking counter-trend positions: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")

if __name__ == "__main__":
    # Set up a recovery mechanism in case of crashes
    max_restarts = 5
    restart_count = 0
    
    while restart_count < max_restarts:
        try:
            console = Console()
            console.print("[bold cyan]═════════════════════════════════════[/bold cyan]")
            console.print("[bold cyan]  BTC EMA-PSAR Bot - Starting Up  [/bold cyan]")
            console.print("[bold cyan]═════════════════════════════════════[/bold cyan]")
            
            # Initialize bot
            bot = BTCEmaPsarBot()
            
            # Run the bot - this will handle its own internal errors
            bot.run()
            
            # If run() completes normally (e.g. through KeyboardInterrupt), exit the loop
            break
            
        except Exception as e:
            restart_count += 1
            console = Console()  # Create new console in case the previous one failed
            console.print(f"[red]Bot crashed with error: {str(e)}[/red]")
            import traceback
            console.print(f"[red]{traceback.format_exc()}[/red]")
            
            # Only attempt restart if we haven't exceeded the maximum
            if restart_count < max_restarts:
                console.print(f"[yellow]Attempting restart {restart_count}/{max_restarts} in 10 seconds...[/yellow]")
                time.sleep(10)
                
                # Ensure clean shutdown of MT5 before restart
                try:
                    mt5.shutdown() 
                except:
                    pass
                    
                time.sleep(3)
            else:
                console.print("[red]Maximum restart attempts reached. Bot will now exit.[/red]")
    
    # Ensure clean shutdown when exiting
    try:
        mt5.shutdown()
        console.print("[green]MT5 connection closed successfully.[/green]")
    except:
        console.print("[yellow]Warning: MT5 shutdown may not have completed properly.[/yellow]")
    
    console.print("[bold cyan]BTC EMA-PSAR Bot has terminated.[/bold cyan]") 