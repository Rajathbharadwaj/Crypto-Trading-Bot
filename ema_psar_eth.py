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
        return json.JSONEncoder.default(self, obj)

class ETHEmaPsarBot:
    def __init__(self, timeframe=mt5.TIMEFRAME_M15):
        # First initialize MT5 to ensure we can access its constants
        if not mt5.initialize():
            raise Exception("MT5 initialization failed!")
            
        # Now we can safely define timeframes
        self.TIMEFRAMES = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        
        self.console = Console()
        self.symbol = "ETHUSD"
        self.timeframe = self.TIMEFRAMES["M15"]  # Default to M15
        self.running = True
        
        # Indicator parameters
        self.ema_fast = 9
        self.ema_slow = 20
        self.psar_step = 0.02
        self.psar_max = 0.2
        
        # Trading parameters
        self.lot_size = 5.0  # Fixed to match BTC bot
        self.price_movement_threshold = 0.15  # 0.15% movement required
        self.psar_distance_multiplier = 0.8   # PSAR should be at least 80% of average distance
        
        # State tracking
        self.crossover_detected = False
        self.crossover_type = None
        self.crossover_price = None
        self.crossover_time = None
        
        # Add PSAR direction tracking
        self.last_psar_direction = None
        self.psar_direction_changed = False
        self.psar_direction_change_time = None
        
        # Initialize MT5
        if not self.initialize_mt5():
            raise Exception("Failed to initialize MT5")

        # Add file paths for state and logs
        self.log_dir = "logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Fix missing state_file attribute
        self.state_file = os.path.join(self.log_dir, f"eth_ema_psar_state.json")
        self.trade_log_file = os.path.join(self.log_dir, f"eth_ema_psar_trades.json")
        
        # Load previous state if exists
        self.load_state()

    def initialize_mt5(self):
        """Initialize MT5 connection"""
        # Check if MT5 is already initialized
        if mt5.terminal_info() is not None:
            self.console.print("[green]MT5 is already initialized[/green]")
            return True
            
        # Try to initialize if not already
        if not mt5.initialize():
            self.console.print("[red]Failed to initialize MT5 terminal[/red]")
            return False
        
        # Try login with multiple attempts
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                self.console.print(f"[yellow]MT5 login attempt {attempt}/{max_attempts}...[/yellow]")
                if mt5.login(login=182992485, password="QAZwsx456!", server="Exness-MT5Trial6"):
                    self.console.print("[green]Successfully logged in to MT5[/green]")
                    
                    # Create log directory if it doesn't exist
                    if not os.path.exists(self.log_dir):
                        os.makedirs(self.log_dir)
                        
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
                self.console.print(f"[red]Error calculating PSAR direction: {str(direction_error)}[/red]")
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
                return None, "Not enough data"
            
            # Get the latest data
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            # Current prices
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                return None, "Failed to get symbol info"
            
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
                
            self.console.print("\n=== Signal Analysis ===")
            self.console.print(f"EMA Fast (9): {ema_fast:.2f}")
            self.console.print(f"EMA Slow (20): {ema_slow:.2f}")
            self.console.print(f"PSAR: {psar:.2f}")
            self.console.print(f"PSAR Direction: {'Bullish' if current_psar_direction == 1 else 'Bearish'}")
            self.console.print(f"PSAR Distance: {psar_distance:.2f}%")
            self.console.print(f"Avg PSAR Distance: {avg_psar_distance:.2f}%")
            
            # No signals if not enough data
            if pd.isna(ema_fast) or pd.isna(ema_slow) or pd.isna(psar):
                self.console.print("[yellow]Some indicators are NaN, waiting for more data...[/yellow]")
                return None, "Indicators contain NaN values"
        
            # Current signal
            signal = None
            reason = None
            
            # Check EMA alignment
            emas_bullish = ema_fast > ema_slow
            emas_bearish = ema_fast < ema_slow
            
            # Check if PSAR is below/above the Average PSAR range
            psar_below_avg = psar_distance > avg_psar_distance and current_psar_direction == 1
            psar_above_avg = psar_distance > avg_psar_distance and current_psar_direction == -1
            
            # Display indicators alignment
            self.console.print(f"EMAs Bullish Aligned: {emas_bullish}")
            self.console.print(f"EMAs Bearish Aligned: {emas_bearish}")
            self.console.print(f"PSAR Direction Changed: {self.psar_direction_changed}")
            self.console.print(f"PSAR Below Avg Range: {psar_below_avg}")
            self.console.print(f"PSAR Above Avg Range: {psar_above_avg}")
            
            # EXPLAIN WHY NO SIGNAL
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
                    
                    # Reset direction change flag if it's been more than 1 hour
                    if time_since_direction_change > 3600:  # 1 hour
                        self.console.print("[yellow]PSAR direction change timed out (>1 hour)[/yellow]")
                        self.psar_direction_changed = False
                        self.save_state()
                except Exception as e:
                    self.console.print(f"[red]Error calculating time difference: {str(e)}[/red]")
                    # Reset to avoid future errors
                    self.psar_direction_change_time = time.time()
                    self.save_state()
            
            # Generate signal when PSAR direction changes and aligns with EMAs
            if self.psar_direction_changed:
                # For BUY signal, we want:
                # - PSAR is below price (bullish)
                # - EMAs are aligned bullishly (fast above slow)
                # - Price movement is positive since direction change
                if current_psar_direction == 1 and emas_bullish:
                    # Calculate price movement since PSAR direction change
                    price_movement = None
                    
                    if hasattr(self, 'psar_direction_change_time') and self.psar_direction_change_time:
                        # Get a few candles ago to check price movement
                        candles_ago = min(5, len(df) - 1)  # At most 5 candles back
                        past_price = df.iloc[-candles_ago]['close']
                        price_movement = ((current_ask / past_price) - 1) * 100
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
                
                # For SELL signal, we want:
                # - PSAR is above price (bearish)
                # - EMAs are aligned bearishly (fast below slow)
                # - Price movement is negative since direction change
                elif current_psar_direction == -1 and emas_bearish:
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
            
            # Additional signals based on PSAR distance from price
            # For BUY, PSAR is below price, far enough away, and EMAs are bullish
            if psar_below_avg and emas_bullish:
                signal = "BUY"
                reason = "PSAR below Average PSAR range, showing strong bullish signal"
                
                self.console.print(f"[bold green]✓ BUY SIGNAL TRIGGERED (PSAR Below Avg Range)![/bold green]")
                self.console.print(f"[green]Will buy at: {current_ask}[/green]")
                self.console.print(f"[yellow]Reason: {reason}[/yellow]")
                
                # Store reason for logging
                self.last_signal_reason = reason
                
                return signal, reason
                
            # For SELL, PSAR is above price, far enough away, and EMAs are bearish
            if psar_above_avg and emas_bearish:
                signal = "SELL"
                reason = "PSAR above Average PSAR range, showing strong bearish signal"
                
                self.console.print(f"[bold red]✓ SELL SIGNAL TRIGGERED (PSAR Above Avg Range)![/bold red]")
                self.console.print(f"[green]Will sell at: {current_bid}[/green]")
                self.console.print(f"[yellow]Reason: {reason}[/yellow]")
                
                # Store reason for logging
                self.last_signal_reason = reason
                
                return signal, reason
            
            # No signal if none of the conditions are met
            return None, None
            
        except Exception as e:
            self.console.print(f"[red]Error getting trading signal: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
            return None, f"Error: {str(e)}"

    def run(self):
        """Main bot loop"""
        self.console.print("[bold green]Starting ETH EMA-PSAR Bot...[/bold green]")
        
        # Initialize for partial exits tracking
        self.partial_exits_taken = {}
        self.last_position_update = 0  # Track when we last updated position info
        self.last_mt5_check = time.time()  # Track when we last checked MT5 connection
        self.consecutive_errors = 0  # Track consecutive errors for backoff
        
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
                            # check_exit_conditions now returns bool to indicate if an action was taken
                            action_taken = self.check_exit_conditions(df, position)
                            
                            # Log the result of checking exit conditions
                            self.console.print(f"[dim]Exit check result: action_taken = {action_taken}[/dim]")
                            
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
            
            # Request a reduced amount of data to prevent memory issues
            # 2000 candles should be more than enough for PSAR and EMA calculations
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 2000)  # Reduced from 5000
            
            if rates is None:
                error_code = mt5.last_error()
                self.console.print(f"[red]Failed to get market data. MT5 error code: {error_code}[/red]")
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

    def execute_trade(self, signal, df):
        """Execute trade based on signal"""
        try:
            last_row = df.iloc[-1]
            
            # Get current price for entry
            price = mt5.symbol_info_tick(self.symbol).ask if signal == "BUY" else mt5.symbol_info_tick(self.symbol).bid
            
            # Calculate PSAR-based stop loss distance (1.5x the PSAR distance for safety)
            sl_distance_pct = last_row['psar_distance'] * 1.5
            
            # Calculate stop loss price
            sl_price = price * (1 - sl_distance_pct/100) if signal == "BUY" else price * (1 + sl_distance_pct/100)
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
            self.crossover_time = datetime.now().isoformat()
            self.save_state()
            
            # Determine order type
            order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
            
            # Create the trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": order_type,
                "price": price,
                "sl": sl_price,
                "deviation": 20,
                "magic": 234000,
                "comment": "ETH EMA-PSAR",
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
                return
            
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
            else:
                # Get the new position's ticket (ID)
                position_id = result.order
                self.console.print(f"[green]Order executed successfully! Ticket: {position_id}[/green]")
                
                # Initialize partial exits tracking for this position
                position_key = str(position_id)
                self.partial_exits_taken[position_key] = set()
                
                # Get the reason for the trade from the most recent signal generation
                trade_reason = getattr(self, 'last_signal_reason', 'Unknown reason')
                
                # Log the successful trade with our exit strategy details
                trade_data = {
                    "action": "ENTRY",
                    "signal": signal,
                    "ticket": position_id,
                    "entryPrice": price,
                    "stopLoss": sl_price,
                    "reason": trade_reason,  # Add the reason for the trade
                    "psarDistance": f"{last_row['psar_distance']:.2f}%",
                    "exitStrategy": {
                        "exit1": {
                            "percentage": exit1_pct,
                            "price": tp1_price,
                            "portion": 0.3,
                            "status": "PENDING"
                        },
                        "exit2": {
                            "percentage": exit2_pct,
                            "price": tp2_price,
                            "portion": 0.3,
                            "status": "PENDING"
                        },
                        "exit3": {
                            "percentage": exit3_pct,
                            "price": tp3_price,
                            "portion": 0.4,
                            "status": "PENDING"
                        }
                    },
                    "status": "OPEN"
                }
                self.log_trade(trade_data)
                
        except Exception as e:
            self.console.print(f"[red]Error executing trade: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")

    def check_exit_conditions(self, df, position):
        """Check if any exit conditions are met for an open position"""
        try:
            # Get position details
            position_type = position.type  # 0 for buy, 1 for sell
            position_key = str(position.ticket)  # Convert ticket to string for dictionary key
            
            # Initialize entry in partial_exits_taken if it doesn't exist
            if not hasattr(self, 'partial_exits_taken'):
                self.partial_exits_taken = {}
            if position_key not in self.partial_exits_taken:
                self.partial_exits_taken[position_key] = set()
            
            # Calculate current profit percentage
            entry_price = position.price_open
            current_price = position.price_current
            profit_pct = ((current_price / entry_price) - 1) * 100
            if position_type == mt5.ORDER_TYPE_SELL:
                profit_pct = -profit_pct
            
            # Get latest data for indicator-based exits
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2] if len(df) > 1 else None
            
            # Display position info for analysis
            self.console.print("\n=== Exit Analysis ===")
            self.console.print(f"Position Type: {'BUY' if position_type == mt5.ORDER_TYPE_BUY else 'SELL'}")
            self.console.print(f"Entry Price: {entry_price}")
            self.console.print(f"Current Price: {current_price}")
            self.console.print(f"Current Profit: {profit_pct:.2f}%")
            
            # Check for indicator-based exit conditions
            ema_cross_against = False
            psar_flip_against = False
            trailing_stop_hit = False
            
            # Check for EMA crossover against position
            if prev_row is not None:
                if position_type == mt5.ORDER_TYPE_BUY:
                    # For long positions, bearish crossover is against us
                    ema_cross_against = (prev_row['ema_9'] > prev_row['ema_20'] and 
                                         last_row['ema_9'] < last_row['ema_20'])
                else:
                    # For short positions, bullish crossover is against us
                    ema_cross_against = (prev_row['ema_9'] < prev_row['ema_20'] and 
                                         last_row['ema_9'] > last_row['ema_20'])
            
            # Check for PSAR flip against position
            if position_type == mt5.ORDER_TYPE_BUY:
                # For long positions, PSAR above price is against us
                psar_flip_against = last_row['psar_direction'] == -1
            else:
                # For short positions, PSAR below price is against us
                psar_flip_against = last_row['psar_direction'] == 1
            
            # Check if trailing stop is hit
            if position_type == mt5.ORDER_TYPE_BUY:
                # For long positions, trailing stop is hit when price falls below PSAR
                trailing_stop_hit = current_price < last_row['psar']
            else:
                # For short positions, trailing stop is hit when price rises above PSAR
                trailing_stop_hit = current_price > last_row['psar']
            
            # Define take-profit levels
            take_profit_1 = 0.3  # 0.3% for first exit
            take_profit_2 = 0.5  # 0.5% for second exit
            take_profit_3 = 0.7  # 0.7% for third exit
            
            # Display take-profit levels and current status
            self.console.print("\nTake-Profit Levels:")
            self.console.print(f"TP1 (0.3%): {'Reached' if profit_pct >= take_profit_1 else 'Not reached'} | Executed: {'Yes' if 'exit1' in self.partial_exits_taken[position_key] else 'No'}")
            self.console.print(f"TP2 (0.5%): {'Reached' if profit_pct >= take_profit_2 else 'Not reached'} | Executed: {'Yes' if 'exit2' in self.partial_exits_taken[position_key] else 'No'}")
            self.console.print(f"TP3 (0.7%): {'Reached' if profit_pct >= take_profit_3 else 'Not reached'} | Executed: {'Yes' if 'exit3' in self.partial_exits_taken[position_key] else 'No'}")
            
            # Check if we've completed all partial exits and need to use trailing stop for remainder
            all_exits_completed = all(exit_key in self.partial_exits_taken[position_key] for exit_key in ['exit1', 'exit2', 'exit3'])
            
            # Get current stop loss level
            current_sl = position.sl
            
            # Display exit conditions
            self.console.print("\nExit Conditions:")
            self.console.print(f"EMA Cross Against: {'Yes' if ema_cross_against else 'No'}")
            self.console.print(f"PSAR Flip Against: {'Yes' if psar_flip_against else 'No'}")
            self.console.print(f"Trailing Stop Hit: {'Yes' if trailing_stop_hit else 'No'}")
            self.console.print(f"All Partial Exits Completed: {'Yes' if all_exits_completed else 'No'}")
            
            # Track if we executed any actions
            action_executed = False
            
            # Check for trailing stop after all partial exits
            if all_exits_completed:
                # Define trailing stop levels after all partial exits
                trailing_levels = [
                    {"level": 0.9, "sl_pct": 0.7},  # When price reaches 0.9%, move SL to 0.7%
                    {"level": 1.1, "sl_pct": 0.9},  # When price reaches 1.1%, move SL to 0.9%
                    {"level": 1.3, "sl_pct": 1.1},  # When price reaches 1.3%, move SL to 1.1%
                    {"level": 1.5, "sl_pct": 1.3},  # When price reaches 1.5%, move SL to 1.3%
                ]
                
                # Sort levels from highest to lowest for proper checking
                if position_type == mt5.ORDER_TYPE_BUY:
                    trailing_levels = sorted(trailing_levels, key=lambda x: x["level"], reverse=True)
                else:  # For SELL positions, check lowest levels first
                    trailing_levels = sorted(trailing_levels, key=lambda x: x["level"])
                
                # Check if we need to update the trailing stop
                for level_data in trailing_levels:
                    level = level_data["level"]
                    sl_pct = level_data["sl_pct"]
                    
                    # Calculate new SL price based on entry price and SL percentage
                    if position_type == mt5.ORDER_TYPE_BUY:
                        new_sl_price = entry_price * (1 + sl_pct/100)
                        # Only update if price has reached the level and new SL is higher than current SL
                        if profit_pct >= level and (current_sl is None or new_sl_price > current_sl):
                            self.console.print(f"[bold green]Updating trailing stop: {sl_pct:.2f}% ({new_sl_price:.2f})[/bold green]")
                            
                            # Update SL in MT5
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "symbol": self.symbol,
                                "position": position.ticket,
                                "sl": new_sl_price,
                                "tp": position.tp  # Keep TP unchanged
                            }
                            
                            result = mt5.order_send(request)
                            if result.retcode == mt5.TRADE_RETCODE_DONE:
                                self.console.print(f"[green]Trailing stop updated to {new_sl_price:.2f}[/green]")
                                return True  # Action executed
                            else:
                                self.console.print(f"[red]Failed to update trailing stop: {result.comment}[/red]")
                            
                            break  # Only try to set the highest applicable level
                    else:  # SELL position
                        new_sl_price = entry_price * (1 - sl_pct/100)
                        # Only update if price has reached the level and new SL is lower than current SL
                        if profit_pct >= level and (current_sl is None or new_sl_price < current_sl):
                            self.console.print(f"[bold green]Updating trailing stop: {sl_pct:.2f}% ({new_sl_price:.2f})[/bold green]")
                            
                            # Update SL in MT5
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "symbol": self.symbol,
                                "position": position.ticket,
                                "sl": new_sl_price,
                                "tp": position.tp  # Keep TP unchanged
                            }
                            
                            result = mt5.order_send(request)
                            if result.retcode == mt5.TRADE_RETCODE_DONE:
                                self.console.print(f"[green]Trailing stop updated to {new_sl_price:.2f}[/green]")
                                return True  # Action executed
                            else:
                                self.console.print(f"[red]Failed to update trailing stop: {result.comment}[/red]")
                            
                            break  # Only try to set the highest applicable level
                
                # Immediately after third exit, set SL to the second exit level (0.5%)
                trailing_sl_key = f"trailing_sl_set_{position_key}"
                if not hasattr(self, trailing_sl_key) or not getattr(self, trailing_sl_key, False):
                    if position_type == mt5.ORDER_TYPE_BUY:
                        new_sl_price = entry_price * (1 + take_profit_2/100)  # Set to TP2 level (0.5%)
                    else:  # SELL position
                        new_sl_price = entry_price * (1 - take_profit_2/100)  # Set to TP2 level (0.5%)
                    
                    # Only update if current SL is below/above the new SL
                    if current_sl is None or (position_type == mt5.ORDER_TYPE_BUY and new_sl_price > current_sl) or \
                                            (position_type == mt5.ORDER_TYPE_SELL and new_sl_price < current_sl):
                        self.console.print(f"[bold green]Setting initial trailing stop after all exits: {take_profit_2:.2f}% ({new_sl_price:.2f})[/bold green]")
                        
                        # Update SL in MT5
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "symbol": self.symbol,
                            "position": position.ticket,
                            "sl": new_sl_price,
                            "tp": position.tp  # Keep TP unchanged
                        }
                        
                        result = mt5.order_send(request)
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            self.console.print(f"[green]Initial trailing stop set to {new_sl_price:.2f}[/green]")
                            # Mark that we've set the initial trailing SL
                            setattr(self, trailing_sl_key, True)
                            self.save_state()  # Save the state to persist this setting
                            return True  # Action executed
                        else:
                            self.console.print(f"[red]Failed to set initial trailing stop: {result.comment}[/red]")
            
            # Check for full exit conditions based on indicators
            if ema_cross_against or psar_flip_against or trailing_stop_hit:
                reason = []
                if ema_cross_against:
                    reason.append("EMA Crossover Against Position")
                if psar_flip_against:
                    reason.append("PSAR Flipped Against Position")
                if trailing_stop_hit:
                    reason.append("Trailing Stop Hit")
                    
                exit_reason = ", ".join(reason)
                self.console.print(f"[bold yellow]Executing full exit due to: {exit_reason}[/bold yellow]")
                result = self.close_position(position, exit_reason)
                if result:
                    self.console.print("[green]Full exit execution successful[/green]")
                    action_executed = True
                else:
                    self.console.print("[red]Full exit execution failed![/red]")
            
            # Check take-profit levels and execute partial exits
            if position_type == mt5.ORDER_TYPE_BUY or position_type == mt5.ORDER_TYPE_SELL:
                # First take-profit level (exit 30% of position)
                if profit_pct >= take_profit_1 and "exit1" not in self.partial_exits_taken[position_key]:
                    self.console.print(f"[bold green]TP1 triggered! Profit: {profit_pct:.2f}% >= {take_profit_1}%[/bold green]")
                    result = self.execute_partial_exit(position, 0.3, f"Take Profit 1 ({take_profit_1}%)")
                    if result:
                        self.console.print("[green]TP1 execution successful, updating state...[/green]")
                        self.partial_exits_taken[position_key].add("exit1")
                        
                        # Update SL to breakeven (or small positive) after TP1 execution
                        # Use a small buffer (e.g., 0.1%) to account for spread and fees
                        buffer_pct = 0.1
                        if position_type == mt5.ORDER_TYPE_BUY:
                            new_sl_price = entry_price * (1 + buffer_pct/100)  # Slightly above entry
                        else:  # SELL position
                            new_sl_price = entry_price * (1 - buffer_pct/100)  # Slightly below entry
                        
                        # Only update if current SL is below/above the new SL
                        current_sl = position.sl
                        if current_sl is None or (position_type == mt5.ORDER_TYPE_BUY and new_sl_price > current_sl) or \
                                                (position_type == mt5.ORDER_TYPE_SELL and new_sl_price < current_sl):
                            self.console.print(f"[bold green]Updating SL after TP1 to breakeven+: {buffer_pct:.2f}% ({new_sl_price:.2f})[/bold green]")
                            
                            # Update SL in MT5
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "symbol": self.symbol,
                                "position": position.ticket,
                                "sl": new_sl_price,
                                "tp": position.tp  # Keep TP unchanged
                            }
                            
                            sl_result = mt5.order_send(request)
                            if sl_result.retcode == mt5.TRADE_RETCODE_DONE:
                                self.console.print(f"[green]SL updated to {new_sl_price:.2f} after TP1[/green]")
                            else:
                                self.console.print(f"[red]Failed to update SL after TP1: {sl_result.comment}[/red]")
                        
                        self.save_state()  # Save state to persist between restarts
                        action_executed = True
                    else:
                        self.console.print("[red]TP1 execution failed![/red]")
                    
                # Second take-profit level (exit another 30% of position)
                elif profit_pct >= take_profit_2 and "exit2" not in self.partial_exits_taken[position_key] and "exit1" in self.partial_exits_taken[position_key]:
                    self.console.print(f"[bold green]TP2 triggered! Profit: {profit_pct:.2f}% >= {take_profit_2}%[/bold green]")
                    result = self.execute_partial_exit(position, 0.3, f"Take Profit 2 ({take_profit_2}%)")
                    if result:
                        self.console.print("[green]TP2 execution successful, updating state...[/green]")
                        self.partial_exits_taken[position_key].add("exit2")
                        
                        # Update SL to TP1 level after TP2 execution
                        if position_type == mt5.ORDER_TYPE_BUY:
                            new_sl_price = entry_price * (1 + take_profit_1/100)  # Set to TP1 level (0.3%)
                        else:  # SELL position
                            new_sl_price = entry_price * (1 - take_profit_1/100)  # Set to TP1 level (0.3%)
                        
                        # Only update if current SL is below/above the new SL
                        current_sl = position.sl
                        if current_sl is None or (position_type == mt5.ORDER_TYPE_BUY and new_sl_price > current_sl) or \
                                                (position_type == mt5.ORDER_TYPE_SELL and new_sl_price < current_sl):
                            self.console.print(f"[bold green]Updating SL after TP2 to TP1 level: {take_profit_1:.2f}% ({new_sl_price:.2f})[/bold green]")
                            
                            # Update SL in MT5
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "symbol": self.symbol,
                                "position": position.ticket,
                                "sl": new_sl_price,
                                "tp": position.tp  # Keep TP unchanged
                            }
                            
                            sl_result = mt5.order_send(request)
                            if sl_result.retcode == mt5.TRADE_RETCODE_DONE:
                                self.console.print(f"[green]SL updated to {new_sl_price:.2f} after TP2[/green]")
                            else:
                                self.console.print(f"[red]Failed to update SL after TP2: {sl_result.comment}[/red]")
                        
                        self.save_state()
                        action_executed = True
                    else:
                        self.console.print("[red]TP2 execution failed![/red]")
                    
                # Third take-profit level (exit remaining 40% of position)
                elif profit_pct >= take_profit_3 and "exit3" not in self.partial_exits_taken[position_key] and "exit2" in self.partial_exits_taken[position_key]:
                    self.console.print(f"[bold green]TP3 triggered! Profit: {profit_pct:.2f}% >= {take_profit_3}%[/bold green]")
                    result = self.execute_partial_exit(position, 0.4, f"Take Profit 3 ({take_profit_3}%)")
                    if result:
                        self.console.print("[green]TP3 execution successful, updating state...[/green]")
                        self.partial_exits_taken[position_key].add("exit3")
                        
                        # Update SL to TP2 level after TP3 execution
                        if position_type == mt5.ORDER_TYPE_BUY:
                            new_sl_price = entry_price * (1 + take_profit_2/100)  # Set to TP2 level (0.5%)
                        else:  # SELL position
                            new_sl_price = entry_price * (1 - take_profit_2/100)  # Set to TP2 level (0.5%)
                        
                        # Only update if current SL is below/above the new SL
                        current_sl = position.sl
                        if current_sl is None or (position_type == mt5.ORDER_TYPE_BUY and new_sl_price > current_sl) or \
                                                (position_type == mt5.ORDER_TYPE_SELL and new_sl_price < current_sl):
                            self.console.print(f"[bold green]Updating SL after TP3 to TP2 level: {take_profit_2:.2f}% ({new_sl_price:.2f})[/bold green]")
                            
                            # Update SL in MT5
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "symbol": self.symbol,
                                "position": position.ticket,
                                "sl": new_sl_price,
                                "tp": position.tp  # Keep TP unchanged
                            }
                            
                            sl_result = mt5.order_send(request)
                            if sl_result.retcode == mt5.TRADE_RETCODE_DONE:
                                self.console.print(f"[green]SL updated to {new_sl_price:.2f} after TP3[/green]")
                            else:
                                self.console.print(f"[red]Failed to update SL after TP3: {sl_result.comment}[/red]")
                        
                        self.save_state()
                        action_executed = True
                    else:
                        self.console.print("[red]TP3 execution failed![/red]")
            
            # Final status update
            self.console.print(f"[dim]Exit check complete. Action taken: {action_executed}[/dim]")
            
            # Return whether we executed any action (partial exit or full exit)
            return action_executed
            
        except Exception as e:
            self.console.print(f"[red]Error checking exit conditions: {str(e)}[/red]")
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

    def execute_partial_exit(self, position, exit_portion, reason):
        """Execute a partial position exit"""
        try:
            self.console.print(f"[bold yellow]Attempting to execute partial exit ({exit_portion*100}%) for position #{position.ticket}[/bold yellow]")
            
            # Calculate volume to close
            original_volume = position.volume
            volume_to_close = position.volume * exit_portion
            
            # Round volume to broker's requirements (usually 0.01 is minimum)
            # Most brokers require 0.01 lot steps
            volume_to_close = round(volume_to_close, 2)
            
            # Ensure volume is at least the minimum (0.01 lots)
            if volume_to_close < 0.01:
                volume_to_close = 0.01
                self.console.print("[yellow]Warning: Adjusted partial exit volume to minimum 0.01 lots[/yellow]")
            
            # Ensure volume doesn't exceed position size
            if volume_to_close > position.volume:
                volume_to_close = position.volume
                self.console.print("[yellow]Warning: Adjusted volume to match position size[/yellow]")
            
            # Get current price for the exit
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.console.print("[red]Failed to get symbol info, cannot execute partial exit[/red]")
                return False
                
            price = mt5.symbol_info_tick(self.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask
            
            self.console.print(f"[yellow]Attempting partial exit: {volume_to_close} lots at {price}...[/yellow]")
            
            # Double-check that volume is within allowed range
            if symbol_info.volume_min is not None and symbol_info.volume_step is not None:
                # Round to the nearest valid step
                volume_to_close = round(volume_to_close / symbol_info.volume_step) * symbol_info.volume_step
                self.console.print(f"[dim]Adjusted volume to broker requirements: {volume_to_close} lots[/dim]")
                
                # Ensure it's not less than minimum
                if volume_to_close < symbol_info.volume_min:
                    volume_to_close = symbol_info.volume_min
                    self.console.print(f"[yellow]Adjusted to minimum volume: {volume_to_close} lots[/yellow]")
            
            # Simplified comment
            comment = "Partial Exit"
            
            # Create the order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume_to_close,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": comment,
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
                
                # Query broker for valid volume steps
                symbol_info = mt5.symbol_info(self.symbol)
                if symbol_info is not None:
                    self.console.print(f"[yellow]Broker info: Min volume: {symbol_info.volume_min}, Step: {symbol_info.volume_step}[/yellow]")
                    
                    # Try again with corrected volume
                    if symbol_info.volume_step > 0:
                        corrected_volume = round(volume_to_close / symbol_info.volume_step) * symbol_info.volume_step
                        if corrected_volume >= symbol_info.volume_min and corrected_volume <= position.volume:
                            self.console.print(f"[yellow]Retrying with corrected volume: {corrected_volume}[/yellow]")
                            request["volume"] = corrected_volume
                            result = mt5.order_send(request)
                        else:
                            self.console.print(f"[red]Cannot execute partial exit: volume {corrected_volume} outside allowed range[/red]")
                            return False
                
                # If still None after retry
                if result is None:
                    # Try to get updated position info
                    positions = mt5.positions_get(ticket=position.ticket)
                    if not positions:
                        self.console.print("[yellow]Position no longer exists - may have been closed already[/yellow]")
                        return True
                        
                    self.console.print("[red]Failed to execute partial exit[/red]")
                    return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.console.print(f"[red]Partial exit failed: {result.comment} (code {result.retcode})[/red]")
                
                # Additional diagnostics for volume errors
                if result.retcode == 10014:  # Invalid volume error
                    symbol_info = mt5.symbol_info(self.symbol)
                    if symbol_info is not None:
                        self.console.print(f"[yellow]Broker requires: Min volume: {symbol_info.volume_min}, Step: {symbol_info.volume_step}[/yellow]")
                        self.console.print(f"[yellow]We tried: {volume_to_close} lots[/yellow]")
                        
                        # Try closing the full position instead
                        self.console.print("[yellow]Attempting to close full position instead...[/yellow]")
                        return self.close_position(position, reason + " (adjusted to full exit)")
                
                return False
            else:
                self.console.print(f"[green]✓ Partial exit successful: {volume_to_close} lots at {price} ({reason})[/green]")
                self.console.print(f"[green]Profit for this portion: ${position.profit * (volume_to_close/original_volume):.2f}[/green]")
                
                # Log partial exit
                exit_data = {
                    "action": "PARTIAL_EXIT",
                    "ticket": position.ticket,
                    "exitPortion": exit_portion,
                    "volume": volume_to_close,
                    "entryPrice": position.price_open,
                    "soldAt" if position.type == mt5.ORDER_TYPE_BUY else "boughtAt": price,
                    "profit": position.profit * (volume_to_close/original_volume),
                    "exitReason": reason,
                    "status": "PARTIAL_CLOSE"
                }
                self.log_trade(exit_data)
                
                # After successful partial exit, save state to ensure we don't lose track
                self.save_state()
                
                # Return success
                return True
                
        except Exception as e:
            self.console.print(f"[red]Error executing partial exit: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
            return False

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
                    
                    # Initialize partial_exits_taken if it doesn't exist
                    if not hasattr(self, 'partial_exits_taken'):
                        self.partial_exits_taken = {}
                    
                    # Load partial_exits_taken from state
                    partial_exits = state.get('partial_exits_taken', {})
                    for position_id, exits in partial_exits.items():
                        if isinstance(exits, list):
                            self.partial_exits_taken[position_id] = set(exits)
                        else:
                            self.partial_exits_taken[position_id] = exits
                    
                    # Load PSAR direction tracking
                    self.last_psar_direction = state.get('last_psar_direction', None)
                    self.psar_direction_changed = state.get('psar_direction_changed', False)
                    self.psar_direction_change_time = state.get('psar_direction_change_time', None)
                
                self.console.print("[green]State loaded successfully[/green]")
            else:
                # Initialize with default values
                self.crossover_detected = False
                self.crossover_type = None
                self.crossover_price = None
                self.crossover_time = None
                self.partial_exits_taken = {}
                self.last_psar_direction = None
                self.psar_direction_changed = False
                self.psar_direction_change_time = None
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

    def save_state(self):
        """Save the bot's state to a file"""
        try:
            # Convert sets to lists for JSON serialization
            partial_exits_json = {}
            if hasattr(self, 'partial_exits_taken'):
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
                'psar_direction_change_time': self.psar_direction_change_time
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, cls=NumpyEncoder)
            
            self.console.print("[green]State saved successfully[/green]")
        except Exception as e:
            self.console.print(f"[red]Error saving state: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")

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

    def display_position_status(self, position, df):
        """Display detailed position status and analysis"""
        try:
            # Get position details
            position_type = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
            entry_price = position.price_open
            current_price = position.price_current
            profit = position.profit
            volume = position.volume
            
            # Calculate profit percentage
            profit_pct = ((current_price / entry_price) - 1) * 100
            if position_type == "SELL":
                profit_pct = -profit_pct
            
            # Get latest indicator values
            last_row = df.iloc[-1]
            ema_fast = last_row['ema_9']
            ema_slow = last_row['ema_20']
            psar = last_row['psar']
            psar_direction = "Bullish" if last_row['psar_direction'] == 1 else "Bearish"
            
            # Format the output
            self.console.print("\n=== Position Analysis ===")
            self.console.print(f"Position #{position.ticket}:")
            self.console.print(f"Type: [{'green' if position_type == 'BUY' else 'red'}]{position_type}[/]")
            self.console.print(f"Volume: {volume} lots")
            self.console.print(f"Entry Price: {entry_price}")
            self.console.print(f"Current Price: {current_price}")
            self.console.print(f"P/L: [{'green' if profit >= 0 else 'red'}]${profit:.2f} ({profit_pct:.2f}%)[/]")
            
            # Show indicator status
            self.console.print("\n=== Indicator Status ===")
            self.console.print(f"EMA9: {ema_fast:.2f}")
            self.console.print(f"EMA20: {ema_slow:.2f}")
            self.console.print(f"PSAR: {psar:.2f} ({psar_direction})")
            
            # Show exit levels if we have them in our tracking
            position_key = str(position.ticket)
            if hasattr(self, 'partial_exits_taken') and self.partial_exits_taken.get(position_key):
                self.console.print("\n=== Exit Status ===")
                self.console.print(f"Partial exits taken: {self.partial_exits_taken[position_key]}")
            
            self.console.print("=" * 40)
            
        except Exception as e:
            self.console.print(f"[red]Error displaying position status: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")

if __name__ == "__main__":
    try:
        bot = ETHEmaPsarBot()
        
        # Set up a recovery mechanism in case of crashes
        max_restarts = 5
        restart_count = 0
        
        while restart_count < max_restarts:
            try:
                bot.run()
                break  # If run() completes normally, exit the loop
            except Exception as e:
                restart_count += 1
                Console().print(f"[red]Bot crashed with error: {str(e)}[/red]")
                import traceback
                Console().print(f"[red]{traceback.format_exc()}[/red]")
                Console().print(f"[yellow]Attempting restart {restart_count}/{max_restarts}...[/yellow]")
                time.sleep(10)
                
                # Reinitialize MT5 and bot if needed
                mt5.shutdown()
                time.sleep(3)
                if not mt5.initialize():
                    Console().print("[red]Failed to reinitialize MT5[/red]")
                    continue
                    
                bot = ETHEmaPsarBot()
        
    except Exception as e:
        Console().print(f"[red]Fatal bot error: {str(e)}[/red]")
    finally:
        mt5.shutdown() 