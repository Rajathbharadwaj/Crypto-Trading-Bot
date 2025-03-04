import MetaTrader5 as mt5
import pandas as pd
import time
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta, timezone
from rich.console import Console
from rich.table import Table
import pytz
import os
import json

class ETHTradingBot:
    def __init__(self, timeframe=mt5.TIMEFRAME_M15, name="ETHBot"):
        self.name = name
        self.timeframe = timeframe
        self.running = True
        self.console = Console()
        self.symbol = "ETHUSD"
        
        # Load saved state or initialize new state
        self.state_file = "eth_state.json"
        self.load_state()
        
        # State tracking variables
        self.crossover_detected = False
        self.crossover_type = None  # "BUY" or "SELL"
        self.crossover_price = None
        self.crossover_price_index = -1
        self.crossover_time = None
        
        # Trading configuration
        self.lot_size = 1.0     # 1 lot as specified
        self.min_cross_distance = 0  # No minimum distance required
        self.confirmation_period = 1  # Place trade at next candle open
        
        # Partial exit configuration
        self.tp1_percent = 0.0025  # 0.25% for 0.50 lots
        self.tp2_percent = 0.0050  # 0.50% for 0.20 lots
        self.tp3_percent = 0.0075  # 0.75% for 0.20 lots
        self.partial_lots = [0.50, 0.20, 0.20, 0.10]  # Lot sizes for each exit
        
        # Stop Loss and Trailing configuration
        self.initial_sl_percent = 0.003  # 0.3% initial stop loss
        self.trailing_enabled = True
        self.trailing_activation_percent = 0.0075  # Activate at 0.75% profit
        self.trailing_start_percent = 0.005  # Move SL to 0.5% above entry
        self.trailing_step_percent = 0.0025  # Move 0.25% with every 0.25% move
        
        # Strategy Parameters
        self.ema_short = 9      # Short EMA period
        self.ema_long = 20      # Long EMA period
        
        # Trading hours (Eastern Time) - Updated for all days
        self.trading_start = "18:00:00"  # Sunday 6 PM ET
        self.trading_end = "17:00:00"    # Friday 5 PM ET
        self.trading_days = [0, 1, 2, 3, 4, 5, 6]  # All days (0 = Monday, 6 = Sunday)
        
        # Initialize active trades
        self.active_trades = []
        
        # Debug flag
        self.debug_mode = True  # Enable detailed logging
        
        if not self.initialize_mt5():
            raise Exception("MT5 initialization failed!")
    
    def load_state(self):
        """Load bot state from file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.console.print(f"[bold green]Loading state from {self.state_file}:[/bold green]")
                self.console.print(json.dumps(state, indent=2))
                
                self.crossover_detected = state.get('crossover_detected', False)
                self.crossover_type = state.get('crossover_type', None)
                self.crossover_price = state.get('crossover_price', None)
                self.crossover_price_index = state.get('crossover_price_index', -1)
                
                # CRITICAL: Handle crossover_time properly
                if 'crossover_time' in state:
                    self.crossover_time = state.get('crossover_time')
                    self.console.print(f"[green]Loaded crossover_time: {self.crossover_time}[/green]")
                elif self.crossover_detected:
                    # If crossover is detected but time is missing, add it now
                    import datetime
                    self.crossover_time = datetime.datetime.now().isoformat()
                    self.console.print(f"[yellow]Missing crossover_time but crossover_detected is True. Added time: {self.crossover_time}[/yellow]")
                    # Save the updated state with the time
                    self.save_state()
                    
                # Print the loaded state values (to verify they were set correctly)
                self.console.print(f"[bold green]State loaded successfully:[/bold green]")
                self.console.print(f"crossover_detected = {self.crossover_detected}")
                self.console.print(f"crossover_type = {self.crossover_type}")
                self.console.print(f"crossover_price = {self.crossover_price}")
                self.console.print(f"crossover_price_index = {self.crossover_price_index}")
                if hasattr(self, 'crossover_time'):
                    self.console.print(f"crossover_time = {self.crossover_time}")
            else:
                self.console.print(f"[yellow]State file {self.state_file} not found. Initializing new state.[/yellow]")
                self._reset_crossover_state()
        except Exception as e:
            self.console.print(f"[red]Error loading state: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
            self._reset_crossover_state()

    def save_state(self):
        """Save bot state to file"""
        try:
            state = {
                'crossover_detected': self.crossover_detected,
                'crossover_type': self.crossover_type,
                'crossover_price': self.crossover_price,
                'crossover_price_index': self.crossover_price_index
            }
            
            # Always include crossover_time in the state if crossover_detected is True
            if self.crossover_detected:
                if hasattr(self, 'crossover_time'):
                    state['crossover_time'] = self.crossover_time
                else:
                    # If missing, add current timestamp as crossover_time
                    import datetime
                    state['crossover_time'] = datetime.datetime.now().isoformat()
                    self.crossover_time = state['crossover_time']
                    self.console.print(f"[yellow]Added missing crossover_time: {self.crossover_time}[/yellow]")
            
            self.console.print(f"[cyan]Saving state to {self.state_file}:[/cyan]")
            self.console.print(json.dumps(state, indent=2))
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            self.console.print(f"[red]Error saving state: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")

    def _reset_crossover_state(self):
        """Reset crossover tracking state"""
        self.crossover_detected = False
        self.crossover_type = None
        self.crossover_price = None
        self.crossover_price_index = -1
        if hasattr(self, 'crossover_time'):
            delattr(self, 'crossover_time')
        self.save_state()  # Save the reset state

    def initialize_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            self.console.print("[bold red]MT5 initialization failed![/bold red]")
            return False
        
        # Login to MT5 (use your credentials)
        if not mt5.login(login=182992485, password="QAZwsx456!", server="Exness-MT5Trial6"):
            self.console.print("[bold red]MT5 login failed![/bold red]")
            mt5.shutdown()
            return False
        print("LOADED .... ")
        
        # Check symbol
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            self.console.print(f"[bold red]{self.symbol} not found![/bold red]")
            mt5.shutdown()
            return False
        
        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                self.console.print(f"[bold red]{self.symbol} symbol selection failed![/bold red]")
                mt5.shutdown()
                return False
            
        self.console.print("[bold green]MT5 initialized successfully![/bold green]")
        return True

    def calculate_indicators(self, df):
        """Calculate EMAs only"""
        try:
            # Calculate EMAs
            df['ema_short'] = ta.ema(df['close'], length=self.ema_short)
            df['ema_long'] = ta.ema(df['close'], length=self.ema_long)
            
            # Debug info
            last_row = df.iloc[-1]
            self.console.print("\n=== Latest Indicator Values ===")
            self.console.print(f"EMA9: {last_row['ema_short']:.2f}")
            self.console.print(f"EMA20: {last_row['ema_long']:.2f}")
            
            return df
            
        except Exception as e:
            self.console.print(f"[bold red]Error calculating indicators: {str(e)}[/bold red]")
            return None

    def get_trading_signal(self, df):
        """Generate trading signals based on EMA crossover with proper next-candle execution"""
        if df.empty or len(df) < 2:
            self.console.print("[red]Not enough data for signal generation[/red]")
            return None, None
            
        # Get last rows for detailed analysis
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # Check if trade is active
        active_trades = self.get_active_trades()
        if active_trades:
            self.console.print("[yellow]Trade already active, skipping signal check[/yellow]")
            return None, None

        # Check current EMA positions with DETAILED OUTPUT
        current_ema_diff = last_row['ema_short'] - last_row['ema_long']
        prev_ema_diff = prev_row['ema_short'] - prev_row['ema_long']
        
        self.console.print("\n=== DETAILED Signal Analysis ===")
        self.console.print(f"Current Time: {last_row['time']}")
        self.console.print(f"Current Price: {last_row['close']:.2f}")
        self.console.print(f"Previous EMA9: {prev_row['ema_short']:.2f}")
        self.console.print(f"Previous EMA20: {prev_row['ema_long']:.2f}")
        self.console.print(f"Current EMA9: {last_row['ema_short']:.2f}")
        self.console.print(f"Current EMA20: {last_row['ema_long']:.2f}")
        self.console.print(f"Previous EMA Difference: {prev_ema_diff:.2f}")
        self.console.print(f"Current EMA Difference: {current_ema_diff:.2f}")
        
        # ADDED DEBUG STATE DISPLAY
        self.console.print("\n=== Current Signal State ===")
        self.console.print(f"Crossover Detected: {self.crossover_detected}")
        self.console.print(f"Crossover Type: {self.crossover_type}")
        self.console.print(f"Crossover Price: {self.crossover_price}")
        self.console.print(f"Crossover Index: {self.crossover_price_index}")
        
        # If we have a crossover_detected state, check if we should execute on this candle
        if self.crossover_detected:
            self.console.print("[cyan]Crossover previously detected, checking if we should execute trade now...[/cyan]")
            
            # Make sure we have the crossover_time attribute
            if not hasattr(self, 'crossover_time'):
                # If missing, add current timestamp as crossover_time
                import datetime
                self.crossover_time = datetime.datetime.now().isoformat()
                self.console.print(f"[yellow]Added missing crossover_time: {self.crossover_time}[/yellow]")
                self.save_state()
            
            crossover_time_dt = pd.to_datetime(self.crossover_time)
            self.console.print(f"Crossover Time: {crossover_time_dt}")
            
            # Check if the latest candle is newer than the crossover candle
            time_diff = last_row['time'] - crossover_time_dt
            self.console.print(f"Time difference: {time_diff}")
            
            if last_row['time'] > crossover_time_dt:
                self.console.print(f"[green]✓ Current candle time ({last_row['time']}) is after crossover time ({crossover_time_dt})[/green]")
                
                # ADDED: Validate that EMA positions still support the crossover type
                is_valid = False
                if self.crossover_type == "BUY":
                    # For a BUY signal, EMA9 should still be above EMA20
                    if current_ema_diff > 0:
                        self.console.print("[green]✓ CONFIRMED: EMA9 is still above EMA20 on this candle[/green]")
                        is_valid = True
                    else:
                        self.console.print("[red]✗ INVALID: EMA9 is no longer above EMA20 - cancelling BUY signal[/red]")
                else:  # SELL signal
                    # For a SELL signal, EMA9 should still be below EMA20
                    if current_ema_diff < 0:
                        self.console.print("[green]✓ CONFIRMED: EMA9 is still below EMA20 on this candle[/green]")
                        is_valid = True
                    else:
                        self.console.print("[red]✗ INVALID: EMA9 is no longer below EMA20 - cancelling SELL signal[/red]")
                
                if is_valid:
                    signal = self.crossover_type
                    self.console.print(f"[green]✓ {signal} TRADE WILL BE PLACED: This is the next candle after crossover with confirmed EMA positions[/green]")
                    return signal, "Executing on next candle after crossover with confirmed EMA positions"
                else:
                    self.console.print("[yellow]Crossover signal invalidated due to EMA position change. Resetting state.[/yellow]")
                    self._reset_crossover_state()
                    return None, "Crossover invalidated"
            else:
                self.console.print(f"[yellow]Still on same candle as crossover - waiting for next candle[/yellow]")
                return None, "Waiting for next candle"
        
        # If no existing crossover detected, check for a new one
        buy_cross = (prev_ema_diff <= 0 and current_ema_diff > 0)
        sell_cross = (prev_ema_diff >= 0 and current_ema_diff < 0)
        
        self.console.print(f"Buy Cross: {buy_cross}")
        self.console.print(f"Sell Cross: {sell_cross}")
        
        if buy_cross or sell_cross:
            # If this is a new crossover
            self.crossover_detected = True
            self.crossover_type = "BUY" if buy_cross else "SELL"
            self.crossover_price = last_row['close']
            # Store both index and timestamp of crossover candle
            self.crossover_price_index = len(df) - 1
            self.crossover_time = last_row['time'].isoformat()
            self.save_state()
            
            self.console.print(f"[green]✓ {self.crossover_type} SIGNAL DETECTED: EMA9 crossed {'above' if buy_cross else 'below'} EMA20![/green]")
            self.console.print("[yellow]Waiting for next candle to place trade...[/yellow]")
            return None, "Waiting for next candle"
        
        # If in a normal candle (not a crossover or next candle after crossover)
        else:
            if current_ema_diff > 0:
                self.console.print("[yellow]EMA9 is above EMA20 - Waiting for crossover[/yellow]")
            else:
                self.console.print("[yellow]EMA9 is below EMA20 - Waiting for crossover[/yellow]")
                
        return None, None

    def calculate_sl_tp(self, signal, df):
        """Calculate Stop Loss and Take Profit based on percentages"""
        current_price = df.iloc[-1]['close']
        
        if signal == "BUY":
            # Stop loss at 0.25% below entry
            sl_price = current_price * (1 - self.initial_sl_percent)
            
            # Take profits at specified percentages
            tp1_price = current_price * (1 + self.tp1_percent)
            tp2_price = current_price * (1 + self.tp2_percent)
            tp3_price = current_price * (1 + self.tp3_percent)
            
        else:  # SELL
            # Stop loss at 0.25% above entry
            sl_price = current_price * (1 + self.initial_sl_percent)
            
            # Take profits at specified percentages
            tp1_price = current_price * (1 - self.tp1_percent)
            tp2_price = current_price * (1 - self.tp2_percent)
            tp3_price = current_price * (1 - self.tp3_percent)
        
        self.console.print("\n=== Trade Levels ===")
        self.console.print(f"Entry: ${current_price:.2f}")
        self.console.print(f"SL: ${sl_price:.2f} ({self.initial_sl_percent*100:.2f}%)")
        self.console.print(f"TP1 (0.50 lots): ${tp1_price:.2f} ({self.tp1_percent*100:.2f}%)")
        self.console.print(f"TP2 (0.20 lots): ${tp2_price:.2f} ({self.tp2_percent*100:.2f}%)")
        self.console.print(f"TP3 (0.20 lots): ${tp3_price:.2f} ({self.tp3_percent*100:.2f}%)")
        self.console.print(f"TP4 (0.10 lots): Trailing")
        
        return sl_price, tp1_price, tp2_price, tp3_price

    def execute_trade(self, signal, df):
        """Execute trade with a single position of 1 lot (instead of multiple positions)"""
        try:
            self.console.print(f"[bold cyan]ATTEMPTING TO EXECUTE {signal} TRADE[/bold cyan]")
            
            current_price = df.iloc[-1]['close']
            sl_price, tp1_price, tp2_price, tp3_price = self.calculate_sl_tp(signal, df)
            
            # Get current market price
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                self.console.print("[red]Failed to get current price[/red]")
                return False
            
            # Use appropriate price for entry
            price = tick.ask if signal == "BUY" else tick.bid
            self.console.print(f"Current Bid: {tick.bid}, Ask: {tick.ask}")
            
            # Check account margin first
            account_info = mt5.account_info()
            if account_info is None:
                self.console.print("[red]Failed to get account info[/red]")
                return False
            
            # Calculate total required margin
            margin = mt5.order_calc_margin(
                mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
                self.symbol,
                self.lot_size,  # Using full 1.0 lot size
                price
            )
            
            if margin is None:
                self.console.print("[red]Failed to calculate margin[/red]")
                return False
                
            if margin > account_info.margin_free:
                self.console.print(f"[red]Not enough margin. Required: ${margin:.2f}, Available: ${account_info.margin_free:.2f}[/red]")
                return False
            
            # FIXED: Place a single trade with 1.0 lot (instead of 4 separate trades)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
                "volume": self.lot_size,  # Full 1.0 lot
                "price": price,
                "sl": sl_price,  # Only set initial SL, no TP (we'll handle exits manually)
                "deviation": 20,
                "magic": 234000,
                "comment": f"Python {signal} 1.0 lot",
                "type_filling": mt5.ORDER_FILLING_IOC,
                "type_time": mt5.ORDER_TIME_GTC
            }
            
            # Display order details
            self.console.print("\n[bold cyan]ORDER DETAILS:[/bold cyan]")
            for key, value in request.items():
                self.console.print(f"  {key}: {value}")
            
            # Send the order
            result = mt5.order_send(request)
            
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                error_code = result.retcode if result else "None"
                error_desc = result.comment if result else "No result returned"
                self.console.print(f"[red]Trade failed! Code: {error_code}, Description: {error_desc}[/red]")
                return False
            
            self.console.print(f"[green]Trade placed successfully! Ticket: {result.order}[/green]")
            
            # Initialize exit tracking for this trade
            position = mt5.positions_get(ticket=result.order)[0]
            
            # Store information about this trade for exit management
            self.active_trade_info = {
                'ticket': result.order,
                'type': signal,
                'entry_price': price,
                'sl': sl_price,
                'original_volume': self.lot_size,
                'remaining_volume': self.lot_size,
                'exits_triggered': [False, False, False],  # For 0.25%, 0.50%, 0.75% exits
                'partial_lots': self.partial_lots,  # [0.50, 0.20, 0.20, 0.10]
                'entry_time': datetime.now()
            }
            
            self.console.print(f"[bold green]✓ {signal} order for 1.0 lot placed successfully![/bold green]")
            self.console.print("[yellow]Will manage partial exits at 0.25%, 0.50%, and 0.75% profit targets.[/yellow]")
            
            # IMPORTANT: Reset crossover state after successful trade execution
            self._reset_crossover_state()
            self.console.print("[green]Crossover state has been reset after trade execution[/green]")
            
            return True
            
        except Exception as e:
            self.console.print(f"[bold red]Trade execution error: {str(e)}[/bold red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
            return False

    def check_exit_conditions(self, df):
        """Check if exit conditions are met and manage partial exits for the single position"""
        # Get active positions
        positions = self.get_active_trades()
        if not positions:
            return False, None
            
        # Get current price
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            self.console.print("[red]Failed to get current price in exit conditions check[/red]")
            return False, None
        
        for position in positions:
            # If we have active_trade_info, use it, otherwise create it
            if not hasattr(self, 'active_trade_info') or self.active_trade_info['ticket'] != position.ticket:
                # Initialize tracking for this position
                pos_type = "BUY" if position.type == 0 else "SELL"
                self.active_trade_info = {
                    'ticket': position.ticket,
                    'type': pos_type,
                    'entry_price': position.price_open,
                    'sl': position.sl,
                    'original_volume': position.volume,
                    'remaining_volume': position.volume,
                    'exits_triggered': [False, False, False],  # For 0.25%, 0.50%, 0.75% exits
                    'partial_lots': self.partial_lots  # [0.50, 0.20, 0.20, 0.10]
                }
                self.console.print(f"[yellow]Initialized tracking for existing position {position.ticket}[/yellow]")
            
            # Current price based on position type
            current_price = tick.last
            
            # Calculate profit percentage
            if position.type == 0:  # BUY
                profit_percent = (current_price - position.price_open) / position.price_open
            else:  # SELL
                profit_percent = (position.price_open - current_price) / position.price_open
            
            self.console.print(f"Current profit: {profit_percent*100:.4f}%")
            
            # Check for partial exits based on our partial exit rules
            if not self.active_trade_info['exits_triggered'][0] and profit_percent >= self.tp1_percent:
                # First exit: 0.50 lots at 0.25% profit
                self.console.print(f"[bold green]TP1 reached at {profit_percent*100:.4f}%[/bold green]")
                if self.close_partial_position(position.ticket, self.partial_lots[0], f"TP1 at {self.tp1_percent*100:.2f}% profit"):
                    self.active_trade_info['exits_triggered'][0] = True
            
            elif self.active_trade_info['exits_triggered'][0] and not self.active_trade_info['exits_triggered'][1] and profit_percent >= self.tp2_percent:
                # Second exit: 0.20 lots at 0.50% profit
                self.console.print(f"[bold green]TP2 reached at {profit_percent*100:.4f}%[/bold green]")
                if self.close_partial_position(position.ticket, self.partial_lots[1], f"TP2 at {self.tp2_percent*100:.2f}% profit"):
                    self.active_trade_info['exits_triggered'][1] = True
            
            elif self.active_trade_info['exits_triggered'][0] and self.active_trade_info['exits_triggered'][1] and not self.active_trade_info['exits_triggered'][2] and profit_percent >= self.tp3_percent:
                # Third exit: 0.20 lots at 0.75% profit
                self.console.print(f"[bold green]TP3 reached at {profit_percent*100:.4f}%[/bold green]")
                if self.close_partial_position(position.ticket, self.partial_lots[2], f"TP3 at {self.tp3_percent*100:.2f}% profit"):
                    self.active_trade_info['exits_triggered'][2] = True
                    
                    # After the third exit, move SL to breakeven for remaining 0.10 lot
                    if position.sl != position.price_open:
                        self.modify_position_sl(position.ticket, position.price_open)
                        self.active_trade_info['sl'] = position.price_open
                        self.console.print("[green]Moving stop loss to breakeven for remaining position[/green]")
            
            # Check if we need to activate trailing stop for the final 0.10 lot
            if all(self.active_trade_info['exits_triggered']) and profit_percent >= self.trailing_activation_percent:
                # Calculate trailing stop levels
                self.check_trailing_stop(position)
        
        return False, None  # We don't use crossover for exit anymore

    def display_bot_status(self):
        """Display comprehensive bot status"""
        self.console.print("\n=== BTC TRADING BOT STATUS ===", style="bold green")
        
        # Configuration table
        config_table = Table(title="Configuration")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", justify="right")
        
        config_table.add_row("Symbol", self.symbol)
        config_table.add_row("Timeframe", str(self.timeframe))
        config_table.add_row("EMA Short", str(self.ema_short))
        config_table.add_row("EMA Long", str(self.ema_long))
        config_table.add_row("Min Cross Distance", str(self.min_cross_distance))
        config_table.add_row("Confirmation Period", str(self.confirmation_period))
        
        self.console.print(config_table)
        
        # Display current market data and trade status
        self.display_market_status()

    def run(self):
        """Main bot loop with status display"""
        self.console.print("[bold green]Starting ETH Trading Bot...[/bold green]")
        
        # Track the last processed candle time to detect new candles
        last_candle_time = None
        # Track if we're in "high alert" mode waiting for a next candle after crossover
        waiting_for_next_candle = False
        
        while self.running:
            try:
                # When waiting for next candle after crossover, use much shorter sleep time
                check_interval = 0.2 if waiting_for_next_candle else 1
                
                # ADDED: State verification at start of loop
                self.console.print("\n=== State Verification ===")
                self.console.print(f"crossover_detected: {self.crossover_detected}")
                self.console.print(f"crossover_type: {self.crossover_type}")
                if hasattr(self, 'crossover_time'):
                    self.console.print(f"crossover_time: {self.crossover_time}")
                    waiting_for_next_candle = self.crossover_detected
                
                # Clear screen for clean display - DISABLED TEMPORARILY FOR DEBUGGING
                # self.console.clear()
                self.console.print("\n" + "="*50 + "\n" + "NEW ITERATION" + "\n" + "="*50)
                
                # Display performance metrics at the top
                self.display_performance()

                # AFTER CLEAR: Re-verify state (to check if clear reset it)
                self.console.print("\n=== State After Clear ===")
                self.console.print(f"crossover_detected: {self.crossover_detected}")
                self.console.print(f"crossover_type: {self.crossover_type}")
                if hasattr(self, 'crossover_time'):
                    self.console.print(f"crossover_time: {self.crossover_time}")
                
                # Get market data
                df = self.get_rates_df(5000)
                if df is None:
                    continue
                    
                # Calculate indicators
                df = self.calculate_indicators(df)
                
                # Check for new candle
                current_candle_time = df.iloc[-1]['time']
                new_candle_detected = False
                
                if last_candle_time is not None and current_candle_time > last_candle_time:
                    new_candle_detected = True
                    self.console.print(f"[bold cyan]NEW CANDLE DETECTED! Previous: {last_candle_time}, Current: {current_candle_time}[/bold cyan]")
                
                # Update last candle time
                last_candle_time = current_candle_time
                
                # Display bot status
                self.display_bot_status()
                
                # Check if we're in trading hours
                in_trading_hours = self.is_trading_hours()
                if not in_trading_hours:
                    self.console.print("[yellow]Outside trading hours - monitoring only[/yellow]")
                
                # Check active trades
                active_trades = self.get_active_trades()
                
                # DEBUG: Print detailed crossover state and candle information
                if self.crossover_detected and df is not None and len(df) > 0 and hasattr(self, 'crossover_time'):
                    last_row = df.iloc[-1]
                    crossover_time_dt = pd.to_datetime(self.crossover_time)
                    self.console.print(f"\n[bold cyan]CROSSOVER STATE DEBUG:[/bold cyan]")
                    self.console.print(f"Current candle time: {last_row['time']}")
                    self.console.print(f"Crossover time: {crossover_time_dt}")
                    self.console.print(f"Current > Crossover: {last_row['time'] > crossover_time_dt}")
                    self.console.print(f"Time difference: {last_row['time'] - crossover_time_dt}")
                    self.console.print(f"Waiting for next candle: {waiting_for_next_candle}")
                    self.console.print(f"New candle detected: {new_candle_detected}")
                
                # SPECIAL HANDLING: If we're waiting for a new candle after crossover and one arrives, execute immediately
                if waiting_for_next_candle and new_candle_detected and in_trading_hours and not active_trades:
                    self.console.print("[bold cyan]NEW CANDLE AFTER CROSSOVER DETECTED! Checking for trade execution...[/bold cyan]")
                    signal, reason = self.get_trading_signal(df)
                    
                    if signal:
                        self.console.print(f"[bold green]IMMEDIATE SIGNAL: {signal} - {reason}[/bold green]")
                        self.console.print("[bold green]Executing trade at the opening of this candle![/bold green]")
                        execution_result = self.execute_trade(signal, df)
                        self.console.print(f"[bold {'green' if execution_result else 'red'}]Trade execution {'succeeded' if execution_result else 'failed'}[/bold {'green' if execution_result else 'red'}]")
                        waiting_for_next_candle = False
                    else:
                        self.console.print("[yellow]Signal was invalidated upon new candle arrival (EMA positions changed)[/yellow]")
                        waiting_for_next_candle = False
                
                # FIXED: Only reset state if not legitimately waiting for next candle after crossover
                if not active_trades and self.crossover_detected:
                    # Check if we're legitimately waiting for the next candle
                    if df is not None and len(df) > 0 and hasattr(self, 'crossover_time'):
                        last_row = df.iloc[-1]
                        crossover_time_dt = pd.to_datetime(self.crossover_time)
                        
                        # If we're on the same candle as the crossover, we're legitimately waiting
                        if last_row['time'] <= crossover_time_dt:
                            self.console.print("[cyan]Crossover detected, waiting for next candle to execute trade...[/cyan]")
                            waiting_for_next_candle = True  # Enable high-frequency checking
                        else:
                            # We've already seen the next candle but didn't execute the trade
                            # Instead of auto-resetting, let's try getting a signal first
                            self.console.print("[yellow]Next candle detected after crossover, checking if we should execute trade...[/yellow]")
                            signal, reason = self.get_trading_signal(df)
                            
                            if signal:
                                self.console.print(f"[bold green]SIGNAL: {signal} - {reason}[/bold green]")
                                execution_result = self.execute_trade(signal, df)
                                self.console.print(f"[bold {'green' if execution_result else 'red'}]Trade execution {'succeeded' if execution_result else 'failed'}[/bold {'green' if execution_result else 'red'}]")
                                waiting_for_next_candle = False
                            else:
                                # If no signal returned, now we can reset
                                self.console.print("[yellow]No active trades and crossover state is active, but next candle already arrived with no signal. Forcing state reset.[/yellow]")
                                self._reset_crossover_state()
                                waiting_for_next_candle = False
                    else:
                        # If we can't determine if we're waiting for next candle, leave state alone
                        self.console.print("[yellow]Crossover detected with no active trades, but unable to verify candle timing. Leaving state intact.[/yellow]")
                
                # Continue with normal processing for exit conditions
                if active_trades:
                    # UPDATED: Check for partial exits and manage existing positions
                    self.console.print("[cyan]Checking exit conditions for active trades...[/cyan]")
                    self.check_exit_conditions(df)
                    
                    # Also check trailing stops
                    for trade in active_trades:
                        self.check_trailing_stop(trade)
                    
                # Only check for new signals if not in special waiting mode
                elif in_trading_hours and not waiting_for_next_candle and not self.crossover_detected:
                    signal, reason = self.get_trading_signal(df)
                    if signal:
                        self.console.print(f"[bold green]SIGNAL: {signal} - {reason}[/bold green]")
                        execution_result = self.execute_trade(signal, df)
                        self.console.print(f"[bold {'green' if execution_result else 'red'}]Trade execution {'succeeded' if execution_result else 'failed'}[/bold {'green' if execution_result else 'red'}]")
                else:
                    # ADDED: Debug why we're not checking for signals
                    self.console.print("\n=== Trade Check Skipped Because: ===")
                    if not in_trading_hours:
                        self.console.print("[yellow]Outside of trading hours[/yellow]")
                    if active_trades:
                        self.console.print("[yellow]Active trades exist[/yellow]")
                    if waiting_for_next_candle:
                        self.console.print("[cyan]Waiting for next candle after crossover[/cyan]")
                    if self.crossover_detected and not waiting_for_next_candle:
                        self.console.print("[cyan]Crossover detected but not waiting for next candle[/cyan]")
                
                # Save state at the end of each iteration
                if self.crossover_detected:
                    self.save_state()
                
                # Adaptive sleep time - much shorter when waiting for next candle
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                self.console.print("[yellow]Bot shutdown requested...[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[bold red]Error: {str(e)}[/bold red]")
                import traceback
                self.console.print(f"[red]{traceback.format_exc()}[/red]")
                time.sleep(10)  # Longer pause on error

    def close_trade(self, position, reason="Manual"):
        """Close trade with reason logging"""
        try:
            # Get current price
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                self.console.print("[red]Failed to get current price[/red]")
                return False
            
            close_price = tick.bid if position.type == 1 else tick.ask
            profit_pips = close_price - position.price_open  # For BTC, 1 pip = $1
            if position.type == 1:
                profit_pips = -profit_pips
                
            close_details = {
                'timestamp': datetime.now(),
                'type': "BUY" if position.type == 0 else "SELL",
                'entry': position.price_open,
                'exit': close_price,
                'profit': position.profit,
                'profit_pips': profit_pips,
                'exit_reason': reason,
                'trade_duration': str(datetime.now() - pd.to_datetime(position.time) if 'time' in position else 'N/A')
            }
            
            # Close the position
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                'symbol': self.symbol,
                'volume': position.volume,
                'type': mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                'position': position.ticket,
                'price': close_price,
                'deviation': 20,
                'magic': 234000,
                'comment': f"Python close: {reason}",
                'type_filling': mt5.ORDER_FILLING_FOK,
                'type_time': mt5.ORDER_TIME_GTC
            }
            
            # Debug info
            self.console.print("\nOrder Details:")
            for key, value in request.items():
                self.console.print(f"{key}: {value}")
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.console.print("[green]Trade closed successfully![/green]")
                self.console.print(f"Profit: ${position.profit:.2f} ({profit_pips:.1f} pips)")
                self._reset_crossover_state()  # Reset crossover state after trade closure
                return True
            else:
                self.console.print(f"[red]Failed to close trade: {result.comment} (code: {result.retcode})[/red]")
                return False
                
        except Exception as e:
            self.console.print(f"[red]Error closing trade: {str(e)}[/red]")
            return False

    def is_market_open(self):
        """Check if the market is currently open (BTC trades 24/7)"""
        # For BTC, market is always open
        self.console.print("[green]BTC market is open 24/7[/green]")
        return True

    def get_rates_df(self, count=5000):
        """Get rates from MT5 and convert to DataFrame"""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, count)
            
            if rates is None:
                error_code = mt5.last_error()
                self.console.print(f"[bold red]Failed to get market data! Error code: {error_code}[/bold red]")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            if df.empty:
                self.console.print("[red]Empty DataFrame received[/red]")
                return None
            
            return df
            
        except Exception as e:
            self.console.print(f"[bold red]Error getting rates: {str(e)}[/bold red]")
            return None

    def display_market_status(self):
        """Display current market status"""
        try:
            # Create a table for market status
            market_table = Table(show_header=True, header_style="bold magenta")
            market_table.add_column("Parameter", style="dim")
            market_table.add_column("Value")
            market_table.add_column("Status")
            
            # Get current market data
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.console.print("[red]Failed to get symbol info[/red]")
                return
                
            # Add market data to table
            market_table.add_row(
                "Symbol",
                self.symbol,
                "[green]Active[/green]" if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL else "[red]Inactive[/red]"
            )
            
            market_table.add_row(
                "Spread",
                f"{(symbol_info.ask - symbol_info.bid):.2f}",
                "[yellow]High[/yellow]" if (symbol_info.ask - symbol_info.bid) > 5 else "[green]Normal[/green]"
            )
            
            # Display active trades
            if self.active_trades:
                market_table.add_row(
                    "Active Trades",
                    f"{len(self.active_trades)} positions",
                    "[green]Yes[/green]"
                )
                
                trade_table = Table(title="Active Trades")
                trade_table.add_column("Ticket", justify="right")
                trade_table.add_column("Type", justify="right")
                trade_table.add_column("Current Price", justify="right")
                trade_table.add_column("SL", justify="right")
                trade_table.add_column("TP", justify="right")
                trade_table.add_column("Profit", justify="right")
                trade_table.add_column("Profit (pips)", justify="right")
                
                for position in self.active_trades:
                    # Access position as named tuple
                    pos_type = "BUY" if position.type == 0 else "SELL"
                    profit_pips = position.profit / self.lot_size
                    
                    trade_table.add_row(
                        str(position.ticket),
                        pos_type,
                        f"${position.price_current:.2f}",
                        f"${position.sl:.2f}" if position.sl else "None",
                        f"${position.tp:.2f}" if position.tp else "None",
                        f"${position.profit:.2f}",
                        f"{profit_pips:.1f} pips"
                    )
            else:
                market_table.add_row(
                    "Active Trades",
                    "No positions",
                    "[yellow]Idle[/yellow]"
                )
            
            self.console.print(market_table)
            if self.active_trades:
                self.console.print(trade_table)
            
        except Exception as e:
            self.console.print(f"[red]Error displaying market status: {str(e)}[/red]")

    def get_active_trades(self):
        """Get current active trades with detailed error handling"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            
            if positions is None:
                error_code = mt5.last_error()
                if error_code:
                    self.console.print(f"[yellow]No positions found or error: {error_code}[/yellow]")
                else:
                    self.console.print("[yellow]No active trades[/yellow]")
                return []
                
            if len(positions) == 0:
                self.console.print("[yellow]No active trades[/yellow]")
                return []
                
            # ENHANCED: Actually return the position objects directly for better compatibility
            self.console.print(f"[green]Found {len(positions)} active trade(s)[/green]")
            for position in positions:
                self.console.print(f"  Ticket: {position.ticket}, Type: {'BUY' if position.type == 0 else 'SELL'}, Comment: {position.comment}, Profit: ${position.profit:.2f}")
            
            return positions
            
        except Exception as e:
            self.console.print(f"[red]Error getting active trades: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
            return []
            
    def manage_trailing_stop(self, position):
        """Manage trailing stop loss for a position"""
        current_price = mt5.symbol_info_tick(self.symbol).last
        entry_price = position.price_open
        current_sl = position.sl
        
        # Calculate profit percentage
        if position.type == mt5.POSITION_TYPE_BUY:
            profit_percent = (current_price - entry_price) / entry_price
            
            # Move SL to breakeven at 0.75% profit
            if profit_percent >= self.trailing_activation_percent and current_sl < entry_price:
                self.modify_position_sl(position.ticket, entry_price)
                return
            
            # After breakeven, move SL up every 0.25% profit
            elif profit_percent >= self.trailing_activation_percent:
                steps_above_activation = (profit_percent - self.trailing_activation_percent) / self.trailing_step_percent
                if steps_above_activation > 0:
                    sl_steps = int(steps_above_activation)
                    new_sl = entry_price * (1 + (sl_steps * self.trailing_step_percent))
                    if new_sl > current_sl:
                        self.modify_position_sl(position.ticket, new_sl)
                        
        else:  # SELL position
            profit_percent = (entry_price - current_price) / entry_price
            
            # Move SL to breakeven at 0.75% profit
            if profit_percent >= self.trailing_activation_percent and current_sl > entry_price:
                self.modify_position_sl(position.ticket, entry_price)
                return
            
            # After breakeven, move SL down every 0.25% profit
            elif profit_percent >= self.trailing_activation_percent:
                steps_above_activation = (profit_percent - self.trailing_activation_percent) / self.trailing_step_percent
                if steps_above_activation > 0:
                    sl_steps = int(steps_above_activation)
                    new_sl = entry_price * (1 - (sl_steps * self.trailing_step_percent))
                    if new_sl < current_sl:
                        self.modify_position_sl(position.ticket, new_sl)
                        
    def modify_position_sl(self, ticket, new_sl):
        """Modify stop loss for a position"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": new_sl
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.console.print(f"[red]Failed to modify SL. Error code: {result.retcode}[/red]")

    def is_trading_hours(self):
        """Check if current time is within trading hours (Eastern Time) - Updated for all days"""
        # Get current time in Eastern Time
        et_tz = pytz.timezone('America/New_York')
        current_time = datetime.now(et_tz)
        
        # Format for comparison
        current_time_str = current_time.strftime('%H:%M:%S')
        current_weekday = current_time.weekday()  # 0=Monday, 6=Sunday
        
        # Allow trading on all days
        if current_weekday == 6:  # Sunday
            # Only trade after 6 PM
            is_valid = current_time_str >= self.trading_start
            status_message = f"Sunday {'after' if is_valid else 'before'} 6PM ET - {current_time_str}"
            self.console.print(f"[{'green' if is_valid else 'yellow'}]{status_message}[/{'green' if is_valid else 'yellow'}]")
            return is_valid
            
        elif current_weekday == 4:  # Friday
            # Only trade until 5 PM
            is_valid = current_time_str <= self.trading_end
            status_message = f"Friday {'before' if is_valid else 'after'} 5PM ET - {current_time_str}"
            self.console.print(f"[{'green' if is_valid else 'yellow'}]{status_message}[/{'green' if is_valid else 'yellow'}]")
            return is_valid
            
        elif current_weekday == 5:  # Saturday - Now we trade on Saturday
            self.console.print(f"[green]Trading on Saturday - {current_time_str}[/green]")
            return True
            
        else:  # Monday-Thursday
            self.console.print(f"[green]Regular trading day - {current_time_str}[/green]")
            return True

    def log_trade(self, trade_details):
        """Log trade details for analysis"""
        try:
            log_file = f"trades_{datetime.now().strftime('%Y%m')}.csv"
            trade_data = {
                'timestamp': trade_details['timestamp'],
                'type': trade_details['type'],
                'entry': trade_details['entry'],
                'exit': trade_details['exit'],
                'sl': trade_details.get('sl', 0),
                'tp': trade_details.get('tp', 0),
                'volume': trade_details.get('volume', 0),
                'profit': trade_details.get('profit', 0),
                'profit_pips': trade_details.get('profit_pips', 0),
                'exit_reason': trade_details.get('exit_reason', '')
            }
            
            pd.DataFrame([trade_data]).to_csv(log_file, mode='a', header=not os.path.exists(log_file))
            self.console.print("[green]Trade logged successfully[/green]")
            
        except Exception as e:
            self.console.print(f"[red]Error logging trade: {str(e)}[/red]")

    def display_performance(self):
        """Display bot performance metrics"""
        try:
            log_file = f"trades_{datetime.now().strftime('%Y%m')}.csv"
            if not os.path.exists(log_file):
                return
            
            trades_df = pd.read_csv(log_file)
            
            metrics = Table(title="Performance Metrics")
            metrics.add_column("Metric")
            metrics.add_column("Value")
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['profit'] > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            metrics.add_row("Total Trades", str(total_trades))
            metrics.add_row("Win Rate", f"{win_rate:.2f}%")
            metrics.add_row("Total Profit", f"${trades_df['profit'].sum():.2f}")
            metrics.add_row("Largest Win", f"${trades_df['profit'].max():.2f}")
            metrics.add_row("Largest Loss", f"${trades_df['profit'].min():.2f}")
            
            self.console.print(metrics)
            
        except Exception as e:
            self.console.print(f"[red]Error displaying performance: {str(e)}[/red]")

    def check_trailing_stop(self, trade):
        """Implements the trailing stop as specified in the requirements"""
        try:
            # Get current price
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                self.console.print("[red]Failed to get current price in trailing stop check[/red]")
                return False
            
            # Handle both dictionary and position object formats
            if isinstance(trade, dict):
                # ENHANCED debugging for dictionary access
                self.console.print(f"[cyan]Position info (dict): {list(trade.keys())}[/cyan]")
                is_trailing = "Trailing" in trade.get('comment', '')
                trade_type = trade['type']
                entry_price = trade['open_price']
                sl = trade['sl']
                ticket = trade['ticket']
            else:
                # ENHANCED debugging for object access
                self.console.print(f"[cyan]Position info (object): {dir(trade)[:10]}...[/cyan]")
                is_trailing = "Trailing" in getattr(trade, 'comment', '')
                trade_type = "BUY" if trade.type == 0 else "SELL"
                entry_price = trade.price_open
                sl = trade.sl
                ticket = trade.ticket
            
            # Only process trailing stops for the trailing position
            if not is_trailing:
                return True
                
            current_price = tick.bid if trade_type == "BUY" else tick.ask
            
            # Check if position still exists - if not, reset state to ensure we can take new trades
            positions = mt5.positions_get(ticket=ticket)
            if positions is None or len(positions) == 0:
                self.console.print(f"[yellow]Position {ticket} no longer exists - likely hit SL/TP. Resetting crossover state.[/yellow]")
                self._reset_crossover_state()
                return True
            
            # For BUY positions
            if trade_type == "BUY":
                # Calculate profit as a percentage
                profit_pct = (current_price - entry_price) / entry_price
                
                # 1. When profit reaches 0.75%, move SL to 0.5% above entry
                if profit_pct >= self.trailing_activation_percent:
                    # Calculate the base trailing SL (0.5% above entry)
                    base_sl = entry_price * (1 + self.trailing_start_percent)
                    
                    # 2. For each additional 0.25% move, increase SL by 0.25%
                    additional_steps = int((profit_pct - self.trailing_activation_percent) / self.trailing_step_percent)
                    if additional_steps > 0:
                        # Calculate new SL with additional steps
                        new_sl = base_sl + (entry_price * additional_steps * self.trailing_step_percent)
                        
                        # Only update if new SL is higher than current SL
                        if new_sl > sl:
                            self.console.print(f"[yellow]Updating trailing stop: ${sl:.2f} -> ${new_sl:.2f}[/yellow]")
                            
                            # Send the order to update SL
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": ticket,
                                "symbol": self.symbol,
                                "sl": new_sl
                            }
                            
                            result = mt5.order_send(request)
                            if result.retcode != mt5.TRADE_RETCODE_DONE:
                                self.console.print(f"[red]Failed to update trailing stop: {result.comment}[/red]")
                            else:
                                self.console.print(f"[green]Trailing stop updated to ${new_sl:.2f}[/green]")
                
            # For SELL positions
            else:
                # Calculate profit as a percentage
                profit_pct = (entry_price - current_price) / entry_price
                
                # 1. When profit reaches 0.75%, move SL to 0.5% below entry
                if profit_pct >= self.trailing_activation_percent:
                    # Calculate the base trailing SL (0.5% below entry)
                    base_sl = entry_price * (1 - self.trailing_start_percent)
                    
                    # 2. For each additional 0.25% move, decrease SL by 0.25%
                    additional_steps = int((profit_pct - self.trailing_activation_percent) / self.trailing_step_percent)
                    if additional_steps > 0:
                        # Calculate new SL with additional steps
                        new_sl = base_sl - (entry_price * additional_steps * self.trailing_step_percent)
                        
                        # Only update if new SL is lower than current SL
                        if new_sl < sl:
                            self.console.print(f"[yellow]Updating trailing stop: ${sl:.2f} -> ${new_sl:.2f}[/yellow]")
                            
                            # Send the order to update SL
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": ticket,
                                "symbol": self.symbol,
                                "sl": new_sl
                            }
                            
                            result = mt5.order_send(request)
                            if result.retcode != mt5.TRADE_RETCODE_DONE:
                                self.console.print(f"[red]Failed to update trailing stop: {result.comment}[/red]")
                            else:
                                self.console.print(f"[green]Trailing stop updated to ${new_sl:.2f}[/green]")
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Error updating trailing stop: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
            return False

    def close_partial_position(self, position_ticket, volume_to_close, reason="Partial Profit Target"):
        """Close a portion of an existing position"""
        try:
            # Get full position details
            positions = mt5.positions_get(ticket=position_ticket)
            if positions is None or len(positions) == 0:
                self.console.print(f"[red]Position {position_ticket} not found for partial close[/red]")
                return False
                
            position = positions[0]
            
            # Get current price
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                self.console.print("[red]Failed to get current price for partial close[/red]")
                return False
                
            # Determine close price based on position type
            close_price = tick.bid if position.type == 0 else tick.ask  # Bid for BUY, Ask for SELL
            
            # Create request to close partial position
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume_to_close,
                "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,  # Opposite of position type
                "position": position_ticket,
                "price": close_price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Partial close: {reason}",
                "type_filling": mt5.ORDER_FILLING_IOC,
                "type_time": mt5.ORDER_TIME_GTC
            }
            
            # Log request details
            self.console.print(f"[bold yellow]PARTIAL CLOSE REQUEST ({volume_to_close} lots):[/bold yellow]")
            for key, value in request.items():
                self.console.print(f"  {key}: {value}")
                
            # Send the request
            result = mt5.order_send(request)
            
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                error_code = result.retcode if result else "None"
                error_desc = result.comment if result else "No result returned"
                self.console.print(f"[red]Partial close failed! Code: {error_code}, Description: {error_desc}[/red]")
                return False
                
            # Calculate profit for the closed portion
            profit_amount = (close_price - position.price_open) * volume_to_close
            if position.type == 1:  # For SELL positions, reverse profit calculation
                profit_amount = -profit_amount
                
            # Update our tracking for the active trade
            if hasattr(self, 'active_trade_info') and self.active_trade_info['ticket'] == position_ticket:
                self.active_trade_info['remaining_volume'] -= volume_to_close
                
            self.console.print(f"[bold green]Partial position closed successfully! {volume_to_close} lots at {close_price}[/bold green]")
            self.console.print(f"[green]Partial profit: ${profit_amount:.2f}[/green]")
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Error in partial close: {str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
            return False

    def get_current_price(self):
        """Get current price of the symbol"""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            self.console.print("[red]Failed to get current price[/red]")
            return None
        
        # Fix to handle empty active_trades list
        if not self.active_trades:
            return (tick.ask + tick.bid) / 2  # Return the mid price if no active trades
            
        return tick.ask if self.active_trades[0].type == 0 else tick.bid