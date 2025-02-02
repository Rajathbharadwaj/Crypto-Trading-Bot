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

class BTCTradingBot:
    def __init__(self, timeframe=mt5.TIMEFRAME_M15, name="BTCBot"):
        self.name = name
        self.timeframe = timeframe
        self.running = True
        self.console = Console()
        self.symbol = "BTCUSD"
        
        # Load saved state or initialize new state
        self.state_file = "bot_state.json"
        self.load_state()
        
        # State tracking variables
        self.crossover_detected = False
        self.crossover_type = None  # "BUY" or "SELL"
        self.crossover_price = None
        self.crossover_price_index = -1
        self.confirmation_window = 5  # Max 
        # Trading configuration
        self.lot_size = 0.5     # Fixed lot size
        self.trade_taken = False
        self.min_cross_distance = 0.005  # 0.5% minimum distance
        self.confirmation_period = 2
        self.rr_ratio = 3.0
        # Trailing stop configuration
        self.trailing_enabled = True
        self.trailing_activation_percent = 0.003  # 0.3% activation
        self.trailing_step_percent = 0.0015       # 0.15% step
        self.trailing_lock_percent = 0.4          # 40% profit lock
        
        # Strategy Parameters
        self.ema_short = 9      # Short EMA period
        self.ema_long = 20      # Long EMA period
        self.lookback_candles = 3  # Number of candles to look back for SL
        
        # Trading hours (IST)
        self.trading_start = "11:00:00"
        self.trading_end = "02:30:00"
        
        # Initialize active trades
        self.active_trades = []
        
        if not self.initialize_mt5():
            raise Exception("MT5 initialization failed!")
    
    def load_state(self):
        """Load bot state from file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.crossover_detected = state.get('crossover_detected', False)
                self.crossover_type = state.get('crossover_type', None)
                self.crossover_price = state.get('crossover_price', None)
                self.crossover_price_index = state.get('crossover_price_index', -1)
            else:
                self._reset_crossover_state()
        except Exception as e:
            self.console.print(f"[red]Error loading state: {str(e)}[/red]")
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
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            self.console.print(f"[red]Error saving state: {str(e)}[/red]")

    def _reset_crossover_state(self):
        """Reset crossover tracking state"""
        self.crossover_detected = False
        self.crossover_type = None
        self.crossover_price = None
        self.crossover_price_index = -1
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
        """Get trading signal based on EMA crossover"""
        if len(df) < 2:
            return None, "Not enough data"
            
        # Get last two rows for crossover detection
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # Check current EMA positions
        current_ema_diff = last_row['ema_short'] - last_row['ema_long']
        prev_ema_diff = prev_row['ema_short'] - prev_row['ema_long']
        
        # Initialize signal variables
        buy_cross = False
        sell_cross = False
        
        # Print current price and required movement
        current_price = last_row['close']
        required_movement = current_price * self.min_cross_distance
        
        self.console.print("\n=== Trade Analysis ===")
        self.console.print(f"Current Price: ${current_price:.2f}")
        self.console.print(f"EMA9: ${last_row['ema_short']:.2f}")
        self.console.print(f"EMA20: ${last_row['ema_long']:.2f}")
        self.console.print(f"EMA Difference: ${current_ema_diff:.2f}")
        
        # Print trade conditions
        self.console.print("\n=== Trade Conditions ===")
        
        # Check for active trades
        active_trades = self.get_active_trades()
        if active_trades:
            self.console.print("[yellow]❌ No new trades: Active trade exists[/yellow]")
            return None, "Active trade exists"
            
        # Check if crossover was previously detected
        if self.crossover_detected:
            price_move_pct = abs(current_price - self.crossover_price) / self.crossover_price
            price_move_value = abs(current_price - self.crossover_price)
            trend_aligned = (
                (self.crossover_type == "BUY" and current_price > self.crossover_price) or
                (self.crossover_type == "SELL" and current_price < self.crossover_price)
            )
            
            self.console.print(f"\n=== Crossover Follow-up ===")
            self.console.print(f"Type: {self.crossover_type}")
            self.console.print(f"Entry Price: ${self.crossover_price:.2f}")
            self.console.print(f"Price Move: ${price_move_value:.2f} ({price_move_pct*100:.3f}%)")
            self.console.print(f"Required: ${required_movement:.2f} ({self.min_cross_distance*100:.3f}%)")
            self.console.print(f"Trend Aligned: {'[green]Yes[/green]' if trend_aligned else '[red]No[/red]'}")
            
            if not trend_aligned:
                self.console.print("[red]❌ Price moving against crossover direction[/red]")
            elif price_move_pct < self.min_cross_distance:
                self.console.print(f"[yellow]⌛ Waiting for {self.min_cross_distance*100:.3f}% move[/yellow]")
            else:
                self.console.print(f"[green]✓ Movement requirement met[/green]")
                signal = self.crossover_type
                self._reset_crossover_state()
                return signal, f"Price moved {price_move_pct*100:.2f}% since crossover"
                
        else:
            # Check for new crossover
            buy_cross = (prev_ema_diff < 0 and current_ema_diff > 0)
            sell_cross = (prev_ema_diff > 0 and current_ema_diff < 0)
            
            self.console.print("\n=== Crossover Detection ===")
            if not (buy_cross or sell_cross):
                if current_ema_diff > 0:
                    self.console.print("[yellow]⌛ EMA9 above EMA20 - Waiting for crossover[/yellow]")
                else:
                    self.console.print("[yellow]⌛ EMA9 below EMA20 - Waiting for crossover[/yellow]")
            else:
                self.crossover_detected = True
                self.crossover_type = "BUY" if buy_cross else "SELL"
                self.crossover_price = current_price
                self.crossover_price_index = len(df) - 1
                self.save_state()
                self.console.print(f"[green]✓ {self.crossover_type} Crossover Detected![/green]")
                return None, "Crossover detected - awaiting confirmation"
        
        return None, "No signal"

    def calculate_sl_tp(self, signal, df):
        """Calculate Stop Loss and Take Profit based on previous candles"""
        current_price = df.iloc[-1]['close']
        
        # Get the last 4 candles (current + 3 previous)
        last_candles = df.iloc[-4:]
        
        if signal == "BUY":
            # Stop loss at lowest low of previous 3 candles
            sl_price = last_candles['low'].iloc[:-1].min()  # Exclude current candle
            sl_distance = current_price - sl_price
            
            # Calculate take profit based on 1:3 risk reward
            tp_price = current_price + (sl_distance * self.rr_ratio)
            
        else:  # SELL
            # Stop loss at highest high of previous 3 candles
            sl_price = last_candles['high'].iloc[:-1].max()  # Exclude current candle
            sl_distance = sl_price - current_price
            
            # Calculate take profit based on 1:3 risk reward
            tp_price = current_price - (sl_distance * self.rr_ratio)
        
        self.console.print("\n=== Trade Levels ===")
        self.console.print(f"Entry: ${current_price:.2f}")
        self.console.print(f"SL: ${sl_price:.2f} ({(abs(sl_price - current_price)/current_price)*100:.2f}%)")
        self.console.print(f"TP: ${tp_price:.2f} ({(abs(tp_price - current_price)/current_price)*100:.2f}%)")
        
        return sl_price, tp_price

    def execute_trade(self, signal, df):
        """Execute a trade based on the signal"""
        current_price = df.iloc[-1]['close']
        sl_price, tp_price = self.calculate_sl_tp(signal, df)
        
        # Calculate lot size
        lot_size = self.lot_size
        
        # Prepare the trade request
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            self.console.print(f"[red]Failed to get symbol info for {self.symbol}[/red]")
            return False
            
        point = symbol_info.point
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": current_price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,
            "magic": 234000,
            "comment": f"Python trade: {signal}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Store additional trade details for trailing
        trade_details = {
            'ticket': None,  # Will be set after order execution
            'type': signal,
            'entry_price': current_price,
            'sl': sl_price,
            'tp': tp_price,
            'initial_sl': sl_price,  # Keep track of initial SL
            'initial_tp': tp_price,  # Keep track of initial TP
            'highest_price': current_price if signal == "BUY" else float('-inf'),
            'lowest_price': current_price if signal == "SELL" else float('inf'),
            'volume': lot_size
        }
        
        # Send order to MT5
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.console.print(f"[red]Order failed, retcode={result.retcode}[/red]")
            return False
            
        trade_details['ticket'] = result.order  # Store the ticket number
        self.active_trades.append(trade_details)  # Store trade details for trailing
        
        self.console.print(f"[green]✓ {signal} order placed successfully[/green]")
        return True

    def check_exit_conditions(self, df):
        """Check if any exit conditions are met"""
        if not self.trade_taken:
            return False, None
            
        active_trades = self.get_active_trades()
        if not active_trades:
            return False, None
            
        trade = active_trades[0]
        last_row = df.iloc[-1]
        
        exit_reason = None
        should_exit = False
        
        # Check EMA cross for exit
        if trade['type'] == "BUY" and last_row['ema_short'] < last_row['ema_long']:
            should_exit = True
            exit_reason = f"EMA cross bearish at {last_row['close']:.2f}"
        elif trade['type'] == "SELL" and last_row['ema_short'] > last_row['ema_long']:
            should_exit = True
            exit_reason = f"EMA cross bullish at {last_row['close']:.2f}"
        
        return should_exit, exit_reason

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
        config_table.add_row("Lookback Candles", str(self.lookback_candles))
        config_table.add_row("Min Cross Distance", str(self.min_cross_distance))
        config_table.add_row("Confirmation Period", str(self.confirmation_period))
        
        self.console.print(config_table)
        
        # Display current market data and trade status
        self.display_market_status()

    def run(self):
        """Main bot loop with status display"""
        self.console.print("[bold green]Starting BTC Trading Bot...[/bold green]")
        
        while self.running:
            try:
                # Clear screen for clean display
                self.console.clear()
                
                # Display performance metrics at the top
                self.display_performance()
                
                # Get market data
                df = self.get_rates_df(5000)
                if df is None:
                    continue
                    
                # Calculate indicators
                df = self.calculate_indicators(df)
                
                # Display bot status
                self.display_bot_status()
                
                # Check active trades
                active_trades = self.get_active_trades()
                if active_trades:
                    trade = active_trades[0]
                    
                    # Check trailing stop
                    self.check_trailing_stop(trade)
                    
                    # Check exit conditions
                    should_exit, exit_reason = self.check_exit_conditions(df)
                    if should_exit:
                        self.close_trade(trade, exit_reason)
                    
                # If no active trade, check for entry signals
                elif not self.trade_taken:
                    signal, reason = self.get_trading_signal(df)
                    if signal:
                        self.execute_trade(signal, df)
                
                time.sleep(1)
                
            except KeyboardInterrupt:
                self.console.print("[yellow]Bot shutdown requested...[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[bold red]Error: {str(e)}[/bold red]")
                time.sleep(5)

    def close_trade(self, position, reason="Manual"):
        """Close trade with reason logging"""
        try:
            # Get current price
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                self.console.print("[red]Failed to get current price[/red]")
                return False
            
            close_price = tick.bid if position.type == 0 else tick.ask
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
                "symbol": self.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": close_price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Python close: {reason}",
                "type_filling": mt5.ORDER_FILLING_FOK,
                "type_time": mt5.ORDER_TIME_GTC
            }
            
            # Debug info
            self.console.print("\nOrder Details:")
            for key, value in request.items():
                self.console.print(f"{key}: {value}")
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.console.print("[green]Trade closed successfully![/green]")
                self.console.print(f"Profit: ${position.profit:.2f} ({profit_pips:.1f} pips)")
                self.trade_taken = False
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
        """Retrieve active trades from MT5"""
        try:
            self.active_trades = mt5.positions_get(symbol=self.symbol) or []
            return self.active_trades
        except Exception as e:
            self.console.print(f"[red]Error getting active trades: {str(e)}[/red]")
            self.active_trades = []
            return []

    def is_trading_hours(self):
        """Check if current time is within trading hours (IST)"""
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist).strftime('%H:%M:%S')
        self.console.print(f"Current time: {current_time}")
        
        # Handle overnight trading hours (11:00 AM to 02:30 AM next day)
        if self.trading_start <= current_time or current_time <= self.trading_end:
            return True
        return False

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

    def check_trailing_stop(self, position):
        """Check and update trailing stop loss"""
        if not self.trailing_enabled:
            return
            
        # Get current bid/ask prices
        symbol_info = mt5.symbol_info_tick(self.symbol)
        if symbol_info is None:
            return
            
        # Use appropriate price based on position type
        current_price = symbol_info.bid if position.type == 1 else symbol_info.ask
            
        # Calculate minimum stop distance (use 5x the spread as minimum)
        min_stop_distance = (symbol_info.ask - symbol_info.bid) * 5
        
        # Debug info
        self.console.print(
            f"[blue]Debug Info:\n"
            f"  Bid: ${symbol_info.bid:.2f}\n"
            f"  Ask: ${symbol_info.ask:.2f}\n"
            f"  Spread: ${symbol_info.ask - symbol_info.bid:.2f}\n"
            f"  Min Stop Distance: ${min_stop_distance:.2f}\n"
            f"  Current Price: ${current_price:.2f}[/blue]"
        )

        if position.type == 0:  # BUY
            price_move = current_price - position.price_open
            move_pct = price_move / position.price_open
            
            if move_pct >= self.trailing_activation_percent:
                # Calculate new SL with minimum distance
                new_sl = min(
                    current_price - (current_price * self.trailing_step_percent),
                    current_price - min_stop_distance
                )
                
                if new_sl > position.sl:
                    new_tp = position.price_open + ((new_sl - position.price_open) * self.rr_ratio)
                    
                    # Round SL/TP to avoid floating point issues
                    new_sl = round(new_sl, 2)
                    new_tp = round(new_tp, 2)
                    
                    self.console.print(
                        f"[yellow]Trailing Update BUY @ ${current_price:.2f}:\n"
                        f"  SL Updated: ${position.sl:.2f} → ${new_sl:.2f}\n"
                        f"  TP Updated: ${position.tp:.2f} → ${new_tp:.2f}\n"
                        f"  Distance to SL: ${current_price - new_sl:.2f}[/yellow]"
                    )
                    
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,  # Only modify SL/TP
                        "position": position.ticket,
                        "symbol": self.symbol,
                        "sl": new_sl,
                        "tp": new_tp,
                        "magic": 234000
                    }
                    
                    self.console.print(f"[cyan]Sending request: {request}[/cyan]")
                    result = mt5.order_send(request)
                    
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        self.console.print(f"[red]Failed to update trailing stop: {result.comment} (code: {result.retcode})[/red]")
                        # If failed due to invalid stops, try with wider distance
                        if result.retcode == 10016:  # Invalid stops
                            new_sl = round(current_price - (min_stop_distance * 3), 2)
                            new_tp = round(position.price_open + ((new_sl - position.price_open) * self.rr_ratio), 2)
                            request["sl"] = new_sl
                            request["tp"] = new_tp
                            
                            self.console.print(f"[cyan]Retrying with request: {request}[/cyan]")
                            result = mt5.order_send(request)
                            
                            if result.retcode != mt5.TRADE_RETCODE_DONE:
                                self.console.print(f"[red]Failed retry with wider stops: {result.comment} (code: {result.retcode})[/red]")
                                # Try getting position info
                                positions = mt5.positions_get(ticket=position.ticket)
                                if positions:
                                    pos = positions[0]
                                    self.console.print(f"[blue]Current position state:\n  SL: ${pos.sl}\n  TP: ${pos.tp}\n  Price: ${pos.price_current}[/blue]")
                    
                    if move_pct >= self.trailing_lock_percent:
                        self.close_partial_position(position)
                        self.console.print(f"[green]Profit Lock: {self.trailing_lock_percent*100}% position closed at ${current_price:.2f}[/green]")
                        
        elif position.type == 1:  # SELL trade
            price_move = position.price_open - current_price
            move_pct = price_move / position.price_open
            
            if move_pct >= self.trailing_activation_percent:
                # Calculate new SL with minimum distance
                new_sl = max(
                    current_price + (current_price * self.trailing_step_percent),
                    current_price + min_stop_distance
                )
                
                if new_sl < position.sl:
                    new_tp = position.price_open - ((position.price_open - new_sl) * self.rr_ratio)
                    
                    # Round SL/TP to avoid floating point issues
                    new_sl = round(new_sl, 2)
                    new_tp = round(new_tp, 2)
                    
                    self.console.print(
                        f"[yellow]Trailing Update SELL @ ${current_price:.2f}:\n"
                        f"  SL Updated: ${position.sl:.2f} → ${new_sl:.2f}\n"
                        f"  TP Updated: ${position.tp:.2f} → ${new_tp:.2f}\n"
                        f"  Distance to SL: ${new_sl - current_price:.2f}[/yellow]"
                    )
                    
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,  # Only modify SL/TP
                        "position": position.ticket,
                        "symbol": self.symbol,
                        "sl": new_sl,
                        "tp": new_tp,
                        "magic": 234000
                    }
                    
                    self.console.print(f"[cyan]Sending request: {request}[/cyan]")
                    result = mt5.order_send(request)
                    
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        self.console.print(f"[red]Failed to update trailing stop: {result.comment} (code: {result.retcode})[/red]")
                        # If failed due to invalid stops, try with wider distance
                        if result.retcode == 10016:  # Invalid stops
                            new_sl = round(current_price + (min_stop_distance * 3), 2)
                            new_tp = round(position.price_open - ((position.price_open - new_sl) * self.rr_ratio), 2)
                            request["sl"] = new_sl
                            request["tp"] = new_tp
                            
                            self.console.print(f"[cyan]Retrying with request: {request}[/cyan]")
                            result = mt5.order_send(request)
                            
                            if result.retcode != mt5.TRADE_RETCODE_DONE:
                                self.console.print(f"[red]Failed retry with wider stops: {result.comment} (code: {result.retcode})[/red]")
                                # Try getting position info
                                positions = mt5.positions_get(ticket=position.ticket)
                                if positions:
                                    pos = positions[0]
                                    self.console.print(f"[blue]Current position state:\n  SL: ${pos.sl}\n  TP: ${pos.tp}\n  Price: ${pos.price_current}[/blue]")
                    
                    if move_pct >= self.trailing_lock_percent:
                        self.close_partial_position(position)
                        self.console.print(f"[green]Profit Lock: {self.trailing_lock_percent*100}% position closed at ${current_price:.2f}[/green]")

    def close_partial_position(self, position):
        """Close a portion of the trade"""
        try:
            # Calculate partial close volume
            partial_volume = position.volume * self.trailing_lock_percent
            
            # Prepare the trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": partial_volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": self.get_current_price(),
                "deviation": 20,
                "magic": 234000,
                "comment": "Partial close - profit lock",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.console.print(f"[red]Failed to execute partial close: {result.comment}[/red]")
                
        except Exception as e:
            self.console.print(f"[red]Error in partial close: {str(e)}[/red]")

    def get_current_price(self):
        """Get current price of the symbol"""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            self.console.print("[red]Failed to get current price[/red]")
            return None
        
        return tick.ask if self.active_trades[0].type == 0 else tick.bid