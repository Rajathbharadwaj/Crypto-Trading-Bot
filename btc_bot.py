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
        # Trading configuration
        self.lot_size = 1.0     # Fixed lot size
        self.min_cross_distance = 0  # No minimum distance required
        self.confirmation_period = 1  # Place trade at next candle open
        
        # Take Profit Levels
        self.tp1_percent = 0.0025  # 0.25% for 0.40 lots
        self.tp2_percent = 0.0050  # 0.50% for 0.20 lots
        self.tp3_percent = 0.0075  # 0.75% for 0.20 lots
        # Last 0.20 lot uses trailing stop
        
        # Stop Loss and Trailing configuration
        self.initial_sl_percent = 0.0025  # 0.25% initial stop loss
        self.trailing_enabled = True
        self.trailing_activation_percent = 0.0075  # Activate at 0.75% profit
        self.trailing_step_percent = 0.0025       # 0.25% step
        self.trailing_trigger_step = 0.0025       # Move SL every 0.25% profit
        
        # Strategy Parameters
        self.ema_short = 9      # Short EMA period
        self.ema_long = 20      # Long EMA period
        self.lookback_candles = 4  # Number of candles to look back for SL
        
        # Trading hours (Eastern Time)
        self.trading_start = "18:00:00"  # Sunday 6 PM ET
        self.trading_end = "17:00:00"    # Friday 5 PM ET
        self.trading_days = [0, 1, 2, 3, 4]  # Monday to Friday (0 = Monday)
        
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
            # If crossover was just detected
            if not self.crossover_detected:
                self.crossover_detected = True
                self.crossover_type = "BUY" if buy_cross else "SELL"
                self.crossover_price = current_price
                self.crossover_price_index = len(df) - 1
                self.save_state()
                self.console.print(f"[green]✓ {self.crossover_type} Crossover Detected! Waiting for next candle to enter.[/green]")
                return None, "Crossover detected - awaiting next candle"
            
            # If crossover was detected in previous candle, place the trade
            elif self.crossover_detected and len(df) > self.crossover_price_index + 1:
                signal = self.crossover_type
                self._reset_crossover_state()  # Reset for next signal
                self.console.print(f"[green]✓ Placing {signal} trade at candle open.[/green]")
                return signal, "Placing trade at new candle open"
            
            return None, "Waiting for next candle"
        
        return None, "No signal"

    def calculate_sl_tp(self, signal, df):
        """Calculate Stop Loss and Take Profit levels"""
        current_price = df.iloc[-1]['close']
        
        # Calculate stop loss
        if signal == "BUY":
            # Stop loss at 0.25% below entry or previous candle low, whichever is bigger
            sl_by_percent = current_price * (1 - self.initial_sl_percent)
            sl_by_candle = df.iloc[-2]['low']  # Previous candle's low
            sl_price = min(sl_by_percent, sl_by_candle)  # Take the lower value for buy SL
            
            # Calculate take profit levels
            tp1_price = current_price * (1 + self.tp1_percent)  # 0.25% for 0.40 lots
            tp2_price = current_price * (1 + self.tp2_percent)  # 0.50% for 0.20 lots
            tp3_price = current_price * (1 + self.tp3_percent)  # 0.75% for 0.20 lots
            
        else:  # SELL
            # Stop loss at 0.25% above entry or previous candle high, whichever is bigger
            sl_by_percent = current_price * (1 + self.initial_sl_percent)
            sl_by_candle = df.iloc[-2]['high']  # Previous candle's high
            sl_price = max(sl_by_percent, sl_by_candle)  # Take the higher value for sell SL
            
            # Calculate take profit levels
            tp1_price = current_price * (1 - self.tp1_percent)  # 0.25% for 0.40 lots
            tp2_price = current_price * (1 - self.tp2_percent)  # 0.50% for 0.20 lots
            tp3_price = current_price * (1 - self.tp3_percent)  # 0.75% for 0.20 lots
        
        return sl_price, tp1_price, tp2_price, tp3_price

    def execute_trade(self, signal, df):
        """Execute a trade based on the signal"""
        try:
            current_price = df.iloc[-1]['close']
            sl_price, tp1_price, tp2_price, tp3_price = self.calculate_sl_tp(signal, df)
            
            # First position: 0.40 lots with TP at 0.25%
            request1 = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": 0.40,
                "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": current_price,
                "sl": sl_price,
                "tp": tp1_price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Python {signal} Part 1",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Second position: 0.20 lots with TP at 0.50%
            request2 = {**request1, 
                       "volume": 0.20, 
                       "tp": tp2_price,
                       "comment": f"Python {signal} Part 2"}
            
            # Third position: 0.20 lots with TP at 0.75%
            request3 = {**request1, 
                       "volume": 0.20, 
                       "tp": tp3_price,
                       "comment": f"Python {signal} Part 3"}
            
            # Fourth position: 0.20 lots with trailing stop
            request4 = {**request1, 
                       "volume": 0.20, 
                       "tp": 0.0,  # No TP, will use trailing stop
                       "comment": f"Python {signal} Part 4 (Trailing)"}
            
            # Send all orders
            results = []
            for request in [request1, request2, request3, request4]:
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    self.console.print(f"[red]Order failed, retcode={result.retcode}[/red]")
                    return False
                results.append(result)
            
            # Store trade details for each position
            for i, result in enumerate(results):
                trade_details = {
                    'ticket': result.order,
                    'type': signal,
                    'entry_price': current_price,
                    'sl': sl_price,
                    'volume': request1["volume"] if i == 0 else request2["volume"],
                    'tp': [tp1_price, tp2_price, tp3_price, 0.0][i],
                    'part': i + 1
                }
                self.active_trades.append(trade_details)
            
            self.console.print(f"[green]✓ {signal} orders placed successfully[/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]Error executing trade: {str(e)}[/red]")
            return False

    def check_exit_conditions(self, df):
        """Check if any exit conditions are met and manage partial exits"""
        if not self.active_trades:
            return False, None
            
        trade = self.active_trades[0]
        current_price = mt5.symbol_info_tick(self.symbol).last
        
        # Calculate profit percentage
        if trade['type'] == "BUY":
            profit_percent = (current_price - trade['entry_price']) / trade['entry_price']
        else:  # SELL
            profit_percent = (trade['entry_price'] - current_price) / trade['entry_price']
        
        # Check for partial exits
        if not trade['exits_triggered'][0] and profit_percent >= 0.0025:  # 0.25% profit
            self.close_partial_position(trade, 0.4, "TP1 at 0.25% profit")
            trade['exits_triggered'][0] = True
            
        elif not trade['exits_triggered'][1] and profit_percent >= 0.0050:  # 0.50% profit
            self.close_partial_position(trade, 0.2, "TP2 at 0.50% profit")
            trade['exits_triggered'][1] = True
            
        elif not trade['exits_triggered'][2] and profit_percent >= 0.0075:  # 0.75% profit
            self.close_partial_position(trade, 0.2, "TP3 at 0.75% profit")
            trade['exits_triggered'][2] = True
            # After this, only 0.20 lots remain for trailing stop
            
            # Move stop loss to entry price when 0.75% profit is reached
            if trade['sl'] != trade['entry_price']:
                self.modify_position_sl(trade['ticket'], trade['entry_price'])
                trade['sl'] = trade['entry_price']
        
        # Manage trailing stop for remaining position after 0.75% profit
        if all(trade['exits_triggered']) and profit_percent >= 0.0075:
            steps_above_activation = (profit_percent - 0.0075) / 0.0025  # Every 0.25% move
            if steps_above_activation > 0:
                sl_steps = int(steps_above_activation)
                if trade['type'] == "BUY":
                    new_sl = trade['entry_price'] * (1 + (sl_steps * 0.0025))
                    if new_sl > trade['sl']:
                        self.modify_position_sl(trade['ticket'], new_sl)
                        trade['sl'] = new_sl
                else:  # SELL
                    new_sl = trade['entry_price'] * (1 - (sl_steps * 0.0025))
                    if new_sl < trade['sl']:
                        self.modify_position_sl(trade['ticket'], new_sl)
                        trade['sl'] = new_sl
        
        return False, None  # Don't use EMA cross for exit

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
                else:
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
        """Retrieve active trades from MT5 and manage trailing stops"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                self.active_trades = []
                return []
            
            # Update trailing stops for positions
            for position in positions:
                if "(Trailing)" in position.comment:
                    self.manage_trailing_stop(position)
            
            self.active_trades = list(positions)
            return self.active_trades
            
        except Exception as e:
            self.console.print(f"[red]Error getting active trades: {str(e)}[/red]")
            self.active_trades = []
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
        """Check if current time is within trading hours (Eastern Time)"""
        et = pytz.timezone('America/New_York')
        current_time = datetime.now(et)
        current_weekday = current_time.weekday()  # 0 = Monday, 6 = Sunday
        
        # Format just the time part for comparison
        current_time_str = current_time.strftime('%H:%M:%S')
        
        # Special handling for Sunday (weekday 6) and Friday (weekday 4)
        if current_weekday == 6:  # Sunday
            # Only trade after 6 PM
            return current_time_str >= self.trading_start
        elif current_weekday == 4:  # Friday
            # Only trade until 5 PM
            return current_time_str <= self.trading_end
        elif current_weekday == 5:  # Saturday
            # No trading on Saturday
            return False
        else:  # Monday to Thursday
            # Trade all day
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