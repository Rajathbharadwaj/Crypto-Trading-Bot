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

class XAUBot:
    def __init__(self, timeframe=mt5.TIMEFRAME_M15, name="XAUBot"):
        self.name = name
        self.timeframe = timeframe
        self.running = True
        self.console = Console()
        self.symbol = "XAUUSD"
        
        # Trading configuration
        self.lot_size = 0.1    # Will be calculated based on risk
        self.max_sl_percent = 0.50  # Maximum 0.50% stop loss
        self.risk_per_trade = 0.01  # 1% risk per trade
        self.trade_taken = False
        
        # Strategy Parameters
        self.ema_short = 13     # Short EMA period
        self.ema_long = 20      # Long EMA period
        self.atr_period = 14    # ATR period
        self.volatility_threshold = 0.5  # Threshold for ATR multiplier
        self.trailing_stop_percent = 0.5  # Trailing stop percentage
        self.trailing_tp_percent = 2.0   # Trailing take profit percentage
        
        # Trading hours (IST)
        self.trading_start = "11:00:00"
        self.trading_end = "02:30:00"
        
        if not self.initialize_mt5():
            raise Exception("MT5 initialization failed!")

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
        """Calculate all required indicators"""
        try:
            # Calculate EMAs
            df['ema_short'] = ta.ema(df['close'], length=self.ema_short)
            df['ema_long'] = ta.ema(df['close'], length=self.ema_long)
            
            # Calculate ATR
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
            
            # Calculate Volatility (14-period standard deviation / close)
            df['volatility'] = df['close'].rolling(window=14).std() / df['close']
            
            # Calculate ATR Multiplier based on Volatility
            df['atr_multiplier'] = df['volatility'].apply(lambda x: 2 if x < self.volatility_threshold else 3)
            
            # Debug info
            last_row = df.iloc[-1]
            self.console.print("\n=== Latest Indicator Values ===")
            self.console.print(f"EMA13: {last_row['ema_short']:.2f}")
            self.console.print(f"EMA50: {last_row['ema_long']:.2f}")
            self.console.print(f"ATR: {last_row['atr']:.2f}")
            self.console.print(f"Volatility: {last_row['volatility']:.3f}")
            self.console.print(f"ATR Multiplier: {last_row['atr_multiplier']}")
            
            return df
            
        except Exception as e:
            self.console.print(f"[bold red]Error calculating indicators: {str(e)}[/bold red]")
            return None

    def get_trading_signal(self, df):
        """Generate trading signals based on EMA crossover"""
        if df.empty or len(df) < 2:
            self.console.print("[red]Not enough data for signal generation[/red]")
            return None, None
            
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # Check if trade is active
        if self.get_active_trades():
            self.console.print("[yellow]Trade already active, skipping signal check[/yellow]")
            return None, None

        # Debug information
        self.console.print("\n=== Signal Conditions ===")
        self.console.print(f"Current Price: {last_row['close']:.2f}")
        self.console.print(f"Previous EMA13: {prev_row['ema_short']:.2f}")
        self.console.print(f"Previous EMA20: {prev_row['ema_long']:.2f}")
        self.console.print(f"Current EMA13: {last_row['ema_short']:.2f}")
        self.console.print(f"Current EMA20: {last_row['ema_long']:.2f}")
        
        # BUY Conditions (EMA13 crosses above EMA50)
        buy_conditions = (
            prev_row['ema_short'] <= prev_row['ema_long'] and  # Previous candle EMA cross
            last_row['ema_short'] > last_row['ema_long']       # Current candle EMA cross
        )
        
        # SELL Conditions (EMA13 crosses below EMA50)
        sell_conditions = (
            prev_row['ema_short'] >= prev_row['ema_long'] and  # Previous candle EMA cross
            last_row['ema_short'] < last_row['ema_long']       # Current candle EMA cross
        )
        
        if buy_conditions:
            return "BUY", "EMA13 crossed above EMA50"
        elif sell_conditions:
            return "SELL", "EMA13 crossed below EMA50"
        
        return None, None

    def calculate_sl_tp(self, signal, df):
        """Calculate Stop Loss and Take Profit based on ATR"""
        current_price = df.iloc[-1]['close']
        prev_candle = df.iloc[-2]
        atr = df.iloc[-1]['atr']
        atr_multiplier = df.iloc[-1]['atr_multiplier']
        
        # Calculate ATR-based stop loss
        atr_sl = atr * atr_multiplier
        
        if signal == "BUY":
            # Stop loss at previous candle low or ATR-based, whichever is closer
            sl_price = current_price - atr_sl
            
            # Calculate take profit (2x the stop loss distance)
            tp_distance = atr_sl * 2
            tp_price = current_price + tp_distance
            
        else:  # SELL
            # Stop loss at previous candle high or ATR-based, whichever is closer
            sl_price = current_price + atr_sl
            
            # Calculate take profit (2x the stop loss distance)
            tp_distance = atr_sl * 2
            tp_price = current_price - tp_distance
        
        self.console.print("\n=== Trade Levels ===")
        self.console.print(f"Entry: ${current_price:.2f}")
        self.console.print(f"SL: ${sl_price:.2f} ({(abs(sl_price - current_price)/current_price)*100:.2f}%)")
        self.console.print(f"TP: ${tp_price:.2f} ({(abs(tp_price - current_price)/current_price)*100:.2f}%)")
        
        return sl_price, tp_price

    def execute_trade(self, signal, df):
        """Execute trade with debug information"""
        if self.trade_taken:
            self.console.print("[yellow]Trade already active, skipping new signal[/yellow]")
            return False
            
        try:
            current_price = df.iloc[-1]['close']
            sl_price, tp_price = self.calculate_sl_tp(signal, df)
            
            # Get current market price
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                self.console.print("[red]Failed to get current price[/red]")
                return False
            
            # Use appropriate price
            price = tick.ask if signal == "BUY" else tick.bid
            
            # Calculate position size based on risk
            account_info = mt5.account_info()
            if account_info is None:
                self.console.print("[red]Failed to get account info[/red]")
                return False
            
            risk_amount = account_info.balance * self.risk_per_trade
            sl_points = abs(price - sl_price)
            
            if sl_points == 0:
                self.console.print("[red]Invalid stop loss points[/red]")
                return False
            
            self.lot_size = round(risk_amount / sl_points, 2)
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Python BTC {signal}",
                "type_filling": mt5.ORDER_FILLING_IOC,
                "type_time": mt5.ORDER_TIME_GTC
            }
            
            # Debug info
            self.console.print("\nOrder Details:")
            for key, value in request.items():
                self.console.print(f"{key}: {value}")
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.console.print(f"[red]Order failed! Code: {result.retcode}, Comment: {result.comment}[/red]")
                return False
            
            # Log the trade after successful execution
            trade_details = {
                'type': signal,
                'entry_price': price,
                'sl': sl_price,
                'tp': tp_price,
                'volume': self.lot_size,
            }
            self.log_trade(trade_details)
            
            self.trade_taken = True
            return True
            
        except Exception as e:
            self.console.print(f"[bold red]Trade execution error: {str(e)}[/bold red]")
            return False

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
        
        # Check volatility switch
        if last_row['volatility'] <= 0.3:  # Exit if volatility becomes too low
            should_exit = True
            exit_reason = f"Low volatility at {last_row['volatility']:.3f}"
        
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
        config_table.add_row("ATR Period", str(self.atr_period))
        config_table.add_row("Volatility Threshold", f"{self.volatility_threshold:.2f}")
        config_table.add_row("Trailing Stop %", f"{self.trailing_stop_percent}%")
        config_table.add_row("Trailing TP %", f"{self.trailing_tp_percent}%")
        
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
                
                # Check trading hours
                # if not self.is_trading_hours():
                #     self.console.print("[yellow]Outside trading hours. Waiting...[/yellow]")
                #     time.sleep(60)
                #     continue
                
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
                
            close_price = tick.bid if position['type'] == "BUY" else tick.ask
            profit_pips = close_price - position['open_price']  # For BTC, 1 pip = $1
            if position['type'] == "SELL":
                profit_pips = -profit_pips
                
            close_details = {
                'ticket': position['ticket'],
                'type': position['type'],
                'entry_price': position['open_price'],
                'exit_price': close_price,
                'profit': position['profit'],
                'profit_pips': profit_pips,
                'exit_reason': reason,
                'trade_duration': str(datetime.now() - pd.to_datetime(position['time']) if 'time' in position else 'N/A')
            }
            
            # Close the position
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": position['volume'],
                "type": mt5.ORDER_TYPE_SELL if position['type'] == "BUY" else mt5.ORDER_TYPE_BUY,
                "position": position['ticket'],
                "price": close_price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Python close: {reason}",
                "type_filling": mt5.ORDER_FILLING_FOK,
                "type_time": mt5.ORDER_TIME_GTC
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.console.print("[green]Trade closed successfully![/green]")
                self.console.print(f"Profit: ${position['profit']:.2f} ({profit_pips:.1f} pips)")
                self.trade_taken = False
                return True
            else:
                self.console.print(f"[red]Failed to close trade: {result.comment}[/red]")
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
        """Display current market status and conditions"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.console.print("[red]Failed to get symbol info[/red]")
                return
            
            # Create market status table
            market_table = Table(title="Market Status")
            market_table.add_column("Metric", style="cyan")
            market_table.add_column("Value", justify="right")
            market_table.add_column("Status", justify="right")
            
            # Add current price info
            market_table.add_row(
                "Bid",
                f"${symbol_info.bid:.2f}",
                ""
            )
            market_table.add_row(
                "Ask",
                f"${symbol_info.ask:.2f}",
                ""
            )
            market_table.add_row(
                "Spread",
                f"{(symbol_info.ask - symbol_info.bid):.2f}",
                "[yellow]High[/yellow]" if (symbol_info.ask - symbol_info.bid) > 5 else "[green]Normal[/green]"
            )
            
            # Get active trades
            active_trades = self.get_active_trades()
            if active_trades:
                market_table.add_row(
                    "Trade Status",
                    "ACTIVE",
                    "[green]Position Open[/green]"
                )
            else:
                market_table.add_row(
                    "Trade Status",
                    "WAITING",
                    "[yellow]Monitoring Market[/yellow]"
                )
            
            self.console.print(market_table)
            
            # If there's an active trade, show its details
            if active_trades:
                trade_table = Table(title="Active Trade")
                trade_table.add_column("Metric")
                trade_table.add_column("Value")
                
                trade = active_trades[0]  # Get first active trade
                
                trade_table.add_row(
                    "Type",
                    f"[{'green' if trade['type'] == 'BUY' else 'red'}]{trade['type']}[/]"
                )
                trade_table.add_row(
                    "Entry Price",
                    f"${trade['open_price']:.2f}"
                )
                trade_table.add_row(
                    "Current Price",
                    f"${trade['current_price']:.2f}"
                )
                trade_table.add_row(
                    "Stop Loss",
                    f"${trade['sl']:.2f}"
                )
                trade_table.add_row(
                    "Take Profit",
                    f"${trade['tp']:.2f}"
                )
                trade_table.add_row(
                    "Profit/Loss",
                    f"[{'green' if trade['profit'] > 0 else 'red'}]${trade['profit']:.2f} ({trade['profit_pips']:.1f} pips)[/]"
                )
                
                self.console.print(trade_table)
            
            # Display recent candles
            df = self.get_rates_df(5)  # Get last 5 candles
            if df is not None:
                candle_table = Table(title="Recent Candles")
                candle_table.add_column("Time")
                candle_table.add_column("Open")
                candle_table.add_column("High")
                candle_table.add_column("Low")
                candle_table.add_column("Close")
                candle_table.add_column("Volume")
                
                for _, row in df.iterrows():
                    candle_table.add_row(
                        row['time'].strftime("%H:%M:%S"),
                        f"${row['open']:.2f}",
                        f"${row['high']:.2f}",
                        f"${row['low']:.2f}",
                        f"${row['close']:.2f}",
                        str(int(row['tick_volume']))
                    )
                
                self.console.print(candle_table)
                
        except Exception as e:
            self.console.print(f"[red]Error displaying market status: {str(e)}[/red]")

    def get_active_trades(self):
        """Get all active trades for the symbol"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                return []
            
            active_trades = []
            for position in positions:
                # Calculate profit in pips
                profit_pips = position.profit / self.lot_size  # For BTC, 1 pip = $1
                
                trade_info = {
                    'ticket': position.ticket,
                    'type': "BUY" if position.type == 0 else "SELL",
                    'volume': position.volume,
                    'open_price': position.price_open,
                    'current_price': position.price_current,
                    'sl': position.sl,
                    'tp': position.tp,
                    'profit': position.profit,
                    'profit_pips': profit_pips,
                    'comment': position.comment,
                    'magic': position.magic
                }
                active_trades.append(trade_info)
            
            return active_trades
            
        except Exception as e:
            self.console.print(f"[red]Error getting active trades: {str(e)}[/red]")
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
            log_file = f"trades_{datetime.now().strftime('%Y%m')}_gold.csv"
            trade_data = {
                'timestamp': datetime.now(),
                'type': trade_details['type'],
                'entry': trade_details['entry_price'],
                'sl': trade_details['sl'],
                'tp': trade_details['tp'],
                'volume': trade_details['volume'],
                'profit': trade_details.get('profit', 0),
                'exit_reason': trade_details.get('exit_reason', '')
            }
            
            pd.DataFrame([trade_data]).to_csv(log_file, mode='a', header=not os.path.exists(log_file))
            self.console.print("[green]Trade logged successfully[/green]")
            
        except Exception as e:
            self.console.print(f"[red]Error logging trade: {str(e)}[/red]")

    def display_performance(self):
        """Display bot performance metrics"""
        try:
            log_file = f"trades_{datetime.now().strftime('%Y%m')}_gold.csv"
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
        """Update trailing stop based on new strategy parameters"""
        try:
            # Get current price
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                return False
            
            current_price = tick.bid if trade['type'] == "BUY" else tick.ask
            entry_price = trade['open_price']
            current_sl = trade['sl']
            
            # Calculate profit percentage
            profit_percent = ((current_price - entry_price) / entry_price) if trade['type'] == "BUY" \
                            else ((entry_price - current_price) / entry_price)
            
            # Debug info
            self.console.print("\n=== Trailing Stop Check ===")
            self.console.print(f"Current Profit: {profit_percent*100:.2f}%")
            self.console.print(f"Current SL: ${current_sl:.2f}")
            
            # Calculate new trailing stop levels
            if trade['type'] == "BUY":
                new_sl = current_price * (1 - self.trailing_stop_percent / 100)
                if new_sl > current_sl:
                    self.modify_sl(trade['ticket'], new_sl)
                    self.console.print(f"[green]Trailing stop updated: ${new_sl:.2f}[/green]")
            else:  # SELL
                new_sl = current_price * (1 + self.trailing_stop_percent / 100)
                if new_sl < current_sl:
                    self.modify_sl(trade['ticket'], new_sl)
                    self.console.print(f"[green]Trailing stop updated: ${new_sl:.2f}[/green]")
                    
            return True
            
        except Exception as e:
            self.console.print(f"[red]Error updating trailing stop: {str(e)}[/red]")
            return False

    def modify_sl(self, ticket, new_sl):
        """Modify stop loss for an existing trade"""
        try:
            # Prepare the request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": self.symbol,
                "sl": new_sl,
                "position": ticket
            }
            
            # Send order to MT5
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.console.print(f"[red]Failed to modify SL: {result.comment}[/red]")
                return False
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Error modifying SL: {str(e)}[/red]")
            return False