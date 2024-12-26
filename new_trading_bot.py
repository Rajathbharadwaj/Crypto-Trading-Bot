import MetaTrader5 as mt5
import pandas as pd
import time
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta, timezone
from rich.console import Console
from rich.table import Table
import pytz

class GoldTradingBot:
    def __init__(self, timeframe=mt5.TIMEFRAME_M5, name="GoldBot"):
        self.name = name
        self.timeframe = timeframe
        self.running = True
        self.console = Console()
        self.symbol = "XAUUSD"
        self.lot_size = 0.7
        self.max_sl_pips = 30
        self.volatility_threshold = 0.5
        self.trade_taken = False
        self.count = 2000
        
        # Trading hours (Toronto Time)
        self.trading_start = "01:30:00"  # 1:30 AM Toronto (= 11:00 AM IST)
        self.trading_end = "17:00:00"    # 5:00 PM Toronto (= 2:30 AM IST next day)
        
        # Check if MT5 is already initialized
        if not self.initialize_mt5():
            self.console.print("[bold red]MT5 initialization failed![/bold red]")
            return
            
    def initialize_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize(path="C:\\Program Files\\MetaTrader 5\\terminal64.exe",  # Add this line with your MT5 path
        login=190524521,        # Verify this is your actual account number
        password="QAZwsx456!",  # Verify this is your actual password
        server="Exness-MT5Trial14"):
            self.console.print("[bold red]MT5 initialization failed![/bold red]")
            return False
            
        # Login to MT5 (use your credentials)
        if not mt5.login(login=190524521, password="QAZwsx456!", server="Exness-MT5Trial14"):
            self.console.print("[bold red]MT5 login failed![/bold red]")
            mt5.shutdown()
            return False
            
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

    def is_trading_time(self):
        """Check if current time is within trading hours (Toronto Time)"""
        toronto_tz = pytz.timezone('America/Toronto')
        current_time = datetime.now(toronto_tz)
        
        # Convert trading hours to datetime objects
        start_time = datetime.strptime(self.trading_start, "%H:%M:%S").time()  # 1:30 AM
        end_time = datetime.strptime(self.trading_end, "%H:%M:%S").time()      # 5:00 PM
        
        current_time = current_time.time()
        
        # Debug info
        self.console.print(f"\n=== Trading Hours (Toronto Time) ===")
        self.console.print(f"Current time: {current_time.strftime('%I:%M:%S %p')}")
        self.console.print(f"Trading window: {start_time.strftime('%I:%M:%S %p')} - {end_time.strftime('%I:%M:%S %p')}")
        
        # Check if current time is within trading hours
        is_trading_allowed = start_time <= current_time <= end_time
        
        if is_trading_allowed:
            self.console.print("[green]Trading allowed: Within session hours[/green]")
        else:
            self.console.print("[red]Trading not allowed: Outside trading hours[/red]")
        
        return is_trading_allowed

    def calculate_supertrend(self, df, period=10, multiplier=3):
        """Calculate Supertrend with direction verification"""
        st = df.ta.supertrend(
            high=df['high'], 
            low=df['low'],
            close=df['close'],
            length=period,
            multiplier=multiplier
        )
        
        df['supertrend'] = st[f'SUPERT_{period}_{multiplier}.0']
        df['supertrend_direction'] = st[f'SUPERTd_{period}_{multiplier}.0']
        
        # Verify last few values
        last_values = df.tail(3)
        self.console.print("\n=== Last 3 Supertrend Values ===")
        for idx, row in last_values.iterrows():
            self.console.print(f"Time: {row['time']}")
            self.console.print(f"Direction: {'Bullish' if row['supertrend_direction'] == 1 else 'Bearish'}")
        
        return df

    def calculate_volatility(self, df):
        """Calculate volatility with minimal output"""
        df['roc'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * 100
        df['sma2'] = df['close'].rolling(window=2).mean()
        df['dr'] = df['roc'] / (df['sma2'] / df['close'].mean())
        df['vola21'] = df['dr'].rolling(window=21).std()
        
        df['volatility_switch'] = 0.0
        for i in range(len(df)):
            if i >= 21:
                count = 1
                current_vola = df['vola21'].iloc[i]
                for j in range(1, 21):
                    if df['vola21'].iloc[i-j] <= current_vola:
                        count += 1
                df.loc[df.index[i], 'volatility_switch'] = count / 21
        
        return df
    
    def get_trading_signal(self, df):
        """Generate trading signals with new candle confirmation"""
        if df.empty or len(df) < 3:  # Need at least 3 candles now
            return None, None
            
        last_row = df.iloc[-1]      # Current candle
        prev_row = df.iloc[-2]      # Previous candle
        third_row = df.iloc[-3]     # Candle before previous
        
        # Check if trade is active
        active_trades = self.get_active_trades()
        if active_trades:
            trade = active_trades[0]
            self.console.print("\n[bold green]=== TRADE ACTIVE ===[/bold green]")
            self.console.print(f"Type: {trade['type']}")
            self.console.print(f"Entry: ${trade['open_price']:.2f}")
            self.console.print(f"Current: ${trade['current_price']:.2f}")
            self.console.print(f"Profit: ${trade['profit']:.2f}")
            return None, None

        # Trading Conditions Check
        self.console.print("\n=== Signal Conditions ===")
        self.console.print(f"Previous Supertrend: {prev_row['supertrend_direction']}")
        self.console.print(f"Current Supertrend: {last_row['supertrend_direction']}")
        self.console.print(f"Volatility Switch: {last_row['volatility_switch']:.3f}")
        
        # Check volatility first
        if last_row['volatility_switch'] <= 0.5:
            self.console.print("[red]❌ Volatility too low (needs > 0.5)[/red]")
            return None, None
        else:
            self.console.print("[green]✓ Volatility condition met[/green]")
        
        # Check for Supertrend change on previous candle and take trade on new candle
        if (prev_row['supertrend_direction'] == 1 and 
            third_row['supertrend_direction'] == -1 and
            last_row['supertrend_direction'] == 1):
            self.console.print("[bold green]✓ BUY SIGNAL: Supertrend turned bullish and new candle confirmed[/bold green]")
            return "BUY", f"Volatility ({last_row['volatility_switch']:.3f})"
            
        elif (prev_row['supertrend_direction'] == -1 and 
              third_row['supertrend_direction'] == 1 and
              last_row['supertrend_direction'] == -1):
            self.console.print("[bold red]✓ SELL SIGNAL: Supertrend turned bearish and new candle confirmed[/bold red]")
            return "SELL", f"Volatility ({last_row['volatility_switch']:.3f})"
        
        # If no transition, show current direction
        if last_row['supertrend_direction'] == 1:
            self.console.print("⚠️ Supertrend is bullish but waiting for new candle confirmation")
        else:
            self.console.print("⚠️ Supertrend is bearish but waiting for new candle confirmation")
        
        return None, None

    
    def calculate_sl_tp(self, signal, df):
        """Calculate Stop Loss and Take Profit with proper pip values for BTCUSD"""
        current_price = df.iloc[-1]['close']
        prev_candle = df.iloc[-2]
        
        # For BTCUSD:
        # 1 pip = $0.1
        # 30 pips = $3.00
        pip_value = 0.1
        
        self.console.print("\n=== Trade Levels ===")
        self.console.print(f"Current Price: ${current_price:.2f}")
        
        # Calculate stop loss based on previous candle low/high
        if signal == "BUY":
            sl_price = prev_candle['low']  # Use candle low for buy trades
            sl_pips = (current_price - sl_price) / pip_value
            
            # Calculate take profit based on risk:reward
            if sl_pips <= 25:  # If SL is <= 25 pips ($2.50)
                tp_pips = 30   # TP = 30 pips ($3.00)
            else:
                tp_pips = 10   # TP = 10 pips ($1.00)
                
            tp_price = current_price + (tp_pips * pip_value)
            
            self.console.print("\n[green]BUY Trade Levels:[/green]")
            self.console.print(f"Entry: ${current_price:.2f}")
            self.console.print(f"SL: ${sl_price:.2f} (Previous candle low)")
            self.console.print(f"TP: ${tp_price:.2f} ({tp_pips} pips = ${tp_pips/10:.2f})")
            
        else:  # SELL
            sl_price = prev_candle['high']  # Use candle high for sell trades
            sl_pips = (sl_price - current_price) / pip_value
            
            # Calculate take profit based on risk:reward
            if sl_pips <= 25:  # If SL is <= 25 pips ($2.50)
                tp_pips = 30   # TP = 30 pips ($3.00)
            else:
                tp_pips = 10   # TP = 10 pips ($1.00)
                
            tp_price = current_price - (tp_pips * pip_value)
            
            self.console.print("\n[red]SELL Trade Levels:[/red]")
            self.console.print(f"Entry: ${current_price:.2f}")
            self.console.print(f"SL: ${sl_price:.2f} (Previous candle high)")
            self.console.print(f"TP: ${tp_price:.2f} ({tp_pips} pips = ${tp_pips/10:.2f})")
        
        # Verify the levels
        self.console.print("\n=== Level Verification ===")
        if signal == "BUY":
            self.console.print(f"Risk: {sl_pips:.1f} pips (${sl_pips/10:.2f})")
            self.console.print(f"Reward: {tp_pips:.1f} pips (${tp_pips/10:.2f})")
        else:
            self.console.print(f"Risk: {sl_pips:.1f} pips (${sl_pips/10:.2f})")
            self.console.print(f"Reward: {tp_pips:.1f} pips (${tp_pips/10:.2f})")
        
        return sl_price, tp_price
    
    def execute_trade(self, signal, df):
        """Execute trade with debug information"""
        if self.trade_taken:
            self.console.print("[yellow]Trade already active, skipping new signal[/yellow]")
            return False
            
        try:
            current_price = df.iloc[-1]['close']
            sl_price, tp_price = self.calculate_sl_tp(signal, df)
            _, trade_reason = self.get_trading_signal(df)

            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.console.print("[red]Failed to get symbol info[/red]")
                return False
                
            # Get allowed filling modes
            # filling_type = symbol_info.filling_mode
            # if filling_type == mt5.SYMBOL_FILLING_FOK:
            #     filling = mt5.ORDER_FILLING_FOK
            # elif filling_type == mt5.SYMBOL_FILLING_IOC:
            #     filling = mt5.ORDER_FILLING_IOC
            # else:
            #     filling = mt5.ORDER_FILLING_RETURN 
            
            # Debug information
            self.console.print("\nTrade Execution:")
            self.console.print(f"Signal: {signal}")
            self.console.print(f"Trade Reason: {trade_reason}")
            self.console.print(f"Current Price: {current_price}")
            self.console.print(f"Stop Loss: {sl_price}")
            self.console.print(f"Take Profit: {tp_price}")
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": current_price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Python {signal} order",
                "type_filling": mt5.ORDER_FILLING_FOK,  # Try FOK first
                "type_time": mt5.ORDER_TIME_GTC
            }
            
            # Show order details
            self.console.print("\nOrder Details:")
            for key, value in request.items():
                self.console.print(f"{key}: {value}")
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.console.print(f"[bold red]Order failed: {result.comment}[/bold red]")
                return False
                
            self.trade_taken = True
            self.console.print(f"[bold green]{signal} order placed successfully![/bold green]")
            return True
            
        except Exception as e:
            self.console.print(f"[bold red]Error executing trade: {str(e)}[/bold red]")
            return False
    
    def display_market_data(self, df):
        """Display market data with volume analysis"""
        market_table = Table(title="Market Data")
        market_table.add_column("Time")
        market_table.add_column("Price")
        market_table.add_column("Volume")
        market_table.add_column("Analysis")
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        # Calculate volume change
        volume_change = last_candle['tick_volume'] - prev_candle['tick_volume']
        volume_change_pct = (volume_change / prev_candle['tick_volume']) * 100
        
        # Volume analysis
        volume_status = (
            "[green]Increasing[/green]" if volume_change > 0
            else "[red]Decreasing[/red]"
        )
        
        market_table.add_row(
            last_candle['time'].strftime("%H:%M:%S"),
            f"{last_candle['close']:.2f}",
            f"{last_candle['tick_volume']} ({volume_change_pct:+.1f}%)",
            volume_status
        )
        
        self.console.print(market_table)
        
    def get_active_trades(self):
        """Get current active trades with essential info"""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None or len(positions) == 0:
            self.trade_taken = False
            return None
            
        active_trades = []
        for position in positions:
            profit_pips = (position.price_current - position.price_open) / 0.1  # For BTC, 0.1 = 1 pip
            if position.type == 1:  # If SELL, reverse the profit calculation
                profit_pips = -profit_pips
                
            trade_info = {
                'type': "BUY" if position.type == 0 else "SELL",
                'open_price': position.price_open,
                'current_price': position.price_current,
                'sl': position.sl,
                'tp': position.tp,
                'profit': position.profit,
                'profit_pips': profit_pips,
                'time': pd.to_datetime(position.time, unit='s')
            }
            active_trades.append(trade_info)
        
        return active_trades

    def display_bot_status(self):
        """Display comprehensive bot status"""
        self.console.print("\n=== GOLD TRADING BOT STATUS ===", style="bold green")
        
        # Configuration table
        config_table = Table(title="Configuration")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", justify="right")
        
        config_table.add_row("Symbol", self.symbol)
        config_table.add_row("Timeframe", "M5")
        config_table.add_row("Volatility Threshold", str(self.volatility_threshold))
        config_table.add_row("Max SL (pips)", str(self.max_sl_pips))
        config_table.add_row("Lot Size", str(self.lot_size))
        
        self.console.print(config_table)
        
        # Current market data
        market_table = Table(title="Current Market Data")
        market_table.add_column("Time")
        market_table.add_column("Open")
        market_table.add_column("High")
        market_table.add_column("Low")
        market_table.add_column("Close")
        market_table.add_column("Volume")
        
        # Get latest candle
        df = self.get_rates_df(1)
        if df is not None and not df.empty:
            last_candle = df.iloc[-1]
            market_table.add_row(
                last_candle['time'].strftime("%H:%M:%S"),
                f"{last_candle['open']:.2f}",
                f"{last_candle['high']:.2f}",
                f"{last_candle['low']:.2f}",
                f"{last_candle['close']:.2f}",
                str(int(last_candle['tick_volume']))
            )
        
        self.console.print(market_table)
        
        # Trade status
        trade_status = Table(title="Trade Status")
        trade_status.add_column("Status")
        trade_status.add_column("Details")
        
        active_trades = self.get_active_trades()
        if active_trades:
            trade = active_trades[0]  # Get first active trade
            trade_status.add_row(
                "[green]ACTIVE TRADE[/green]",
                f"Type: {trade['type']}"
            )
            trade_status.add_row(
                "Entry",
                f"{trade['open_price']:.2f}"
            )
            trade_status.add_row(
                "Current",
                f"{trade['current_price']:.2f}"
            )
            trade_status.add_row(
                "Profit",
                f"${trade['profit']:.2f} ({trade['profit_pips']:.1f} pips)"
            )
        else:
            trade_status.add_row(
                "[yellow]NO ACTIVE TRADE[/yellow]",
                "Monitoring market..."
            )
        
        self.console.print(trade_status)
        self.console.print("\nPress Ctrl+C to stop the bot", style="dim")

    def get_rates_df(self, num_candles=5000):
        """Get market data from MT5"""
        rates = mt5.copy_rates_from_pos(
            self.symbol,
            self.timeframe,
            0,
            num_candles
        )
        
        if rates is None:
            self.console.print("[bold red]Failed to get market data![/bold red]")
            return None
            
        # Convert to pandas DataFrame
        df = pd.DataFrame(rates)
        
        # Convert timestamp to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Add both UTC+3 and local time columns for reference
        df['mt5_time'] = df['time'].dt.strftime('%H:%M:%S')  # MT5 server time (UTC+3)
        df['local_time'] = (df['time'] - timedelta(hours=3)).dt.strftime('%H:%M:%S')  # Your local time
        
        # Rename columns for clarity
        df.columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'mt5_time', 'local_time']
        
        return df
    
    def display_market_status(self):
        """Display market status and current trade information"""
        # Get current tick data
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            self.console.print("[bold red]Failed to get current tick data![/bold red]")
            return
            
        # Create market status table
        market_table = Table(title="Market Status")
        market_table.add_column("Metric")
        market_table.add_column("Value")
        market_table.add_column("Status")
        
        # Add current price information
        market_table.add_row(
            "Current Bid",
            f"{tick.bid:.2f}",
            "[green]Active[/green]"
        )
        market_table.add_row(
            "Current Ask",
            f"{tick.ask:.2f}",
            f"Spread: {(tick.ask - tick.bid):.2f}"
        )
        
        # Add trade status
        if self.trade_taken:
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
        active_trades = self.get_active_trades()
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
                f"{trade['open_price']:.2f}"
            )
            trade_table.add_row(
                "Current Price",
                f"{trade['current_price']:.2f}"
            )
            trade_table.add_row(
                "Stop Loss",
                f"{trade['sl']:.2f}"
            )
            trade_table.add_row(
                "Take Profit",
                f"{trade['tp']:.2f}"
            )
            trade_table.add_row(
                "Profit/Loss",
                f"[{'green' if trade['profit'] > 0 else 'red'}]${trade['profit']:.2f} ({trade['profit_pips']:.1f} pips)[/]"
            )
            
            self.console.print(trade_table)
        
        # Display last few candles
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
                    f"{row['open']:.2f}",
                    f"{row['high']:.2f}",
                    f"{row['low']:.2f}",
                    f"{row['close']:.2f}",
                    str(int(row['tick_volume']))
                )
            
            self.console.print(candle_table)
    
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
        
        # 1. Check if SL or TP was hit (this is handled by MT5 automatically)
        
        # 2. Check Supertrend reversal
        if trade['type'] == "BUY" and last_row['supertrend_direction'] == -1:
            should_exit = True
            exit_reason = f"Supertrend turned bearish at {last_row['close']:.2f}"
        elif trade['type'] == "SELL" and last_row['supertrend_direction'] == 1:
            should_exit = True
            exit_reason = f"Supertrend turned bullish at {last_row['close']:.2f}"
        
        return should_exit, exit_reason

    def run(self):
        """Main bot loop with status display"""
        self.console.print("[bold green]Starting Gold Trading Bot...[/bold green]")
        
        while True:
            try:
                # Clear screen for clean display
                self.console.clear()
                
                # Get market data
                df = self.get_rates_df(5000)
                if df is None:
                    continue
                    
                # Calculate indicators
                df = self.calculate_supertrend(df)
                df = self.calculate_volatility(df)
                
                # Check exit conditions first
                should_exit, exit_reason = self.check_exit_conditions(df)
                
                if should_exit:
                    active_trades = self.get_active_trades()
                    if active_trades:
                        self.close_trade(active_trades[0], exit_reason)
                        self.console.print(f"[yellow]Exiting trade: {exit_reason}[/yellow]")
                
                # If no active trade, check for entry signals
                elif not self.trade_taken:
                    signal, reason = self.get_trading_signal(df)
                    if signal:
                        self.execute_trade(signal, df)
                
                time.sleep(1)  # Check every second
                
            except KeyboardInterrupt:
                self.console.print("[yellow]Bot shutdown requested...[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[bold red]Error: {str(e)}[/bold red]")
                time.sleep(5)

    def close_trade(self, position, reason="Manual"):
        """Close trade with reason logging"""
        try:
            close_price = position.price_current
            profit_pips = (close_price - position.price_open) * 10000
            if position.type == 1:  # SELL
                profit_pips = -profit_pips
                
            close_details = {
                'ticket': position.ticket,
                'type': "BUY" if position.type == 0 else "SELL",
                'entry_price': position.price_open,
                'exit_price': close_price,
                'profit': position.profit,
                'profit_pips': profit_pips,
                'exit_reason': reason,
                'trade_duration': str(datetime.now() - pd.to_datetime(position.time, unit='s'))
            }
            
            # Close the position
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": close_price,
                "deviation": 5,
                "magic": 234000,
                "comment": f"Python close: {reason}",
                "type_filling": mt5.ORDER_FILLING_IOC,
                "type_time": mt5.ORDER_TIME_GTC
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.log_trade('EXIT', 'SUCCESS', close_details)
                self.trade_taken = False
                return True
            else:
                close_details['error'] = result.comment
                self.log_trade('EXIT', 'FAILED', close_details)
                return False
                
        except Exception as e:
            self.log_trade('EXIT', 'ERROR', {
                'error': str(e),
                'position_ticket': position.ticket
            })
            return False

    def is_market_open(self):
        """Check if the market is currently open"""
        # Get current UTC time
        current_time = datetime.now(timezone.utc)
        
        # Convert to hours and minutes
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_weekday = current_time.weekday()  # Monday is 0, Sunday is 6
        
        # Check if it's during the daily break (22:00 - 23:00 UTC)
        if current_hour == 22:
            self.console.print("[yellow]Market is in daily break (22:00 - 23:00 UTC)[/yellow]")
            return False
        
        # Check if it's weekend
        if current_weekday == 5 and current_hour >= 22:  # Friday after 22:00
            self.console.print("[yellow]Market is closed for weekend[/yellow]")
            return False
        if current_weekday == 6:  # Saturday
            self.console.print("[yellow]Market is closed for weekend[/yellow]")
            return False
        if current_weekday == 0 and current_hour < 23:  # Sunday before 23:00
            self.console.print("[yellow]Market is closed for weekend[/yellow]")
            return False
        
        return True

    def check_trading_status(self):
        """Check overall trading status"""
        self.console.print("\n=== Trading Status Check ===")
        self.console.print(f"Symbol: {self.symbol}")
        self.console.print(f"Trade Taken: {self.trade_taken}")
        
        # Check MT5 connection
        if not mt5.initialize():
            self.console.print("[red]MT5 not connected![/red]")
            return
        
        # Check symbol info
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            self.console.print("[red]Symbol not found![/red]")
            return
        
        self.console.print(f"Current Bid: {symbol_info.bid}")
        self.console.print(f"Current Ask: {symbol_info.ask}")
        
        # Check active trades
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            self.console.print("[red]Failed to get positions![/red]")
        else:
            self.console.print(f"Active positions: {len(positions)}")