import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, time as datetime_time, timedelta
import pytz
import talib
from news_events import ForexNewsChecker
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
import time as time_module
import time

console = Console()

class GoldTradingBot:
    def __init__(self):
        self.console = Console()
        
        # Check if MT5 is initialized
        if not mt5.initialize(
            path="C:\\Program Files\\MetaTrader 5\\terminal64.exe",  
            login=190111246,        
            password="QAZwsx456!",  
            server="Exness-MT5Trial14"  
        ):
            self.console.print("[bold red]MT5 is not initialized![/bold red]")
            raise Exception("MT5 initialization failed")
            
        self.setup_initial_config()
        self.display_welcome_message()
        self.price_tracker = PriceTracker(timeframe_minutes=5)  # 5-minute candles
        self.console.print("[green]Price tracker initialized[/green]")

    def setup_initial_config(self):
        """Setup initial configuration"""
        self.symbol = "XAUUSD"
        self.timeframe = mt5.TIMEFRAME_M5
        self.volatility_threshold = 0.5
        self.max_sl_pips = 30
        self.st_period = 10
        self.st_multiplier = 3
        self.trading_start = datetime_time(11, 0)
        self.trading_end = datetime_time(2, 30)
        self.news_checker = ForexNewsChecker()
        self.lot_size = 0.01
        self.magic_number = 123456

    def display_welcome_message(self):
        """Display welcome message with bot configuration"""
        self.console.print("\n[bold cyan]🤖 Gold Trading Bot Initializing...[/bold cyan]")
        
        config_table = Table(show_header=True, header_style="bold magenta")
        config_table.add_column("Parameter", style="dim")
        config_table.add_column("Value", justify="right")
        
        config_table.add_row("Symbol", self.symbol)
        config_table.add_row("Timeframe", "M5")
        config_table.add_row("Volatility Threshold", str(self.volatility_threshold))
        config_table.add_row("Max SL (pips)", str(self.max_sl_pips))
        config_table.add_row("Supertrend Period", str(self.st_period))
        config_table.add_row("Supertrend Multiplier", str(self.st_multiplier))
        config_table.add_row("Trading Hours (IST)", f"{self.trading_start} - {self.trading_end}")
        config_table.add_row("Initial Lot Size", str(self.lot_size))
        
        self.console.print(Panel(config_table, title="[bold]Configuration", border_style="cyan"))

    def display_market_data(self, df):
        """Display market data in a formatted table"""
        if df is None or df.empty:
            self.console.print("[red]No market data available[/red]")
            return

        market_table = Table(show_header=True, header_style="bold green")
        market_table.add_column("Time", style="dim")
        market_table.add_column("Open", justify="right")
        market_table.add_column("High", justify="right")
        market_table.add_column("Low", justify="right")
        market_table.add_column("Close", justify="right")
        market_table.add_column("Volume", justify="right")

        for _, row in df.tail().iterrows():
            market_table.add_row(
                str(row['time']),
                f"{row['open']:.2f}",
                f"{row['high']:.2f}",
                f"{row['low']:.2f}",
                f"{row['close']:.2f}",
                str(int(row['tick_volume']))
            )

        self.console.print(Panel(market_table, title="[bold]Latest Market Data", border_style="green"))

    def display_indicators(self, df):
        """Display indicator values in a formatted table"""
        if df is None or df.empty:
            return

        indicator_table = Table(show_header=True, header_style="bold yellow")
        indicator_table.add_column("Indicator", style="dim")
        indicator_table.add_column("Value", justify="right")

        last_row = df.iloc[-1]
        
        indicator_table.add_row(
            "Supertrend Direction",
            "[green]Bullish[/green]" if last_row['supertrend_direction'] == 1 
            else "[red]Bearish[/red]"
        )
        indicator_table.add_row(
            "Volatility",
            f"{last_row['volatility_switch']:.4f}"
        )

        self.console.print(Panel(indicator_table, title="[bold]Indicator Values", border_style="yellow"))

    def get_rates_df(self, num_candles=1000):
        """Fetch rates and convert to DataFrame"""
        self.console.print("\n[yellow]Fetching market data...[/yellow]")
        
        try:
            # Get current time
            current_time = datetime.now()
            
            # Calculate start time (1000 candles * 5 minutes per candle)
            start_time = current_time - timedelta(minutes=5 * num_candles)
            
            # Fetch historical data
            rates = mt5.copy_rates_from_pos(
                self.symbol,
                self.timeframe,
                0,
                num_candles
            )
            
            if rates is None or len(rates) == 0:
                self.console.print("[red]Failed to fetch rates from MT5[/red]")
                return None
                
            # Convert to DataFrame
            print(f"Rates\n: {rates}")
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Display data info
            self.console.print(f"\n[green]Fetched {len(df)} candles of historical data[/green]")
            self.console.print(f"[green]Date range: {df['time'].min()} to {df['time'].max()}[/green]")
            
            # Display latest market data
            self.console.print("\n[cyan]Latest Market Data:[/cyan]")
            market_table = Table(show_header=True, header_style="bold green")
            market_table.add_column("Time")
            market_table.add_column("Open", justify="right")
            market_table.add_column("High", justify="right")
            market_table.add_column("Low", justify="right")
            market_table.add_column("Close", justify="right")
            market_table.add_column("Volume", justify="right")
            
            # Show last 5 candles
            for _, row in df.tail().iterrows():
                market_table.add_row(
                    str(row['time']),
                    f"{row['open']:.2f}",
                    f"{row['high']:.2f}",
                    f"{row['low']:.2f}",
                    f"{row['close']:.2f}",
                    f"{row['tick_volume']}"
                )
            
            self.console.print(market_table)
            
            # Basic data validation
            if df['close'].isnull().any():
                self.console.print("[red]Warning: Dataset contains missing values[/red]")
                
            return df
            
        except Exception as e:
            self.console.print(f"[bold red]Error fetching market data: {str(e)}[/bold red]")
            return None

    def calculate_supertrend(self, df, period=10, multiplier=3):
        """Calculate Supertrend using pandas_ta"""
        self.console.print("\n[yellow]Calculating Supertrend...[/yellow]")
        
        try:
            # Ensure we have enough data for calculation
            min_periods = max(period * 3, 50)  # At least 3x the period or 50 candles
            if len(df) < min_periods:
                self.console.print(f"[red]Warning: Not enough data for reliable Supertrend calculation. Have {len(df)} candles, need at least {min_periods}[/red]")
            
            # Calculate Supertrend
            st = df.ta.supertrend(
                high=df['high'], 
                low=df['low'],
                close=df['close'],
                length=period,
                multiplier=multiplier
            )
            print(f"Supertrend\n: {st}")
            
            # Extract supertrend values and direction using correct column names
            df['supertrend'] = st[f'SUPERT_{period}_{multiplier}.0']
            df['supertrend_direction'] = st[f'SUPERTd_{period}_{multiplier}.0']
            df['supertrend_long'] = st[f'SUPERTl_{period}_{multiplier}.0']
            df['supertrend_short'] = st[f'SUPERTs_{period}_{multiplier}.0']
            
            # Display calculation summary
            self.console.print(f"\n[green]Calculated Supertrend using {len(df)} candles[/green]")
            
            # Display last few values
            self.console.print("\n[green]Supertrend Calculation Results:[/green]")
            st_table = Table(show_header=True, header_style="bold green")
            st_table.add_column("Time")
            st_table.add_column("Close")
            st_table.add_column("Supertrend")
            st_table.add_column("Direction")
            st_table.add_column("Long Level")
            st_table.add_column("Short Level")
            
            for idx in range(-5, 0):  # Show last 5 rows
                row = df.iloc[idx]
                direction = "🟢 BUY" if row['supertrend_direction'] == 1 else "🔴 SELL"
                st_table.add_row(
                    str(row['time']),
                    f"{row['close']:.2f}",
                    f"{row['supertrend']:.2f}",
                    direction,
                    f"{row['supertrend_long']:.2f}",
                    f"{row['supertrend_short']:.2f}"
                )
            self.console.print(st_table)
            
            return df
            
        except Exception as e:
            self.console.print(f"[bold red]Error calculating Supertrend: {str(e)}[/bold red]")
            raise

    def calculate_volatility(self, df):
        """Calculate Volatility Switch using ATR"""
        self.console.print("\n[yellow]Calculating Volatility Switch...[/yellow]")
        
        try:
            # Ensure we have enough data for calculation
            min_periods = 50  # At least 50 candles for reliable ATR
            if len(df) < min_periods:
                self.console.print(f"[red]Warning: Not enough data for reliable volatility calculation. Have {len(df)} candles, need at least {min_periods}[/red]")
            
            # Calculate ATR
            df['atr'] = df.ta.atr(length=14)
            
            # Calculate Volatility Switch (ATR normalized by price)
            df['volatility_switch'] = df['atr'] / df['close']
            
            # Display calculation summary
            self.console.print(f"\n[green]Calculated Volatility using {len(df)} candles[/green]")
            
            # Display last few values
            self.console.print("\n[green]Volatility Calculation Results:[/green]")
            vol_table = Table(show_header=True, header_style="bold green")
            vol_table.add_column("Time")
            vol_table.add_column("ATR")
            vol_table.add_column("Volatility Switch")
            vol_table.add_column("Status")
            
            for idx in range(-5, 0):  # Show last 5 rows
                row = df.iloc[idx]
                status = "✅ ACTIVE" if row['volatility_switch'] >= self.volatility_threshold else "❌ INACTIVE"
                vol_table.add_row(
                    str(row['time']),
                    f"{row['atr']:.5f}",
                    f"{row['volatility_switch']:.5f}",
                    status
                )
            self.console.print(vol_table)
            
            return df
            
        except Exception as e:
            self.console.print(f"[bold red]Error calculating Volatility: {str(e)}[/bold red]")
            raise

    def get_trading_signal(self, df):
        """Generate trading signals based on Supertrend transitions"""
        if df.empty or len(df) < 2:
            return None, None
            
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]  # Previous candle
        
        # Debug information
        self.console.print(f"\nSignal Check:")
        self.console.print(f"Volatility Switch: {last_row['volatility_switch']:.3f} (Threshold: 0.5)")
        self.console.print(f"Current Supertrend: {last_row['supertrend_direction']}")
        self.console.print(f"Previous Supertrend: {prev_row['supertrend_direction']}")
        
        # Check if trade already active
        if self.trade_taken:
            self.console.print("[yellow]Trade already active, monitoring for exit[/yellow]")
            return None, None
        
        # First check: Volatility Switch must be ABOVE 0.5
        if last_row['volatility_switch'] <= 0.5:
            self.console.print("[yellow]Waiting for Volatility Switch > 0.5[/yellow]")
            return None, None
        
        # Check for Supertrend transitions
        if last_row['supertrend_direction'] == 1 and prev_row['supertrend_direction'] == -1:
            self.console.print("[green]BUY Signal: Supertrend just turned bullish[/green]")
            return "BUY", f"Volatility Switch ({last_row['volatility_switch']:.3f} > 0.5) with Supertrend bullish transition"
        
        elif last_row['supertrend_direction'] == -1 and prev_row['supertrend_direction'] == 1:
            self.console.print("[red]SELL Signal: Supertrend just turned bearish[/red]")
            return "SELL", f"Volatility Switch ({last_row['volatility_switch']:.3f} > 0.5) with Supertrend bearish transition"
        
        self.console.print("[yellow]No Supertrend transition detected[/yellow]")
        return None, None

    def calculate_sl_tp(self, signal, df):
        """Calculate Stop Loss and Take Profit based on specified rules"""
        current_price = df.iloc[-1]['close']
        prev_candle = df.iloc[-2]
        
        # Calculate initial stop loss based on previous candle
        if signal == "BUY":
            sl_price = prev_candle['low']
            sl_pips = (current_price - sl_price) * 10000
        else:  # SELL
            sl_price = prev_candle['high']
            sl_pips = (sl_price - current_price) * 10000
        
        # Adjust stop loss if it exceeds 30 pips
        if sl_pips > 30:
            sl_pips = 30
            sl_price = current_price - (0.0030 if signal == "BUY" else -0.0030)
        
        # Calculate take profit based on stop loss size
        if sl_pips <= 25:
            tp_pips = 30
        else:  # sl_pips between 25-30
            tp_pips = 10
        
        # Calculate final take profit price
        if signal == "BUY":
            tp_price = current_price + (tp_pips * 0.0001)
        else:  # SELL
            tp_price = current_price - (tp_pips * 0.0001)
        
        # Debug information
        self.console.print("\nTrade Levels:")
        self.console.print(f"Entry: {current_price:.2f}")
        self.console.print(f"Stop Loss: {sl_price:.2f} ({sl_pips:.1f} pips)")
        self.console.print(f"Take Profit: {tp_price:.2f} ({tp_pips:.1f} pips)")
        
        return sl_price, tp_price

    def execute_trade(self, signal, df):
        """Execute trade with exact entry conditions"""
        if self.trade_taken:
            self.console.print("[yellow]Trade already active, skipping new signal[/yellow]")
            return False
        
        # Get price analysis
        analyzer = PriceAnalyzer()
        analysis = analyzer.analyze_price_action(df)
        
        # Display analysis summary
        self.display_analysis_summary(analysis)
        
        # Trading conditions
        conditions = {
            'BUY': [
                analysis['trend']['direction'] == 'bullish',
                analysis['momentum']['rsi'].iloc[-1] > 40,
                analysis['momentum']['rsi'].iloc[-1] < 70,
                analysis['momentum']['macd']['histogram'].iloc[-1] > 0,
                analysis['volume']['trend'] == 'increasing',
                df['close'].iloc[-1] > analysis['volatility']['bbands']['middle'].iloc[-1]
            ],
            'SELL': [
                analysis['trend']['direction'] == 'bearish',
                analysis['momentum']['rsi'].iloc[-1] < 60,
                analysis['momentum']['rsi'].iloc[-1] > 30,
                analysis['momentum']['macd']['histogram'].iloc[-1] < 0,
                analysis['volume']['trend'] == 'increasing',
                df['close'].iloc[-1] < analysis['volatility']['bbands']['middle'].iloc[-1]
            ]
        }
        
        # Check for strong signals
        if all(conditions['BUY']):
            return self.validate_signal("BUY", df, analysis)
        elif all(conditions['SELL']):
            return self.validate_signal("SELL", df, analysis)
        
        return None

    def display_analysis_summary(self, analysis):
        """Display price analysis summary"""
        table = Table(title="Price Analysis Summary", show_header=True)
        table.add_column("Indicator")
        table.add_column("Value")
        table.add_column("Signal")
        
        # Trend
        trend_direction = analysis['trend']['direction'][-1]  # Use array indexing
        trend_strength = analysis['trend']['adx'][-1]  # Use array indexing
        table.add_row(
            "Trend",
            trend_direction,
            "🟢" if trend_direction == 'bullish' else "🔴"
        )
        
        # Momentum
        rsi = analysis['momentum']['rsi'][-1]  # Use array indexing
        table.add_row(
            "RSI",
            f"{rsi:.2f}",
            "🟢" if 40 < rsi < 70 else "🔴"
        )
        
        # Volatility
        bb_position = (
            analysis['volatility']['bbands']['upper'][-1] -
            analysis['volatility']['bbands']['lower'][-1]
        ) / analysis['volatility']['bbands']['middle'][-1]
        table.add_row(
            "Volatility",
            f"{bb_position:.2f}",
            "🟢" if 0.01 < bb_position < 0.05 else "🔴"
        )
        
        self.console.print(table)

    def validate_signal(self, signal_type, df, analysis):
        """Validate trading signal with additional checks"""
        # Check for candlestick patterns
        patterns = analysis['candlestick_patterns']
        if signal_type == "BUY":
            pattern_confirm = any([
                patterns['hammer'].iloc[-1] > 0,
                patterns['engulfing_bullish'].iloc[-1] > 0,
                patterns['morning_star'].iloc[-1] > 0
            ])
        else:
            pattern_confirm = any([
                patterns['shooting_star'].iloc[-1] > 0,
                patterns['engulfing_bearish'].iloc[-1] > 0,
                patterns['evening_star'].iloc[-1] > 0
            ])
        
        if not pattern_confirm:
            return None
        
        # Check key levels
        levels = analysis['support_resistance']
        current_price = df['close'].iloc[-1]
        
        if signal_type == "BUY":
            # Check if price is near support
            near_support = any(
                abs(current_price - support) / current_price < 0.001
                for support in levels['support']
            )
            if not near_support:
                return None
        else:
            # Check if price is near resistance
            near_resistance = any(
                abs(current_price - resistance) / current_price < 0.001
                for resistance in levels['resistance']
            )
            if not near_resistance:
                return None
        
        return signal_type

    def check_recent_trades(self):
        """Check if we have any recent trades"""
        try:
            # Get recent trades
            from_date = datetime.now() - timedelta(hours=1)
            trades = mt5.history_deals_get(from_date, datetime.now())
            
            if trades is None:
                return False
            
            # Convert to DataFrame
            trades_df = pd.DataFrame(list(trades), columns=trades[0]._asdict().keys())
            
            # Filter for our symbol
            symbol_trades = trades_df[trades_df['symbol'] == self.symbol]
            
            return len(symbol_trades) > 0
            
        except Exception as e:
            self.console.print(f"[red]Error checking recent trades: {e}[/red]")
            return False

    def calculate_stop_loss(self, signal, df):
        """Calculate stop loss and take profit levels"""
        last_row = df.iloc[-1]
        
        if signal == "BUY":
            # For buy trades:
            # SL: Below the Supertrend line or recent low
            sl_price = min(last_row['supertrend'], last_row['low'])
            entry_price = last_row['close']
            sl_pips = (entry_price - sl_price) * 10000
            tp_pips = sl_pips * 2  # 1:2 risk-reward ratio
            
        else:  # SELL
            # For sell trades:
            # SL: Above the Supertrend line or recent high
            sl_price = max(last_row['supertrend'], last_row['high'])
            entry_price = last_row['close']
            sl_pips = (sl_price - entry_price) * 10000
            tp_pips = sl_pips * 2  # 1:2 risk-reward ratio
        
        # Limit maximum stop loss
        if sl_pips > self.max_sl_pips:
            self.console.print("[yellow]Warning: Stop loss too large, adjusting...[/yellow]")
            sl_pips = self.max_sl_pips
            tp_pips = sl_pips * 2
        
        return sl_pips, tp_pips

    def execute_trade(self, signal, df):
        """Execute trade with current OHLC data"""
        current_candle = self.price_tracker.get_current_candle()
        
        # Display current market context
        self.console.print("\n[bold cyan]Market Context:[/bold cyan]")
        context_table = Table(show_header=True)
        context_table.add_column("Parameter")
        context_table.add_column("Value")
        
        context_table.add_row("Signal", signal)
        context_table.add_row("Current Open", f"{current_candle['open']:.2f}")
        context_table.add_row("Current High", f"{current_candle['high']:.2f}")
        context_table.add_row("Current Low", f"{current_candle['low']:.2f}")
        context_table.add_row("Current Close", f"{current_candle['close']:.2f}")
        
        self.console.print(context_table)
        
        # Calculate entry, SL, and TP based on current candle
        if signal == "BUY":
            entry_price = current_candle['close']  # Use current close as entry
            sl_price = current_candle['low']       # Use current low as initial SL
            tp_price = entry_price + (entry_price - sl_price) * 2  # 1:2 RR
        else:  # SELL
            entry_price = current_candle['close']  # Use current close as entry
            sl_price = current_candle['high']      # Use current high as initial SL
            tp_price = entry_price - (sl_price - entry_price) * 2  # 1:2 RR
        
        # Place the trade
        self.place_order(signal, entry_price, sl_price, tp_price)

    def place_order(self, signal, entry_price, sl_price, tp_price):
        """Place order with current OHLC data"""
        # Calculate position size
        lot_size = self.calculate_position_size(sl_price)
        
        # Display trade details before execution
        self.console.print("\n[bold cyan]🎯 Preparing to execute trade:[/bold cyan]")
        trade_table = Table(show_header=True, header_style="bold blue")
        trade_table.add_column("Parameter", style="cyan")
        trade_table.add_column("Value", justify="right")
        
        trade_table.add_row("Signal", f"[{'green' if signal == 'BUY' else 'red'}]{signal}[/]")
        trade_table.add_row("Entry Price", f"{entry_price:.2f}")
        trade_table.add_row("Stop Loss", f"{sl_price:.2f}")
        trade_table.add_row("Take Profit", f"{tp_price:.2f}")
        trade_table.add_row("Lot Size", f"{lot_size}")
        trade_table.add_row("SL Pips", f"{sl_price}")
        trade_table.add_row("TP Pips", f"{tp_price}")
        
        self.console.print(Panel(trade_table, title="Trade Details", border_style="blue"))
        
        # Prepare the trade request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 10,
            "magic": self.magic_number,
            "comment": f"Gold Bot {signal}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send the order
        self.console.print("[yellow]Sending order...[/yellow]")
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.console.print(f"[bold red]❌ Order failed: {result.comment}[/bold red]")
            self.console.print(f"[red]Error code: {result.retcode}[/red]")
            return False
                
        # Order successful
        self.console.print("[bold green]✅ Order placed successfully![/bold green]")
        
        # Display order details
        order_table = Table(show_header=True, header_style="bold green")
        order_table.add_column("Parameter", style="cyan")
        order_table.add_column("Value", justify="right")
        
        order_table.add_row("Order Ticket", str(result.order))
        order_table.add_row("Execution Price", f"{result.price:.2f}")
        order_table.add_row("Requested Price", f"{request['price']:.2f}")
        order_table.add_row("Slippage", f"{(result.price - request['price']):.5f}")
        
        self.console.print(Panel(order_table, title="Order Details", border_style="green"))
        
        return True

    def display_trade_details(self, signal, df, sl_pips, tp_pips):
        """Display trade details in a formatted table"""
        current_price = df.iloc[-1]['close']
        
        if signal == "BUY":
            sl_price = current_price - (sl_pips / 10000)
            tp_price = current_price + (tp_pips / 10000)
        else:
            sl_price = current_price + (sl_pips / 10000)
            tp_price = current_price - (tp_pips / 10000)
            
        trade_table = Table(title="Trade Details", show_header=True, header_style="bold blue")
        trade_table.add_column("Parameter", style="cyan")
        trade_table.add_column("Value", justify="right")
        
        trade_table.add_row("Signal", f"[{'green' if signal == 'BUY' else 'red'}]{signal}[/]")
        trade_table.add_row("Entry Price", f"{current_price:.2f}")
        trade_table.add_row("Stop Loss", f"{sl_price:.2f}")
        trade_table.add_row("Take Profit", f"{tp_price:.2f}")
        trade_table.add_row("SL Pips", str(sl_pips))
        trade_table.add_row("TP Pips", str(tp_pips))
        
        self.console.print(Panel(trade_table, border_style="blue"))

    def check_connection(self):
        """Check MT5 connection and symbol availability"""
        self.console.print("\n[yellow]Checking MT5 Connection...[/yellow]")
        
        # Check MT5 initialization
        if not mt5.initialize():
            self.console.print("[red]Failed to initialize MT5[/red]")
            return False
        
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            self.console.print("[red]Failed to get account info[/red]")
            return False
        
        self.console.print(f"[green]Connected to: {account_info.server}[/green]")
        self.console.print(f"[green]Account: {account_info.login}[/green]")
        
        # Check symbol info
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            self.console.print(f"[red]Symbol {self.symbol} not found![/red]")
            return False
        
        if not symbol_info.visible:
            self.console.print(f"[yellow]Symbol {self.symbol} is not visible, trying to add...[/yellow]")
            if not mt5.symbol_select(self.symbol, True):
                self.console.print(f"[red]Failed to select {self.symbol}![/red]")
                return False
        
        # Get latest tick
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            self.console.print("[red]Failed to get latest tick[/red]")
            return False
        
        self.console.print(f"[green]Current Bid: {tick.bid}[/green]")
        self.console.print(f"[green]Current Ask: {tick.ask}[/green]")
        
        return True

    def get_live_data(self):
        """Get the most recent candle data"""
        try:
            # Get the latest tick
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                self.console.print("[red]Failed to get latest tick[/red]")
                return None
            
            # Get the latest completed candle
            rates = mt5.copy_rates_from(
                self.symbol,
                self.timeframe,
                datetime.now(),
                1
            )
            
            if rates is None:
                self.console.print("[red]Failed to get latest candle[/red]")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Add current tick data
            df['current_bid'] = tick.bid
            df['current_ask'] = tick.ask
            
            return df
            
        except Exception as e:
            self.console.print(f"[red]Error getting live data: {e}[/red]")
            return None

    def monitor_live_prices(self):
        """Monitor live price updates with OHLC tracking"""
        def generate_price_table():
            """Generate current candle display table"""
            current_candle = self.price_tracker.get_current_candle()
            
            table = Table(show_header=True, header_style="bold green")
            table.add_column("Time")
            table.add_column("Open")
            table.add_column("High")
            table.add_column("Low")
            table.add_column("Close")
            table.add_column("Volume")
            
            if current_candle['time'] is not None:
                table.add_row(
                    current_candle['time'].strftime("%H:%M:%S"),
                    f"{current_candle['open']:.2f}",
                    f"{current_candle['high']:.2f}",
                    f"{current_candle['low']:.2f}",
                    f"{current_candle['close']:.2f}",
                    str(current_candle['volume'])
                )
            
            return table

        def process_tick(tick):
            """Process new tick data"""
            self.price_tracker.update(tick)
            
            # Check if new candle formed
            if self.price_tracker.is_candle_closed(pd.to_datetime(tick.time, unit='s')):
                # Get recent candles for analysis
                df = self.price_tracker.get_recent_candles()
                
                # Calculate indicators
                df = self.calculate_supertrend(df)
                df = self.calculate_volatility(df)
                
                # Check for signals
                signal = self.get_trading_signal(df)
                if signal:
                    self.execute_trade(signal, df)

        # Main monitoring loop
        with Live(generate_price_table(), refresh_per_second=2) as live:
            while True:
                try:
                    # Get latest tick
                    tick = mt5.symbol_info_tick(self.symbol)
                    if tick is not None:
                        process_tick(tick)
                        live.update(generate_price_table())
                    
                    time.sleep(0.1)  # Check every 100ms
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.console.print(f"[red]Error: {str(e)}[/red]")
                    time.sleep(1)

    def run(self):
        """Main bot loop"""
        if not self.check_connection():
            self.console.print("[bold red]Failed to establish proper connection![/bold red]")
            return
        
        self.console.print("[bold green]Bot started successfully![/bold green]")
        self.console.print(f"[green]Trading hours: {self.trading_start} - {self.trading_end} IST[/green]")
        
        try:
            self.monitor_live_prices()
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Bot shutdown requested...[/yellow]")
        finally:
            mt5.shutdown()

    def is_trading_time(self):
        """Check if current time is within trading hours"""
        # Convert current UTC time to IST
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist).time()
        
        # Handle time range that crosses midnight
        if self.trading_start > self.trading_end:
            # If current time is after start OR before end
            return current_time >= self.trading_start or current_time <= self.trading_end
        else:
            # If current time is between start and end
            return self.trading_start <= current_time <= self.trading_end
            
        # self.console.print(f"[cyan]Current time (IST): {current_time}[/cyan]")
        # self.console.print(f"[cyan]Trading window: {self.trading_start} - {self.trading_end}[/cyan]")
        
        # return is_trading_time

class PriceTracker:
    def __init__(self, timeframe_minutes=5):
        self.timeframe_minutes = timeframe_minutes
        self.current_candle = {
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'time': None,
            'volume': 0
        }
        self.candle_history = []
        self.last_tick_time = None

    def update(self, tick):
        """Update current candle with new tick"""
        current_time = pd.to_datetime(tick.time, unit='s')
        price = tick.bid  # Using bid price for calculations
        
        # Initialize new candle if needed
        if self.current_candle['time'] is None:
            self.start_new_candle(current_time, price)
        
        # Check if we need to close current candle
        elif self.is_candle_closed(current_time):
            self.close_current_candle(price)
            self.start_new_candle(current_time, price)
        
        # Update current candle
        else:
            self.current_candle['high'] = max(self.current_candle['high'], price)
            self.current_candle['low'] = min(self.current_candle['low'], price)
            self.current_candle['close'] = price
            self.current_candle['volume'] += 1
        
        self.last_tick_time = current_time

    def start_new_candle(self, time, price):
        """Start a new candle"""
        self.current_candle = {
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'time': time,
            'volume': 1
        }

    def close_current_candle(self, last_price):
        """Close current candle and add to history"""
        self.current_candle['close'] = last_price
        self.candle_history.append(self.current_candle.copy())
        
        # Keep only last 1000 candles
        if len(self.candle_history) > 1000:
            self.candle_history.pop(0)

    def is_candle_closed(self, current_time):
        """Check if current candle should be closed"""
        if self.current_candle['time'] is None:
            return False
            
        minutes_elapsed = (current_time - self.current_candle['time']).total_seconds() / 60
        return minutes_elapsed >= self.timeframe_minutes

    def get_current_candle(self):
        """Get current candle data"""
        return self.current_candle.copy()

    def get_recent_candles(self, n=100):
        """Get recent candles as DataFrame"""
        all_candles = self.candle_history[-n:] + [self.current_candle]
        return pd.DataFrame(all_candles)

class PriceAnalyzer:
    def __init__(self):
        self.key_levels = []
        self.swing_highs = []
        self.swing_lows = []

    def analyze_price_action(self, df):
        """Comprehensive price analysis"""
        analysis = {
            'candlestick_patterns': self.detect_candlestick_patterns(df),
            'support_resistance': self.find_key_levels(df),
            'momentum': self.analyze_momentum(df),
            'volatility': self.analyze_volatility(df),
            'trend': self.analyze_trend(df),
            # 'volume': self.analyze_volume(df)
        }
        return analysis

    def detect_candlestick_patterns(self, df):
        """Detect candlestick patterns"""
        patterns = {}
        
        # Bullish patterns
        patterns['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        patterns['engulfing_bullish'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        patterns['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        
        # Bearish patterns
        patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
        patterns['engulfing_bearish'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        patterns['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
        
        return patterns

    def find_key_levels(self, df, window=20):
        """Find support and resistance levels"""
        levels = {
            'support': [],
            'resistance': [],
            'pivot_points': {}
        }
        
        # Calculate Pivot Points
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        pivot = typical_price.rolling(window=window).mean()
        
        levels['pivot_points']['P'] = pivot.iloc[-1]
        levels['pivot_points']['R1'] = pivot + (pivot - levels['pivot_points']['P'])
        levels['pivot_points']['S1'] = pivot - (levels['pivot_points']['P'] - pivot)
        
        # Find swing highs and lows
        for i in range(window, len(df)-window):
            if self.is_swing_high(df, i, window):
                levels['resistance'].append(df['high'].iloc[i])
            if self.is_swing_low(df, i, window):
                levels['support'].append(df['low'].iloc[i])
        
        return levels

    def analyze_momentum(self, df):
        """Analyze price momentum"""
        momentum = {}
        
        # RSI
        momentum['rsi'] = talib.RSI(df['close'])
        
        # MACD
        macd, signal, hist = talib.MACD(df['close'])
        momentum['macd'] = {
            'macd': macd,
            'signal': signal,
            'histogram': hist
        }
        
        # Stochastic
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
        momentum['stochastic'] = {
            'k': slowk,
            'd': slowd
        }
        
        # Rate of Change
        momentum['roc'] = talib.ROC(df['close'])
        
        return momentum

    def analyze_volatility(self, df):
        """Analyze price volatility"""
        volatility = {}
        
        # ATR
        volatility['atr'] = talib.ATR(df['high'], df['low'], df['close'])
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'])
        volatility['bbands'] = {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'bandwidth': (upper - lower) / middle
        }
        
        # Historical Volatility
        volatility['hist_vol'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        return volatility

    def analyze_trend(self, df):
        """Analyze price trend"""
        trend = {}
        
        # Multiple EMAs
        trend['ema_9'] = talib.EMA(df['close'], timeperiod=9)
        trend['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        trend['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        
        # ADX for trend strength
        trend['adx'] = talib.ADX(df['high'], df['low'], df['close'])
        
        # Trend direction
        trend['direction'] = np.where(
            trend['ema_20'] > trend['ema_50'],
            'bullish',
            'bearish'
        )
        
        return trend

    def analyze_volume(self, df):
        """Analyze volume patterns"""
        volume = {}
        
        # Volume SMA
        volume['sma'] = df['volume'].rolling(window=20).mean()
        
        # Volume relative to average
        volume['relative'] = df['volume'] / volume['sma']
        
        # OBV (On Balance Volume)
        volume['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Volume trend
        volume['trend'] = np.where(
            volume['relative'] > 1,
            'increasing',
            'decreasing'
        )
        
        return volume

    def is_swing_high(self, df, index, window):
        """Check if price point is a swing high"""
        return df['high'].iloc[index] == df['high'].iloc[index-window:index+window].max()

    def is_swing_low(self, df, index, window):
        """Check if price point is a swing low"""
        return df['low'].iloc[index] == df['low'].iloc[index-window:index+window].min()