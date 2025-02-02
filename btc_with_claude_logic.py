import MetaTrader5 as mt5
import pandas as pd
import time
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta, timezone
from rich.console import Console
from rich.table import Table
import pytz

class BTCTradingBot:
    def __init__(self, timeframe=mt5.TIMEFRAME_M5, name="BTCBot"):
        self.name = name
        self.timeframe = timeframe
        self.running = True
        self.console = Console()
        self.symbol = "BTCUSD"
        self.lot_size = 0.1    # Adjusted for BTC
        self.max_sl_pips = 30   # Maximum 30 pips stop loss
        self.trade_taken = False
        self.count = 2000
        
        # Initialize indicators
        self.ema_short = 5   # 5 EMA
        self.ema_long = 20   # 20 EMA
        self.rsi_period = 14 # RSI period
        
        if not self.initialize_mt5():
            self.console.print("[bold red]MT5 initialization failed![/bold red]")
            return
        print("INITIALIZED ...")

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
        """Calculate indicators for BTC strategy"""
        # Bollinger Bands (20,2) - Good for crypto volatility
        df['bb_middle'] = ta.sma(df['close'], length=20)
        df['bb_upper'], df['bb_lower'] = ta.bbands(df['close'], length=20, std=2)
        
        # RSI with modified settings for crypto
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # Volume Moving Average
        df['volume_ma'] = ta.sma(df['tick_volume'], length=20)
        
        # ATR for volatility measurement
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        return df

    def get_trading_signal(self, df):
        """Generate trading signals for BTC"""
        if df.empty or len(df) < 20:
            return None, None
            
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # Check if trade is active
        if self.get_active_trades():
            return None, None

        # Debug info
        self.console.print("\n=== Market Conditions ===")
        self.console.print(f"Current Price: ${last_row['close']:.2f}")
        self.console.print(f"RSI: {last_row['rsi']:.2f}")
        self.console.print(f"ATR: {last_row['atr']:.2f}")
        
        # BUY Conditions:
        buy_conditions = (
            # Price near lower Bollinger Band
            last_row['close'] <= last_row['bb_lower'] * 1.005 and
            # RSI showing oversold but recovering
            30 <= last_row['rsi'] <= 45 and
            # Volume confirmation
            last_row['tick_volume'] > last_row['volume_ma'] * 1.2 and
            # Minimum volatility
            last_row['atr'] > 100  # $100 minimum volatility
        )
        
        # SELL Conditions:
        sell_conditions = (
            # Price near upper Bollinger Band
            last_row['close'] >= last_row['bb_upper'] * 0.995 and
            # RSI showing overbought
            last_row['rsi'] >= 70 and
            # Volume confirmation
            last_row['tick_volume'] > last_row['volume_ma'] * 1.2 and
            # Minimum volatility
            last_row['atr'] > 100
        )

        if buy_conditions:
            return "BUY", "Price at support with volume confirmation"
        elif sell_conditions:
            return "SELL", "Price at resistance with volume confirmation"
        
        return None, None

    def calculate_sl_tp(self, signal, df):
        """Calculate Stop Loss and Take Profit based on ATR"""
        current_price = df.iloc[-1]['close']
        atr = df.iloc[-1]['atr']
        
        if signal == "BUY":
            sl_price = current_price - (atr * 1.5)  # 1.5x ATR for SL
            tp_price = current_price + (atr * 2.0)  # 2x ATR for TP
        else:
            sl_price = current_price + (atr * 1.5)
            tp_price = current_price - (atr * 2.0)
        
        self.console.print("\n=== Trade Levels ===")
        self.console.print(f"Entry: ${current_price:.2f}")
        self.console.print(f"SL: ${sl_price:.2f}")
        self.console.print(f"TP: ${tp_price:.2f}")
        
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
                "comment": f"Python BTC {signal} with claude logic",
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
        if trade['type'] == "BUY" and last_row['ema_5'] < last_row['ema_20']:
            should_exit = True
            exit_reason = f"EMA cross bearish at {last_row['close']:.2f}"
        elif trade['type'] == "SELL" and last_row['ema_5'] > last_row['ema_20']:
            should_exit = True
            exit_reason = f"EMA cross bullish at {last_row['close']:.2f}"
        
        # Check RSI extremes
        if trade['type'] == "BUY" and last_row['rsi'] > 70:
            should_exit = True
            exit_reason = f"RSI overbought at {last_row['rsi']:.2f}"
        elif trade['type'] == "SELL" and last_row['rsi'] < 30:
            should_exit = True
            exit_reason = f"RSI oversold at {last_row['rsi']:.2f}"
        
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
        config_table.add_row("RSI Period", str(self.rsi_period))
        config_table.add_row("Max SL (pips)", str(self.max_sl_pips))
        config_table.add_row("Lot Size", str(self.lot_size))
        
        self.console.print(config_table)
        
        # Display current market data and trade status
        self.display_market_status()

    def run(self):
        """Main bot loop with status display"""
        self.console.print("[bold green]Starting BTC Trading Bot...[/bold green]")
        
        while True:
            try:
                # Clear screen for clean display
                self.console.clear()
                
                # Get market data
                df = self.get_rates_df(5000)
                if df is None:
                    continue
                    
                # Calculate indicators
                df = self.calculate_indicators(df)
                
                # Display bot status
                self.display_bot_status()
                
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
            profit_pips = close_price - position.price_open  # For BTC, 1 pip = $1
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
                "deviation": 20,
                "magic": 234000,
                "comment": f"Python close: {reason}",
                "type_filling": mt5.ORDER_FILLING_FOK,
                "type_time": mt5.ORDER_TIME_GTC
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.console.print("[green]Trade closed successfully![/green]")
                self.console.print(f"Profit: ${position.profit:.2f} ({profit_pips:.1f} pips)")
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