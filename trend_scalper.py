import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
import logging
from rich.console import Console
from rich.table import Table
import time

class TrendScalper:
    def __init__(self, symbol="BTCUSD", timeframe=mt5.TIMEFRAME_M5):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.console = Console()
        
        # Initialize basic parameters
        self.symbol = symbol
        self.timeframe = timeframe
        
        # MT5 Credentials - should be moved to environment variables
        self.mt5_login = 182992485
        self.mt5_password = "QAZwsx456!"
        self.mt5_server = "Exness-MT5Trial6"
        
        # Strategy Parameters
        self.supertrend_period = 10
        self.supertrend_multiplier = 3
        self.psar_step = 0.02
        self.psar_max = 0.2
        self.adx_period = 14
        self.min_adx = 25
        self.volume_ma = 20
        self.min_volume_ratio = 0.8  # Lowered from 1.5x to 0.8x
        
        # Risk Management
        self.position_size = 0.1  # 0.1 BTC
        self.profit_target = 20   # $20 per trade
        self.daily_profit_target = 600  # $600 per day
        self.max_risk = 200       # $200 max loss per trade
        self.daily_loss_limit = -300  # Stop trading if down $300
        
        # Trailing Stop Parameters
        self.trailing_enabled = True
        self.trailing_activation = 10  # Start trailing at $10 profit
        self.trailing_step = 5        # Move SL up every $5
        self.min_profit_lock = 5      # Minimum profit to lock in
        self.profit_target_min = 20   # Start taking profits at $20
        self.profit_target_max = 25   # Must take profits by $25
        
        # State tracking
        self.daily_profit = 0
        self.trades_today = []
        self.last_trade_time = None
        self.min_trade_interval = 60  # Minimum seconds between trades
        self.active_trades = []
        
        # Trade tracking
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        
        # Signal tracking
        self.last_signal_time = None
        self.last_signal_price = None
        self.min_signal_change = 0.001  # 0.1% price change needed for new signal
        
        # Monthly Target Tracking
        self.monthly_target = 13000  # $13,000 per month
        self.monthly_profit = 0
        self.trades_this_month = 0
        self.month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Load previous day's trades if exists
        self.load_trade_history()
        
        # Load this month's trades
        self.load_monthly_trades()
        
        if not self.initialize_mt5():
            raise Exception("MT5 initialization failed!")
        
        # Mean Reversion Parameters
        self.price_levels = []  # Track price levels where trades occurred
        self.max_levels = 5     # Maximum number of price levels to track
        self.level_threshold = 50  # Distance between price levels ($)
        self.reversion_active = False  # Whether we're in mean reversion mode
        
    def load_trade_history(self):
        """Load trade history from file"""
        try:
            today = datetime.now().date()
            history_orders = mt5.history_orders_get(
                datetime(today.year, today.month, today.day),
                datetime.now()
            )
            
            if history_orders is None:
                self.logger.info("No trade history found for today")
                return
                
            for order in history_orders:
                if order.symbol == self.symbol:
                    profit = order.profit
                    self.daily_profit += profit
                    self.trade_count += 1
                    if profit > 0:
                        self.wins += 1
                    else:
                        self.losses += 1
                    
            self.logger.info(f"Loaded today's history: {self.trade_count} trades, ${self.daily_profit:.2f} profit")
            
        except Exception as e:
            self.logger.error(f"Error loading trade history: {str(e)}")
            
    def load_monthly_trades(self):
        """Load this month's trade history"""
        try:
            history_orders = mt5.history_orders_get(
                self.month_start,
                datetime.now()
            )
            
            if history_orders is None:
                self.logger.info("No trade history found for this month")
                return
                
            for order in history_orders:
                if order.symbol == self.symbol:
                    self.monthly_profit += order.profit
                    self.trades_this_month += 1
                    
            self.logger.info(f"Loaded month's history: {self.trades_this_month} trades, ${self.monthly_profit:.2f} profit")
            
        except Exception as e:
            self.logger.error(f"Error loading monthly history: {str(e)}")
            
    def update_trade_stats(self, profit):
        """Update trade statistics"""
        self.daily_profit += profit
        self.trade_count += 1
        if profit > 0:
            self.wins += 1
        else:
            self.losses += 1

    def initialize_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            self.logger.error("❌ MT5 initialization failed!")
            return False
            
        # Check if symbol is available
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            self.logger.error(f"❌ {self.symbol} not found in MT5!")
            return False
            
        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                self.logger.error(f"❌ {self.symbol} not visible in Market Watch!")
                return False
            
        # Try to login
        if not mt5.login(login=self.mt5_login, password=self.mt5_password, server=self.mt5_server):
            self.logger.error("❌ MT5 login failed!")
            mt5.shutdown()
            return False
            
        self.logger.info("✅ MT5 initialized successfully")
        self.logger.info(f"Trading {self.symbol} on {self.timeframe} timeframe")
        return True

    def calculate_indicators(self, df):
        """Calculate all required indicators"""
        try:
            # Calculate SuperTrend
            supertrend = ta.supertrend(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                length=self.supertrend_period,
                multiplier=self.supertrend_multiplier
            )
            
            # The correct column names with decimal point
            supert_col = f"SUPERT_{self.supertrend_period}_{self.supertrend_multiplier}.0"
            supert_dir_col = f"SUPERTd_{self.supertrend_period}_{self.supertrend_multiplier}.0"
            
            df['supertrend'] = supertrend[supert_col]
            df['supertrend_direction'] = supertrend[supert_dir_col]
            
            # Calculate PSAR
            psar = ta.psar(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                af0=self.psar_step,
                af=self.psar_step,
                max_af=self.psar_max
            )
            df['psar'] = psar['PSARl_' + str(self.psar_step) + '_' + str(self.psar_max)]
            df['psar_direction'] = np.where(df['close'] > df['psar'], 1, -1)
            
            # Calculate ADX
            adx = ta.adx(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                length=self.adx_period
            )
            df['adx'] = adx['ADX_' + str(self.adx_period)]
            
            # Calculate Volume MA
            df['volume_ma'] = ta.sma(df['tick_volume'], length=self.volume_ma)
            df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return None

    def get_trading_signal(self, df):
        """Get trading signal based on indicators"""
        try:
            last_row = df.iloc[-1]
            current_price = last_row['close']
            
            # Check if we've hit daily targets
            if self.daily_profit >= self.daily_profit_target:
                self.logger.info("Daily profit target reached")
                return None, "Daily profit target reached"
                
            if self.daily_profit <= self.daily_loss_limit:
                self.logger.info("Daily loss limit reached")
                return None, "Daily loss limit reached"
            
            # Check trade interval
            if self.last_trade_time and \
               (datetime.now() - self.last_trade_time).total_seconds() < self.min_trade_interval:
                self.logger.info("Minimum trade interval not met")
                return None, "Minimum trade interval not met"
            
            # Check if price has moved enough since last signal
            if self.last_signal_price is not None:
                price_change = abs(current_price - self.last_signal_price) / self.last_signal_price
                if price_change < self.min_signal_change:
                    self.logger.info(f"Price hasn't moved enough since last signal ({price_change*100:.2f}%)")
                    return None, "Waiting for price movement"
            
            # Log indicator values
            self.logger.info(f"\nIndicator Values:")
            self.logger.info(f"SuperTrend: {'Bullish' if last_row['supertrend_direction'] == 1 else 'Bearish'}")
            self.logger.info(f"PSAR: {'Bullish' if last_row['psar_direction'] == 1 else 'Bearish'}")
            self.logger.info(f"ADX: {last_row['adx']:.1f}")
            self.logger.info(f"Volume Ratio: {last_row['volume_ratio']:.2f}x")
            
            # Check trend alignment
            supertrend_bullish = last_row['supertrend_direction'] == 1
            psar_bullish = last_row['psar_direction'] == 1
            
            # Check trend strength
            strong_trend = last_row['adx'] > self.min_adx
            good_volume = last_row['volume_ratio'] > self.min_volume_ratio
            
            self.logger.info(f"\nCondition Checks:")
            self.logger.info(f"Strong Trend (ADX > {self.min_adx}): {strong_trend}")
            self.logger.info(f"Good Volume (> {self.min_volume_ratio}x): {good_volume}")
            self.logger.info(f"Trends Aligned: {supertrend_bullish == psar_bullish}")
            
            # Entry conditions
            if strong_trend and good_volume:
                if supertrend_bullish and psar_bullish:
                    self.logger.info("✅ Taking BUY trade")
                    self.last_signal_time = datetime.now()
                    self.last_signal_price = current_price
                    return "BUY", "Bullish trend confirmed"
                elif not supertrend_bullish and not psar_bullish:
                    self.logger.info("✅ Taking SELL trade")
                    self.last_signal_time = datetime.now()
                    self.last_signal_price = current_price
                    return "SELL", "Bearish trend confirmed"
                else:
                    self.logger.info("❌ Trends not aligned")
            else:
                if not strong_trend:
                    self.logger.info("❌ Trend not strong enough")
                if not good_volume:
                    self.logger.info("❌ Volume too low")
            
            return None, "No signal"
            
        except Exception as e:
            self.logger.error(f"Error getting trading signal: {str(e)}")
            return None, str(e)

    def calculate_position_params(self, signal, df):
        """Calculate entry, stop loss, and take profit levels"""
        current_price = df.iloc[-1]['close']
        
        if signal == "BUY":
            entry_price = current_price
            stop_loss = entry_price - (self.max_risk / self.position_size)
            take_profit = entry_price + (self.profit_target / self.position_size)
        else:  # SELL
            entry_price = current_price
            stop_loss = entry_price + (self.max_risk / self.position_size)
            take_profit = entry_price - (self.profit_target / self.position_size)
            
        return entry_price, stop_loss, take_profit

    def execute_trade(self, signal, df):
        """Execute a trade with the calculated parameters"""
        try:
            entry_price, sl_price, tp_price = self.calculate_position_params(signal, df)
            
            self.logger.info(f"\nTrade Parameters:")
            self.logger.info(f"Signal: {signal}")
            self.logger.info(f"Entry: ${entry_price:.2f}")
            self.logger.info(f"Stop Loss: ${sl_price:.2f}")
            self.logger.info(f"Take Profit: ${tp_price:.2f}")
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.position_size,
                "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Trend Scalper {signal}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            self.logger.info("\nSending trade request...")
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"❌ Order failed: {result.comment}")
                return False
                
            self.logger.info(f"✅ Trade executed successfully")
            self.last_trade_time = datetime.now()
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return False

    def calculate_potential_levels(self, current_price, signal_type):
        """Calculate potential entry, SL and TP levels"""
        if signal_type == "BUY":
            entry = current_price
            tp = entry + (self.profit_target / self.position_size)
            sl = entry - (self.max_risk / self.position_size)
        else:  # SELL
            entry = current_price
            tp = entry - (self.profit_target / self.position_size)
            sl = entry + (self.max_risk / self.position_size)
        return entry, tp, sl

    def display_status(self, df):
        """Display current trading status"""
        self.console.clear()
        last_row = df.iloc[-1]
        current_price = last_row['close']
        
        # Create status table
        status_table = Table(title="Trend Scalper Status")
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", justify="right", style="green")
        
        # Market Info
        status_table.add_row("Current Price", f"${current_price:.2f}")
        
        # Monthly Progress
        remaining_days = 22 - datetime.now().day  # Assuming 22 trading days
        if remaining_days < 0:
            remaining_days = 0
            
        trades_needed = int((self.monthly_target - self.monthly_profit) / 10)  # Assuming $10 per trade
        trades_per_day = trades_needed / remaining_days if remaining_days > 0 else 0
        
        status_table.add_row("Monthly Target", f"${self.monthly_target:,.2f}")
        status_table.add_row("Monthly Profit", f"${self.monthly_profit:,.2f}")
        status_table.add_row("Progress", f"{(self.monthly_profit/self.monthly_target)*100:.1f}%")
        status_table.add_row("Trades This Month", str(self.trades_this_month))
        status_table.add_row("Remaining Days", str(remaining_days))
        if trades_needed > 0:
            status_table.add_row("Trades Needed", f"{trades_needed:,} ({trades_per_day:.1f}/day)")
        
        # Daily Progress
        status_table.add_row("Daily Profit", f"${self.daily_profit:.2f}")
        status_table.add_row("Trade Count", f"{self.trade_count} (W: {self.wins} L: {self.losses})")
        
        # Win Rate
        if self.trade_count > 0:
            win_rate = (self.wins / self.trade_count) * 100
            status_table.add_row("Win Rate", f"{win_rate:.1f}%")
            
        # Trade Readiness first
        trade_ready = True
        ready_messages = []
        
        # Check cooldown
        if self.last_trade_time:
            seconds_since_trade = (datetime.now() - self.last_trade_time).total_seconds()
            if seconds_since_trade < self.min_trade_interval:
                trade_ready = False
                wait_time = self.min_trade_interval - seconds_since_trade
                ready_messages.append(f"Cooldown: {wait_time:.0f}s remaining")
        
        # Check price movement
        if self.last_signal_price:
            price_change = abs(current_price - self.last_signal_price) / self.last_signal_price
            if price_change < self.min_signal_change:
                trade_ready = False
                needed_move = (self.min_signal_change - price_change) * current_price
                ready_messages.append(f"Need ${needed_move:.2f} more price movement")
        
        # Check daily targets
        if self.daily_profit >= self.daily_profit_target:
            trade_ready = False
            ready_messages.append("Daily profit target reached")
        elif self.daily_profit <= self.daily_loss_limit:
            trade_ready = False
            ready_messages.append("Daily loss limit reached")
            
        # Check indicators
        if last_row['adx'] <= self.min_adx:
            trade_ready = False
            ready_messages.append(f"ADX too low: {last_row['adx']:.1f} < {self.min_adx}")
            
        if last_row['volume_ratio'] <= self.min_volume_ratio:
            trade_ready = False
            ready_messages.append(f"Volume too low: {last_row['volume_ratio']:.2f}x < {self.min_volume_ratio}x")
            
        # Check if trends are aligned
        supertrend_bullish = last_row['supertrend_direction'] == 1
        psar_bullish = last_row['psar_direction'] == 1
        if supertrend_bullish != psar_bullish:
            trade_ready = False
            ready_messages.append("Indicators not aligned")
        
        # Add trade readiness status first
        if trade_ready:
            status_table.add_row("Trade Ready", "[bold green]✅ Ready to trade[/bold green]")
        else:
            status_table.add_row("Trade Ready", "[bold red]❌ Not ready[/bold red]")
            for msg in ready_messages:
                status_table.add_row("", f"[yellow]{msg}[/yellow]")
            
        # Get potential trade direction and show levels (only if trends aligned)
        if supertrend_bullish == psar_bullish:
            potential_signal = "BUY" if supertrend_bullish else "SELL"
            entry, tp, sl = self.calculate_potential_levels(current_price, potential_signal)
            
            # Show potential levels but with warning if not ready
            status_table.add_row("Potential Setup", "[yellow]If conditions are met:[/yellow]" if not trade_ready else "")
            status_table.add_row("Signal Type", f"[{'green' if potential_signal == 'BUY' else 'red'}]{potential_signal}[/{'green' if potential_signal == 'BUY' else 'red'}]")
            status_table.add_row("Entry Around", f"${entry:.2f}")
            status_table.add_row("Take Profit", f"${tp:.2f} ({self.profit_target:.2f}$)")
            status_table.add_row("Stop Loss", f"${sl:.2f} (-{self.max_risk:.2f}$)")
            
            # Calculate risk-reward ratio
            risk = abs(entry - sl)
            reward = abs(entry - tp)
            rr_ratio = reward / risk if risk > 0 else 0
            status_table.add_row("Risk/Reward", f"{rr_ratio:.2f}")
            
        # Trend Indicators
        supertrend_status = "🟢 Bullish" if last_row['supertrend_direction'] == 1 else "🔴 Bearish"
        psar_status = "🟢 Bullish" if last_row['psar_direction'] == 1 else "🔴 Bearish"
        
        status_table.add_row("SuperTrend", supertrend_status)
        status_table.add_row("PSAR", psar_status)
        
        # Confirmations
        adx_color = "green" if last_row['adx'] > self.min_adx else "red"
        volume_color = "green" if last_row['volume_ratio'] > self.min_volume_ratio else "red"
        
        status_table.add_row("ADX", f"[{adx_color}]{last_row['adx']:.1f}[/{adx_color}] (min: {self.min_adx})")
        status_table.add_row("Volume Ratio", f"[{volume_color}]{last_row['volume_ratio']:.2f}x[/{volume_color}] (min: {self.min_volume_ratio}x)")
        
        # Trade Parameters
        status_table.add_row("Take Profit", f"${self.profit_target}")
        status_table.add_row("Position Size", f"{self.position_size} BTC")
        
        # Mean Reversion Status
        avg_level = self.calculate_average_level()
        if avg_level is not None:
            status_table.add_row("Average Level", f"${avg_level:.2f}")
            status_table.add_row("Price Levels", ", ".join([f"${p:.2f}" for p in self.price_levels]))
            if self.reversion_active:
                status_table.add_row("Strategy Mode", "Mean Reversion")
            else:
                status_table.add_row("Strategy Mode", "Trend Following")
        
        self.console.print(status_table)
        
        # Print active trade if exists
        if self.active_trades:
            trade = self.active_trades[0]
            trade_table = Table(title="Active Trade", style="bold")
            trade_table.add_column("Detail", style="cyan")
            trade_table.add_column("Value", justify="right")
            
            trade_table.add_row("Type", "BUY" if trade.type == mt5.ORDER_TYPE_BUY else "SELL")
            trade_table.add_row("Entry Price", f"${trade.price_open:.2f}")
            trade_table.add_row("Current Profit", f"${trade.profit:.2f}")
            
            self.console.print(trade_table)

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
        
        # Calculate current profit
        if position.type == 0:  # BUY
            current_profit = (current_price - position.price_open) * self.position_size
            
            # Log current state
            self.logger.info(f"\n=== Trailing Stop Check (BUY) ===")
            self.logger.info(f"Entry Price: ${position.price_open:.2f}")
            self.logger.info(f"Current Price: ${current_price:.2f}")
            self.logger.info(f"Current Profit: ${current_profit:.2f}")
            self.logger.info(f"Current SL: ${position.sl:.2f}")
            
            if current_profit >= self.trailing_activation:
                # Calculate how many steps we've moved
                profit_steps = int((current_profit - self.trailing_activation) / self.trailing_step)
                # Calculate profit to lock in (minimum $5)
                profit_to_lock = max(
                    self.min_profit_lock,
                    self.trailing_activation + (profit_steps * self.trailing_step) - self.trailing_step
                )
                
                # Calculate new stop loss
                new_sl = position.price_open + (profit_to_lock / self.position_size)
                
                self.logger.info(f"Profit Steps: {profit_steps}")
                self.logger.info(f"Locking in: ${profit_to_lock:.2f}")
                self.logger.info(f"New SL: ${new_sl:.2f}")
                
                # If profit exceeds our target range (20-25), close the trade
                if current_profit >= self.profit_target_max:
                    self.logger.info(f"✅ Closing trade at max target: ${current_price:.2f} (Profit: ${current_profit:.2f})")
                    self.close_trade(position, "Hit max profit target")
                    return
                elif current_profit >= self.profit_target_min:
                    self.logger.info("In profit-taking zone ($20-$25)")
                
                # Only update if new SL is higher
                if new_sl > position.sl:
                    self.logger.info(f"📈 Moving stop loss up: ${position.sl:.2f} → ${new_sl:.2f}")
                    self.logger.info(f"This locks in ${profit_to_lock:.2f} profit")
                    
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position.ticket,
                        "symbol": self.symbol,
                        "sl": new_sl,
                        "tp": position.tp,  # Keep original TP
                        "magic": 234000
                    }
                    
                    result = mt5.order_send(request)
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        self.logger.error(f"Failed to update trailing stop: {result.comment}")
                        
        else:  # SELL
            current_profit = (position.price_open - current_price) * self.position_size
            
            # Log current state
            self.logger.info(f"\n=== Trailing Stop Check (SELL) ===")
            self.logger.info(f"Entry Price: ${position.price_open:.2f}")
            self.logger.info(f"Current Price: ${current_price:.2f}")
            self.logger.info(f"Current Profit: ${current_profit:.2f}")
            self.logger.info(f"Current SL: ${position.sl:.2f}")
            
            if current_profit >= self.trailing_activation:
                # Calculate how many steps we've moved
                profit_steps = int((current_profit - self.trailing_activation) / self.trailing_step)
                # Calculate profit to lock in (minimum $5)
                profit_to_lock = max(
                    self.min_profit_lock,
                    self.trailing_activation + (profit_steps * self.trailing_step) - self.trailing_step
                )
                
                # Calculate new stop loss
                new_sl = position.price_open - (profit_to_lock / self.position_size)
                
                self.logger.info(f"Profit Steps: {profit_steps}")
                self.logger.info(f"Locking in: ${profit_to_lock:.2f}")
                self.logger.info(f"New SL: ${new_sl:.2f}")
                
                # If profit exceeds our target range (20-25), close the trade
                if current_profit >= self.profit_target_max:
                    self.logger.info(f"✅ Closing trade at max target: ${current_price:.2f} (Profit: ${current_profit:.2f})")
                    self.close_trade(position, "Hit max profit target")
                    return
                elif current_profit >= self.profit_target_min:
                    self.logger.info("In profit-taking zone ($20-$25)")
                
                # Only update if new SL is lower
                if new_sl < position.sl:
                    self.logger.info(f"📉 Moving stop loss down: ${position.sl:.2f} → ${new_sl:.2f}")
                    self.logger.info(f"This locks in ${profit_to_lock:.2f} profit")
                    
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position.ticket,
                        "symbol": self.symbol,
                        "sl": new_sl,
                        "tp": position.tp,  # Keep original TP
                        "magic": 234000
                    }
                    
                    result = mt5.order_send(request)
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        self.logger.error(f"Failed to update trailing stop: {result.comment}")
                        
    def calculate_average_level(self):
        """Calculate the average price level"""
        if not self.price_levels:
            return None
        return sum(self.price_levels) / len(self.price_levels)
        
    def add_price_level(self, price):
        """Add a new price level and maintain max size"""
        self.price_levels.append(price)
        if len(self.price_levels) > self.max_levels:
            self.price_levels.pop(0)
            
    def is_near_level(self, current_price):
        """Check if price is near any tracked level"""
        for level in self.price_levels:
            if abs(current_price - level) < self.level_threshold:
                return True
        return False
        
    def check_trading_conditions(self, df):
        """Check if conditions are met for trading"""
        if len(df) < 2:
            return False, "Not enough data"
            
        current_price = df.iloc[-1]['close']
        
        # Get indicators
        supertrend = df.iloc[-1]['Supertrend']
        psar = df.iloc[-1]['PSARl_0.02_0.2']
        adx = df.iloc[-1]['ADX_14']
        volume_ratio = df.iloc[-1]['tick_volume'] / df['tick_volume'].rolling(20).mean().iloc[-1]
        
        # Initialize conditions list
        conditions = []
        
        # Check ADX (trend strength)
        if adx < 25:
            conditions.append("ADX too low: {:.1f} < 25".format(adx))
            
        # Check volume
        if volume_ratio < 0.8:
            conditions.append("Volume too low: {:.2f}x < 0.8x".format(volume_ratio))
            
        # Mean Reversion Check
        avg_level = self.calculate_average_level()
        if avg_level is not None:
            self.logger.info(f"Average price level: ${avg_level:.2f}")
            self.logger.info(f"Tracked levels: {[f'${p:.2f}' for p in self.price_levels]}")
            
            # If we're in mean reversion mode
            if self.reversion_active:
                # Price has returned to average, look for trend signals
                if abs(current_price - avg_level) < self.level_threshold:
                    self.reversion_active = False
                    self.logger.info("Price returned to average level, checking trend signals")
                else:
                    conditions.append(f"Waiting for price (${current_price:.2f}) to return to level (${avg_level:.2f})")
            
        # All conditions met
        if not conditions:
            # If we have a strong trend signal
            if supertrend < current_price and psar < current_price:  # Bullish
                signal = "BUY"
                # Add this level if it's not near existing ones
                if not self.is_near_level(current_price):
                    self.add_price_level(current_price)
                    self.logger.info(f"Added new price level: ${current_price:.2f}")
                self.reversion_active = True
                return True, signal
            elif supertrend > current_price and psar > current_price:  # Bearish
                signal = "SELL"
                if not self.is_near_level(current_price):
                    self.add_price_level(current_price)
                    self.logger.info(f"Added new price level: ${current_price:.2f}")
                self.reversion_active = True
                return True, signal
                
        return False, "\n".join(conditions)
        
    def run(self):
        """Main trading loop"""
        self.console.print("[bold green]Starting Trend Scalper...[/bold green]")
        
        while True:
            try:
                # Get market data
                rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 7000)
                if rates is None:
                    self.logger.error("❌ Failed to get market data")
                    time.sleep(5)  # Wait before retry
                    continue
                    
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                # Calculate indicators
                df = self.calculate_indicators(df)
                if df is None:
                    self.logger.error("❌ Failed to calculate indicators")
                    time.sleep(5)  # Wait before retry
                    continue
                
                # Check trailing stops for active trades
                positions = mt5.positions_get(symbol=self.symbol)
                if positions:
                    for position in positions:
                        self.check_trailing_stop(position)
                
                # Check for new signals
                signal, reason = self.get_trading_signal(df)
                if signal:
                    self.execute_trade(signal, df)
                
                # Display status
                self.display_status(df)
                
                # Add a small delay to prevent excessive CPU usage
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ Error in main loop: {str(e)}")
                time.sleep(5)  # Wait before retry

if __name__ == "__main__":
    bot = TrendScalper()
    bot.run()
