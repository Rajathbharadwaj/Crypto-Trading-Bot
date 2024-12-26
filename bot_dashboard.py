import streamlit as st
import pandas as pd
from datetime import datetime
import time
import MetaTrader5 as mt5
from new_trading_bot import GoldTradingBot

class DashboardBot(GoldTradingBot):
    def __init__(self, timeframe=mt5.TIMEFRAME_M5):
        super().__init__(timeframe=timeframe)
        
    def update_dashboard(self, df, signal_info):
        if df is None or df.empty:
            st.error("No market data available")
            return
            
        # Clear previous state if needed
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
            
        # Update main status
        st.subheader("Bot Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
            st.metric("Active Trades", len(self.get_active_trades()))
        with col2:
            st.metric("Current Direction", 
                     "Bullish" if df.iloc[-1]['supertrend_direction'] == 1 else "Bearish")
            st.metric("Volatility", f"{df.iloc[-1]['volatility_switch']:.3f}")
            
        # Show last candles
        st.subheader("Last 2 Candles")
        if len(df) >= 2:
            candle_data = []
            for i in [-2, -1]:
                row = df.iloc[i]
                candle_data.append({
                    "Time": row['time'],
                    "Direction": "Bullish" if row['supertrend_direction'] == 1 else "Bearish",
                    "Close": row['close'],
                    "Volatility": f"{row['volatility_switch']:.3f}"
                })
            st.table(pd.DataFrame(candle_data))
        else:
            st.warning("Not enough candles available")
        
        # Show signal status
        st.subheader("Signal Status")
        if hasattr(self, 'transition_detected') and self.transition_detected:
            st.info(f"Transition detected, waiting for new candle confirmation. Direction: {'Bullish' if self.transition_direction == 1 else 'Bearish'}")
        
        # Show active trades
        active_trades = self.get_active_trades()
        if active_trades:
            st.subheader("Active Trade")
            trade = active_trades[0]
            trade_data = {
                "Type": trade['type'],
                "Entry": f"${trade['open_price']:.2f}",
                "Current": f"${trade['current_price']:.2f}",
                "Profit": f"${trade['profit']:.2f}"
            }
            st.table(pd.DataFrame([trade_data]))
            
        # Show recent signals
        if signal_info:
            st.subheader("Signal Information")
            st.write(signal_info)
            
    def run(self):
        st.title("Gold Trading Bot Dashboard")
        
        if st.button("Stop Bot"):
            st.session_state.running = False
            return
            
        placeholder = st.empty()
        
        while True:
            try:
                with placeholder.container():
                    # Get market data
                    df = self.get_rates_df(5000)
                    if df is None or df.empty:
                        st.error("Unable to fetch market data. Retrying...")
                        time.sleep(5)
                        continue
                        
                    # Calculate indicators
                    df = self.calculate_supertrend(df)
                    df = self.calculate_volatility(df)
                    
                    # Check exit conditions
                    should_exit, exit_reason = self.check_exit_conditions(df)
                    signal_info = None
                    
                    if should_exit:
                        active_trades = self.get_active_trades()
                        if active_trades:
                            self.close_trade(active_trades[0], exit_reason)
                            signal_info = f"Exiting trade: {exit_reason}"
                    
                    # If no active trade, check for entry signals
                    elif not self.trade_taken:
                        signal, reason = self.get_trading_signal(df)
                        if signal:
                            if self.execute_trade(signal, df):
                                signal_info = f"{signal} signal executed: {reason}"
                    
                    # Update dashboard
                    self.update_dashboard(df, signal_info)
                
                time.sleep(1)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                time.sleep(5)

if __name__ == "__main__":
    st.set_page_config(page_title="Gold Trading Bot", layout="wide")
    
    if 'running' not in st.session_state:
        st.session_state.running = True
        
    bot = DashboardBot()
    bot.run() 