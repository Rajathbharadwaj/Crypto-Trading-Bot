from fasthtml.common import *
from datetime import datetime
from new_trading_bot import GoldTradingBot
import MetaTrader5 as mt5
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the trading bot
bot = GoldTradingBot()

# Global variable to store latest data
latest_data = {
    "market_status": "Initializing...",
    "current_price": 0,
    "volatility": 0,
    "supertrend": 0,
    "active_trades": [],
    "trade_history": [],
    "last_update": datetime.now()
}

def update_data():
    """Background thread to update trading data"""
    global latest_data
    while True:
        try:
            logger.info("Fetching new data...")
            
            # Get current market data
            df = bot.get_rates_df(100)
            if df is not None:
                logger.info(f"Got rates data, shape: {df.shape}")
                df = bot.calculate_volatility(df)
                df = bot.calculate_supertrend(df)
                last_row = df.iloc[-1]
                
                # Update market status
                latest_data["market_status"] = "Open" if bot.is_market_open() else "Closed"
                latest_data["current_price"] = last_row['close']
                latest_data["volatility"] = last_row['volatility_switch']
                latest_data["supertrend"] = last_row['supertrend_direction']
                
                # Get active trades
                active_trades = bot.get_active_trades()
                logger.info(f"Active trades: {len(active_trades) if active_trades else 0}")
                if active_trades:
                    latest_data["active_trades"] = active_trades
                
                latest_data["last_update"] = datetime.now()
                logger.info("Data update complete")
            else:
                logger.warning("Failed to get rates data")
            
            time.sleep(1)  # Update every second
            
        except Exception as e:
            logger.error(f"Error updating data: {e}", exc_info=True)
            time.sleep(5)

# Create the thread object but don't start it yet
update_thread = threading.Thread(target=update_data, daemon=True)

app, rt = fast_app(
    hdrs=(
        Script(src="https://cdn.plot.ly/plotly-2.32.0.min.js"),
        Style("""
            .dashboard { padding: 20px; }
            .card {
                border: 1px solid #ddd;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                background: white;
            }
            .market-status { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
            .buy { border-left: 4px solid #4CAF50; }
            .sell { border-left: 4px solid #f44336; }
            .indicator { font-weight: bold; }
            .positive { color: #4CAF50; }
            .negative { color: #f44336; }
        """)
    )
)

def MarketStatus():
    """Create market status card"""
    return Div(cls="card market-status", content=[
        Div([
            H3("Market Status"),
            P(latest_data["market_status"], cls="indicator")
        ]),
        Div([
            H3("Current Price"),
            P(f"${latest_data['current_price']:.2f}", cls="indicator")
        ]),
        Div([
            H3("Volatility Switch"),
            P(f"{latest_data['volatility']:.3f}", 
              cls=f"indicator {'positive' if latest_data['volatility'] > 0.5 else ''}")
        ]),
        Div([
            H3("Supertrend"),
            P("Bullish" if latest_data['supertrend'] == 1 else "Bearish", 
              cls=f"indicator {'positive' if latest_data['supertrend'] == 1 else 'negative'}")
        ])
    ])

def TradeCard(trade):
    """Create a trade card"""
    return Div(
        cls=f"card {'buy' if trade['type']=='BUY' else 'sell'}", 
        content=[
            H3(f"{trade['type']} Trade"),
            Div([
                P(f"Entry: ${trade['open_price']:.2f}"),
                P(f"Current: ${trade['current_price']:.2f}"),
                P(f"SL: ${trade['sl']:.2f}"),
                P(f"TP: ${trade['tp']:.2f}"),
                P(f"Profit: ${trade['profit']:.2f} ({trade['profit_pips']:.1f} pips)",
                  cls="positive" if trade['profit'] > 0 else "negative")
            ])
        ]
    )

@rt("/")
def get():
    return Titled("Gold Trading Dashboard",
        Main(cls="dashboard", content=[
            H1("Gold Trading Bot Dashboard"),
            
            # Debug Information
            Section([
                H2("Debug Information"),
                P(f"Bot Status: {'Connected' if bot else 'Not Connected'}"),
                P(f"Last Update Attempt: {latest_data['last_update'].strftime('%Y-%m-%d %H:%M:%S')}"),
                P(f"Market Status: {latest_data['market_status']}"),
                Pre(f"Latest Data: {str(latest_data)}")
            ]),
            
            # Market Status
            Section([
                H2("Market Status"),
                P(f"Last Update: {latest_data['last_update'].strftime('%Y-%m-%d %H:%M:%S')}"),
                MarketStatus()
            ]),
            
            # Active Trades
            Section([
                H2("Active Trades"),
                *([TradeCard(trade) for trade in latest_data["active_trades"]]
                  if latest_data["active_trades"] 
                  else [P("No active trades")])
            ]),
            
            # Auto-refresh script
            Script("""
                setTimeout(function() {
                    window.location.reload();
                }, 5000);  // Changed to 5 seconds for less frequent updates
            """)
        ])
    )
if __name__ == "__main__":
    logger.info("Starting trading dashboard...")
    logger.info("Initializing trading bot...")
    logger.info("Starting update thread...")
    update_thread.start()
    logger.info("Starting web server...")
    serve() 