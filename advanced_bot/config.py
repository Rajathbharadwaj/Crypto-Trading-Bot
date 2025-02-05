class Config:
    # Exchange API Keys (to be filled)
    BINANCE_API_KEY = ""
    BINANCE_SECRET_KEY = ""
    COINBASE_API_KEY = ""
    COINBASE_SECRET_KEY = ""
    BYBIT_API_KEY = ""
    BYBIT_SECRET_KEY = ""

    # Trading Parameters
    FLASH_ARBITRAGE = {
        'min_spread': 0.0015,  # 0.15%
        'max_position': 0.1,   # BTC
        'execution_timeout': 0.5,  # seconds
        'exchanges': ['binance', 'coinbase', 'bybit']
    }

    LIQUIDITY_PROVISION = {
        'spread': 0.0005,  # 0.05%
        'order_size': 0.05,  # BTC
        'max_open_orders': 10,
        'min_profit': 50  # USD
    }

    NEWS_TRADING = {
        'keywords': ['listing', 'partnership', 'acquisition', 'upgrade'],
        'influencers': ['cz_binance', 'SBF_FTX', 'elonmusk'],
        'min_followers': 100000,
        'reaction_time': 0.1  # seconds
    }

    ORDER_BOOK = {
        'min_wall_size': 100,  # BTC
        'wall_distance': 0.002,  # 0.2%
        'max_position': 0.2,  # BTC
        'profit_target': 0.002  # 0.2%
    }

    # Risk Management
    MAX_DAILY_LOSS = 1000  # USD
    MAX_POSITION_SIZE = 1  # BTC
    MAX_OPEN_TRADES = 5
    STOP_LOSS_PCT = 0.002  # 0.2%

    # Notification Settings
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""

    # Database
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
