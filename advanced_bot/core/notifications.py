import asyncio
from typing import Dict, List
import logging
import telegram
from datetime import datetime

class NotificationManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.bot = telegram.Bot(token=self.config.TELEGRAM_BOT_TOKEN)
        self.chat_id = self.config.TELEGRAM_CHAT_ID
        self.notification_queue = asyncio.Queue()
        self.last_notification = {}

    async def send_notification(self, message: str, priority: str = 'normal'):
        """Add notification to queue"""
        await self.notification_queue.put({
            'message': message,
            'priority': priority,
            'timestamp': datetime.now()
        })

    async def send_trade_notification(self, trade_data: Dict):
        """Send formatted trade notification"""
        emoji = '🟢' if trade_data.get('profit', 0) > 0 else '🔴'
        
        message = (
            f"{emoji} Trade Completed\n"
            f"Strategy: {trade_data['strategy']}\n"
            f"Profit: ${trade_data['profit']:.2f}\n"
            f"Entry: ${trade_data['entry_price']:.2f}\n"
            f"Exit: ${trade_data['exit_price']:.2f}\n"
            f"Size: {trade_data['size']} BTC"
        )
        
        await self.send_notification(message, 'normal')

    async def send_risk_alert(self, risk_data: Dict):
        """Send risk management alert"""
        message = (
            "⚠️ Risk Alert ⚠️\n"
            f"Risk Score: {risk_data['risk_score']:.1f}/100\n"
            f"Total Exposure: ${risk_data['total_exposure']:,.2f}\n"
            f"Daily Loss: ${abs(risk_data['daily_loss']):,.2f}\n"
            "Taking risk reduction measures..."
        )
        
        await self.send_notification(message, 'high')

    async def send_performance_update(self, stats: Dict):
        """Send periodic performance update"""
        message = (
            "📊 Performance Update\n"
            f"Total Profit: ${stats['total_profit']:,.2f}\n"
            f"Win Rate: {stats['win_rate']*100:.1f}%\n"
            f"Active Trades: {stats['active_trades']}\n"
            "\nBy Strategy:\n"
        )
        
        for strategy, profit in stats['strategy_profits'].items():
            message += f"{strategy}: ${profit:,.2f}\n"
        
        await self.send_notification(message, 'normal')

    async def send_opportunity_alert(self, opportunity: Dict):
        """Send trading opportunity alert"""
        message = (
            "🎯 Trading Opportunity\n"
            f"Type: {opportunity['type']}\n"
            f"Strategy: {opportunity['strategy']}\n"
            f"Potential Profit: ${opportunity['potential_profit']:,.2f}\n"
            f"Confidence: {opportunity['confidence']*100:.1f}%"
        )
        
        await self.send_notification(message, 'normal')

    async def process_notification_queue(self):
        """Process and send notifications from queue"""
        while True:
            try:
                notification = await self.notification_queue.get()
                
                # Rate limiting by notification type
                current_time = datetime.now()
                last_time = self.last_notification.get(notification['priority'])
                
                if last_time:
                    # High priority: 1 minute delay
                    # Normal priority: 5 minute delay
                    min_delay = timedelta(minutes=1 if notification['priority'] == 'high' else 5)
                    
                    if current_time - last_time < min_delay:
                        continue
                
                # Send notification
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=notification['message'],
                    parse_mode=telegram.ParseMode.MARKDOWN
                )
                
                # Update last notification time
                self.last_notification[notification['priority']] = current_time
                
            except Exception as e:
                self.logger.error(f"Error sending notification: {str(e)}")
                await asyncio.sleep(5)
            
            await asyncio.sleep(0.1)  # Prevent spam

    async def run(self):
        """Start notification processing"""
        try:
            # Send startup notification
            await self.send_notification("🚀 Trading Bot Started", 'high')
            
            # Start processing queue
            await self.process_notification_queue()
            
        except Exception as e:
            self.logger.error(f"Error in notification manager: {str(e)}")
            raise
