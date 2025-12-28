import requests  # Import requests to send HTTP requests to Telegram
import logging

# Add your Telegram Bot Token and Chat ID here
TELEGRAM_BOT_TOKEN = '8116738142:AAHAR84spukz3PJVPM6tk5jv94WkDeOJO_U'
TELEGRAM_CHAT_ID = '928383272'


def send_telegram_message(message):
    """
    Send a message to the Telegram group using the Telegram Bot API.
    Args:
        message (str): The message content to send.
    """
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            logging.info("Message sent to Telegram successfully.")
        else:
            logging.error(f"Failed to send message to Telegram. Response: {
                          response.json()}")
    except Exception as e:
        logging.error(f"Unexpected error while sending Telegram message: {e}")
        print(f"Error sending Telegram message: {e}")

