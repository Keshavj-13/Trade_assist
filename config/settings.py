"""
Configuration constants for fin_assist service.
"""

import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# File paths
CONFIG_FILE = os.path.join(DATA_DIR, 'config.json')
PORTFOLIO_FILE = os.path.join(DATA_DIR, 'portfolio.json')
SYMBOLS_FILE = os.path.join(DATA_DIR, 'nse_symbols.csv')
NEWS_CACHE_FILE = os.path.join(DATA_DIR, 'news_cache.json')

# Market settings
INTERVAL = '5m'
LOOKBACK = '5d'
TOP_N = 5

MIN_PRICE = 5
MIN_AVG_VOLUME = 50000
MIN_ATR_PCT = 0.2
MAX_ATR_PCT = 8.0

# Load config values if present
NEWS_API_KEY = None
TELEGRAM_BOT_TOKEN = None
TELEGRAM_CHAT_ID = None
try:
	if os.path.exists(CONFIG_FILE):
		with open(CONFIG_FILE, 'r') as f:
			cfg = json.load(f)
			NEWS_API_KEY = cfg.get('NEWS_API_KEY')
			TELEGRAM_BOT_TOKEN = cfg.get('TELEGRAM_BOT_TOKEN')
			TELEGRAM_CHAT_ID = cfg.get('TELEGRAM_CHAT_ID')
except Exception:
	pass

# Placeholder for positions loaded from portfolio file
POSITIONS = {}
try:
	if os.path.exists(PORTFOLIO_FILE):
		with open(PORTFOLIO_FILE, 'r') as f:
			p = json.load(f)
			POSITIONS = p.get('positions', {})
except Exception:
	POSITIONS = {}
