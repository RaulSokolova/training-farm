# directory
from __future__ import annotations

DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"
INTERM_RESULTS = "interm_results"

# date format: '%Y-%m-%d'
TRAIN_START_DATE = "2018-01-01"  # bug fix: set Monday right, start date set 2014-01-01 ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 1658 and the array at index 1 has size 1657
TRAIN_END_DATE = "2023-06-30"

TEST_START_DATE = "2023-07-01"
TEST_END_DATE = "2023-12-19"

TRADE_START_DATE = "2023-07-01"
TRADE_END_DATE = "2023-12-19"

# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names


# INDICATORS = [
#     "atr_14", "atr_7", "tr", "mfi_14", "rsi_12", 
#     "close_150_ema", "close_9_ema", "supertrend", 
#     "supertrend_ub", "supertrend_lb", 
#     #"ichimoku_7,22,44", 
#     "kdjk", "kdjd", "kdjj", 
#     "tema", "middle_10_tema", "ao", "aroon_14", 
#     "wr_6", "stochrsi_6", "stochrsi", "cci_6", "wt1", 
#     #"close_10_roc",
#     #"close_10_mad", 
#     "trix", 
#     "middle_10_trix", "vr", "vr_6", "middle", 
#     "close_7_mvar", 
#     #"rsv", 
#     "boll", "boll_ub", "boll_lb", 
#     "macd", "macds", "macdh", "cr", "cr-ma1", "cr-ma2", 
#     "cr-ma3", "dma", "pdi", "mdi", "dx", "adx", "adxr", 
#     "trix_9_sma", "mfi", "vwma", "chop", 
#     #"ker", 
#     #"kama", 
#     "ppo", "wt2", 
#     #"zlema", 
#     #"mad", 
#     #"roc", 
#     #"cti", 
#     #"lrma", 
#     #"eri", 
#     #"ftr", 
#     #"rvgi", 
#     #"inertia", 
#     #"kst", 
#     #"pgo", 
#     #"pvo", 
#     #"qqe"
# ]


# INDICATORS = [
#     "atr_14",
#     "atr_7",
#     "tr",
#     "mfi_14",
#     "rsi_12",
#     "close_150_ema",
#     "close_9_ema",
#     "supertrend",
#     "supertrend_ub",
#     "supertrend_lb",
#     # "ichimoku",
#     # "psl",
#     "kdjk",
#     "kdjd",
#     "kdjj",
#     "tema",
#     # "coppock",
#     # "bop",
#     "ao",
#     "aroon_14",
#     # "close_10,2,30_kama",
#     # "high_5_ker",
#     "wr_6",
#     "stochrsi_6",
#     "cci_6",
# ]

INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]


# Model Parameters
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,  # bug fix:KeyError: 'eval_times' line 68, in get_model model.eval_times = model_kwargs["eval_times"]
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}


# Possible time zones
TIME_ZONE_SHANGHAI = "Asia/Shanghai"  # Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = "US/Eastern"  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = "Europe/Paris"  # CAC,
TIME_ZONE_BERLIN = "Europe/Berlin"  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = "Asia/Jakarta"  # LQ45
TIME_ZONE_SELFDEFINED = "xxx"  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 0  # 0 (default) or 1 (use the self defined)

# parameters for data sources
ALPACA_API_KEY = "PKGGZ6HG80DSM67V5J4Q"  # your ALPACA_API_KEY
ALPACA_API_SECRET = "UE9Phq2ezmifdgEfdYNqNXJw556y0auDE1cBDVu7"  # your ALPACA_API_SECRET
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"  # alpaca url
ALPACA_CRYPTO_API_BASE_URL = "https://paper-api.alpaca.markets"  # alpaca url
BINANCE_BASE_URL = "https://data.binance.vision/"  # binance url

DATA_API_KEY = "PKGGZ6HG80DSM67V5J4Q"
DATA_API_SECRET = "UE9Phq2ezmifdgEfdYNqNXJw556y0auDE1cBDVu7"
DATA_API_BASE_URL = "wss://data.alpaca.markets"
TRADING_API_KEY = "PKGGZ6HG80DSM67V5J4Q"
TRADING_API_SECRET = "UE9Phq2ezmifdgEfdYNqNXJw556y0auDE1cBDVu7"
TRADING_API_BASE_URL = "https://paper-api.alpaca.markets"
