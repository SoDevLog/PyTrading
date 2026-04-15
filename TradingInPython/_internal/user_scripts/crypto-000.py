import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================
TICKERS = [
    "BTC-USD",  # Bitcoin
    "ETH-USD",  # Ethereum
    "BNB-USD",  # Binance Coin
    "SOL-USD",  # Solana
    "ADA-USD",  # Cardano
    "XRP-USD"   # Ripple
]

PERIOD = "1y"
INTERVAL = "1d"

# =========================
# DOWNLOAD DATA
# =========================
data = yf.download(
    TICKERS,
    period="1y",
    interval="1d",
    auto_adjust=False
)

# On récupère uniquement les clôtures
close = data["Close"]

# =========================
# NORMALISATION (base 100)
# =========================
normalized = close / close.iloc[0] * 100

# =========================
# PLOT
# =========================
plt.figure(figsize=(12, 6))

for col in normalized.columns:
    plt.plot(normalized.index, normalized[col], label=col)

plt.title("Comparaison des principales cryptomonnaies (Base 100)")
plt.xlabel("Date")
plt.ylabel("Performance (%)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()