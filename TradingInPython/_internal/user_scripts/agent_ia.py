""" Agent IA de trading technique - MLP (Multi-Layer Perceptrons) Keras (backend JAX)
    Indicateurs : RSI, MACD, STOCH
    Sortie  : score de confiance [0, 1]
            > 0.6  : signal BUY
            < 0.4  : signal SELL
            sinon  : HOLD
    
    Ce script s'excute seul, en simulant des data synthétiques.
    Mais il s'excute également par le runner de scripts de l'application TradingInPython, 
    en utilisant les données réelles de l'app via l'API.
"""
import os
os.environ["KERAS_BACKEND"] = "jax" # backend JAX

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append( str( Path(__file__).resolve().parent.parent ) )
    
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from keras import layers, callbacks
from sklearn.preprocessing import MinMaxScaler

from user_scripts.api import api, UserScriptAPI

# ----------------------------------------------
# 1. INDICATEURS TECHNIQUES
# ----------------------------------------------

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename("RSI")


def compute_macd(close: pd.Series,
                 fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return (macd_line.rename("MACD"),
            signal_line.rename("MACD_signal"),
            histogram.rename("MACD_hist"))


def compute_stoch(high: pd.Series, low: pd.Series, close: pd.Series,
                  k_period: int = 14, d_period: int = 3):
    lowest_low   = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k.rename("STOCH_K"), d.rename("STOCH_D")

# -----------------------------------------------------------------------------

def build_features( df: pd.DataFrame ) -> pd.DataFrame:
    """
    Entrée  : DataFrame avec colonnes open, high, low, close, volume
    Sortie  : DataFrame enrichi + colonne 'label' (target pour l'entraînement)
    """
    df = df.copy()
    df.columns = df.columns.str.lower()

    # Indicateurs
    df["RSI"]          = compute_rsi(df["close"])
    macd, macd_sig, macd_hist = compute_macd(df["close"])
    df["MACD"]         = macd
    df["MACD_signal"]  = macd_sig
    df["MACD_hist"]    = macd_hist
    df["STOCH_K"], df["STOCH_D"] = compute_stoch(df["high"], df["low"], df["close"])

    # Features de prix normalisées
    df["returns"]      = df["close"].pct_change()
    df["hl_range"]     = (df["high"] - df["low"]) / df["close"]

    # -- Label : rendement futur sur N bougies (pour l'entraînement) -- #
    # Score 1.0 si le prix monte de + seuil, 0.0 s'il descend, 0.5 sinon
    horizon    = 5      # bougies à l'avance
    threshold  = 0.005  # 0.5 %
    future_ret = df["close"].shift(-horizon) / df["close"] - 1
    df["label"] = 0.5
    df.loc[future_ret >  threshold, "label"] = 1.0
    df.loc[future_ret < -threshold, "label"] = 0.0

    return df.dropna()

# ----------------------------------------------
# 2. PRÉPARATION DES DONNÉES
# ----------------------------------------------

FEATURE_COLS = [
    "RSI", "MACD", "MACD_signal", "MACD_hist",
    "STOCH_K", "STOCH_D", "returns", "hl_range"
]

def prepare_data( df: pd.DataFrame ):
    """Retourne X_train, X_val, X_test, y_train, y_val, y_test + scaler."""
    feat_df = build_features( df )

    X = feat_df[FEATURE_COLS].values
    y = feat_df["label"].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Découpage temporel — pas de shuffle pour respecter l'ordre chronologique
    n        = len(X)
    n_train  = int(n * 0.70)
    n_val    = int(n * 0.15)

    X_train, y_train = X[:n_train],          y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],    y[n_train+n_val:]

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

# ----------------------------------------------
# 3. MODÈLE MLP (Keras / JAX)
# ----------------------------------------------

def build_model( input_dim: int ) -> keras.Model:
    """
    MLP : 3 couches Dense + Dropout + BatchNormalization
    Sortie : sigmoid → score ∈ [0, 1]
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(32, activation="relu"),
        layers.Dropout(0.1),

        layers.Dense(1, activation="sigmoid"),   # score [0, 1]
    ], name="trading_mlp")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["mae"]
    )
    return model

# ----------------------------------------------
# 4. ENTRAÎNEMENT
# ----------------------------------------------

def train( df: pd.DataFrame, epochs: int = 100, batch_size: int = 64 ):
    """
    Lance l'entraînement complet à partir d'un DataFrame OHLCV brut.
    Retourne le modèle entraîné et le scaler.
    """
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(df)

    model = build_model(input_dim=X_train.shape[1])
    model.summary()

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=6, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=1,
    )

    # Évaluation finale sur le jeu de test
    test_loss, test_mae = model.evaluate( X_test, y_test, verbose=0 )
    print(f"\n[Test] loss={test_loss:.4f}  MAE={test_mae:.4f}")

    return model, scaler, history

# ----------------------------------------------
# 5. INFÉRENCE — SIGNAL EN TEMPS RÉEL
# ----------------------------------------------

BUY_THRESHOLD  = 0.60
SELL_THRESHOLD = 0.40

def predict_signal(model: keras.Model,
                   scaler: MinMaxScaler,
                   df: pd.DataFrame) -> dict:
    """
    Prend le dernier état du DataFrame (1 bougie) et retourne :
      {
        'score'  : float [0, 1],
        'signal' : 'BUY' | 'HOLD' | 'SELL',
        'features': dict des indicateurs calculés
      }
    """
    feat_df = build_features(df)
    if feat_df.empty:
        return {"score": 0.5, "signal": "HOLD", "features": {}}

    last_row = feat_df[FEATURE_COLS].iloc[[-1]].values
    last_row_scaled = scaler.transform(last_row)

    score = float(model.predict(last_row_scaled, verbose=0)[0][0])

    if score > BUY_THRESHOLD:
        signal = "BUY"
    elif score < SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    features = feat_df[FEATURE_COLS].iloc[-1].to_dict()

    return {"score": round(score, 4), "signal": signal, "features": features}

# -----------------------------------------------------------------------------

def plot_history( history ):
    fig, (ax1, ax2) = plt.subplots( 1, 2, figsize=(12, 4) )

    # Loss
    ax1.plot(history.history["loss"],     label="train")
    ax1.plot(history.history["val_loss"], label="val")
    ax1.set_title("Loss")
    ax1.legend()

    # MAE Mean Absolute Error - erreur absolue moyenne.
    ax2.plot(history.history["mae"],     label="train")
    ax2.plot(history.history["val_mae"], label="val")
    ax2.set_title("MAE")
    ax2.legend()

    plt.tight_layout()
    #plt.savefig("training_history.png")
    plt.show()

# -----------------------------------------------------------------------------
    
def main():
        
    # Dataframe from API
    data = api.df.copy()
    if data.empty:
        print("Dataframe : aucune donnée disponible.")
        return

    # -- Entraînement -- #
    model, scaler, history = train( data, epochs=80, batch_size=32 )

    # -- Signal sur les dernières données -- #
    result = predict_signal( model, scaler, data )

    print("\n-- Réglages --")
    print(f"  Seuil d'achat  : {BUY_THRESHOLD}")
    print(f"  Seuil de vente : {SELL_THRESHOLD}")

    print(f"\n-- Signal de trading pour : {api.name} --")
    print(f"  Score    : {result['score']}")
    print(f"  Signal   : {result['signal']}")
    print(f"  RSI      : {result['features'].get('RSI', 'N/A'):.2f}")
    print(f"  MACD     : {result['features'].get('MACD', 'N/A'):.4f}")
    print(f"  STOCH_K  : {result['features'].get('STOCH_K', 'N/A'):.2f}")

    # -- Sauvegarde -- #
    #model.save( "trading_mlp.keras" )
    #print("\nModèle sauvegardé : trading_mlp.keras")

    # --- Validation de la qualité du modèle avant sauvegarde --- #
    final_val_loss = history.history["val_loss"][-1]
    best_val_loss  = min(history.history["val_loss"])
    epochs_run     = len(history.history["loss"])

    print("\n-- Validation du modèle --")
    print(f"Epochs effectuées : {epochs_run}")
    print(f"Meilleure val_loss : {best_val_loss:.4f}")

    if best_val_loss < 0.4: # seuil à calibrer selon les données
        #save_agent( model, scaler )
        print("Modèle sauvegardé : qualité suffisante")
    else:
        print("Modèle rejeté : val_loss trop élevée, réentraîner le modele avec plus de données ou ajuster les hyperparamètres.")
    
    #plot_history( history )

# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # Remplacer par l'injection réelle par des data synthétiques
    np.random.seed(42)
    n = 800
    dates  = pd.date_range("2022-01-01", periods=n, freq="1D")
    close  = 100 + np.cumsum(np.random.randn(n) * 0.8)
    high   = close + np.abs(np.random.randn(n) * 0.5)
    low    = close - np.abs(np.random.randn(n) * 0.5)
    open_  = close + np.random.randn(n) * 0.3
    volume = np.random.randint(100_000, 1_000_000, n).astype(float)
 
    demo_df = pd.DataFrame({
        "date": dates, "open": open_, "high": high,
        "low": low, "close": close, "volume": volume
    }).set_index("date")

    api = UserScriptAPI()
    api.update( 
        symbol='Données synthétiques',
        df=demo_df
    )
                
    main()