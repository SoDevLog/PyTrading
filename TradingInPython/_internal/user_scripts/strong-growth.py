
""" Filtre les actions en forte croissance sur n jours.
"""
import yfinance
import pandas as pd
from datetime import datetime, timedelta

if __name__ == "__main__":
    import sys
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    sys.path.append( str(base_dir) )
    from user_scripts.api import UserScriptAPI
    
from user_scripts.api import api

# ------------------------------------------------------------------------------

NB_DAYS = 10
PRICE_THRESHOLD = 0.05
VOLUME_MULTIPLIER = 1.2

def filter_strong_growth(
        tickers: list[str], 
        nb_days: int = 20,
        price_threshold: float = 0.10,
        volume_multiplier: float = 1.5
    ) -> pd.DataFrame:
    """
    Filtre les actions en forte croissance sur n jours.

    Critères combinés :
      - Performance prix >= price_threshold (défaut : +10%)
      - Volume moyen sur n jours >= volume_multiplier x volume moyen long terme (défaut : 1.5x)

    Args:
        tickers          : liste de symboles boursiers
        n_days           : fenêtre d'analyse en jours calendaires
        price_threshold  : seuil de hausse prix (0.10 = +10%)
        volume_multiplier: ratio volume récent / volume long terme

    Returns:
        DataFrame trié par performance décroissante, avec les actions qualifiées.
    """
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=nb_days)
    # Historique long terme pour calculer le volume de référence (1 an)
    lt_start   = end_date - timedelta(days=365)

    results = []

    for ticker in tickers:
        try:
            _ticker = yfinance.Ticker( ticker )
            _stock_info = _ticker.get_info()
            _short_name = _stock_info.get( 'shortName' )
            _result = f"{_short_name} ({ticker})"
            
            # Données long terme (volume de référence)
            df_lt = _ticker.history( start=lt_start, end=end_date )
            if df_lt.empty or len( df_lt ) < nb_days:
                print(f"[{ticker}] Données insuffisantes, ignoré.")
                continue

            # Fenêtre récente
            df_recent = df_lt.iloc[ -nb_days: ]  # dernières n sessions ouvrées

            # --- Critère 1 : performance prix --- #
            price_start = df_recent["Close"].iloc[0]
            price_end   = df_recent["Close"].iloc[-1]
            perf        = (price_end - price_start) / price_start

            # --- Critère 2 : volume anormal --- #
            avg_vol_recent = df_recent["Volume"].mean()
            avg_vol_lt     = df_lt["Volume"].mean()  # référence long terme
            vol_ratio      = avg_vol_recent / avg_vol_lt if avg_vol_lt > 0 else 0

            _result += f" : {perf:+.2%} | {vol_ratio:.2f}"

            # --- Filtre combiné --- #
            if perf >= price_threshold and vol_ratio >= volume_multiplier:
                results.append({
                    "Ticker":        ticker,
                    "Prix départ":   round(price_start, 2),
                    "Prix actuel":   round(price_end, 2),
                    "Performance":   f"{perf:+.2%}",
                    "Perf (float)":  perf,
                    "Vol ratio":     round(vol_ratio, 2),
                    "Vol moyen (n)": int(avg_vol_recent),
                    "Vol moyen LT":  int(avg_vol_lt),
                })
                _result += " -> OK"

            print( _result )
        except Exception as e:
            print(f"[{ticker}] Erreur : {e}")

    if not results:
        print("Aucune action ne remplit les critères.")
        return pd.DataFrame()

    df_out = pd.DataFrame(results).sort_values( "Perf (float)", ascending=False )
    df_out = df_out.drop(columns=["Perf (float)"])
    df_out = df_out.reset_index(drop=True)
    return df_out

# -----------------------------------------------------------------------------

def main():

    # Check parameters for the script
    print( "--- Check API's parameters ---" )
    if not api.check_parameters( ['tickers'] ):
        exit(1)

    tickers = api.tickers

    print( "--- Filtre des actions à Croissance Forte ---" )
    print( f"Fenêtre : {NB_DAYS} jours | Prix ≥ +{PRICE_THRESHOLD*100:.0f}% | Volume ≥ {VOLUME_MULTIPLIER} x la moyenne long terme\n" )
        
    df = filter_strong_growth(
        tickers,
        nb_days=NB_DAYS,
        price_threshold=PRICE_THRESHOLD,
        volume_multiplier=VOLUME_MULTIPLIER
    )

    if not df.empty:
        print(df.to_string(index=False))

if __name__ == "__main__":

    api_context = {
        'tickers': [ 'HO.PA', 'AM.PA', 'AIR.PA', 'GE', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'IT' ],
    }
        
    api = UserScriptAPI()
    api.update( **api_context )
    
    main()