"""Détecte les épisodes de capitulation vendeuse et suit leur confirmation.

    Le scanner distingue trois phases :
    1. Capitulation potentielle : baisse préalable avec nouveau plus-bas, volume extrême et amplitude
        anormale ; les séances de publication de résultats sont écartées.
    2. En confirmation : un rebond / une stabilisation apparaît après le pic vendeur.
    3. Retournement confirmé : le cours reprend une dynamique haussière et les vendeurs
        ne parviennent plus à enfoncer le creux. Un test du creux sur volume réduit renforce
        le signal, mais n'est pas obligatoire pour afficher une confirmation technique.
        La confirmation exige au moins 5 séances après le pic et un volume qui la soutient
        (volume haussier ou test réussi).

    Les seuils sont des paramètres de départ à valider par backtest. Aucun signal ne garantit
    que le point bas définitif est atteint.
    
"""
import yfinance
import numpy
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Circle
from dataclasses import dataclass, field
from datetime import datetime

run_standalone = (__name__ == "__main__")

if run_standalone:
    import sys
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    sys.path.append( str(base_dir) )
    from user_scripts.api import UserScriptAPI

import settings.default as settings
from user_scripts.api import api
from styles.watermark import Watermark
from digitsignalprocessing.indicators import rsi_welles_wilder, macd

# ------------------------------------------------------------------------------
# Paramètrage de l'algorithme
# ------------------------------------------------------------------------------

# Historique : 2 ans permettent à la fois de calculer les références et d'évaluer
# les résultats historiques sur les 5, 10 et 20 séances suivant un événement.
HISTORY_PERIOD          = "2y"
MIN_BARS                = 100
EXCLUDE_LIVE_BAR        = True     # retire la séance du jour tant que la bourse n'a pas clôturé (bougie partielle)
SESSION_CLOSE_HOUR      = 18       # heure locale de la place, fuseau de l'index, après laquelle la séance du jour est complète

# C1 : baisse préalable
DROP_LOOKBACK           = 60       # plus haut recherché dans les 60 séances précédentes
MIN_DROP                = 0.15     # baisse minimale du plus haut au plus bas de la bougie événementielle
SMA_PERIOD              = 50
NEW_LOW_LOOKBACK        = 20       # le creux de l'événement doit être un plus-bas des 20 séances précédentes

# Résultats : un gap sur publication est une information, pas une capitulation
EXCLUDE_EARNINGS_EVENTS = True     # écarte les événements tombant sur la séance des résultats ou la suivante
EARNINGS_LIMIT          = 16       # nombre de dates de résultats demandées (~4 ans) ; à augmenter si HISTORY_PERIOD augmente

# C2 : volume exceptionnel
VOL_REF                 = 20       # référence des 20 séances précédentes, hors séance événementielle
VOLUME_MULTIPLIER       = 2.5      # volume de l'événement >= 2,5 x la médiane
VOL_PERCENTILE_LOOKBACK = 60       # ou volume dans les 5 % les plus élevés des 60 séances précédentes
VOL_PERCENTILE          = 0.95

# C3 : choc de volatilité
ATR_PERIOD              = 20
MIN_ATR_MULTIPLIER      = 1.5      # True Range événementiel >= 1,5 x ATR de référence

# C4 / C5 : indices de capitulation
RSI_PERIOD              = 14
RSI_OVERSOLD            = 30
RSI_EVENT_WINDOW        = 3        # survente le jour de l'événement ou dans les 2 séances précédentes
CLOSE_LOCATION_MIN      = 0.60     # clôture dans les 40 % supérieurs de la bougie
LOWER_WICK_MIN          = 0.35     # ou mèche basse >= 35 % de l'amplitude totale

# Recherche de l'événement le plus récent
MAX_EVENT_AGE           = 20       # ne remonte pas au-delà de 20 séances pour le tableau courant
EVENT_CLUSTER_GAP       = 5        # fusionne les détections rapprochées d'un même épisode

# Confirmation après capitulation
MIN_AUTOMATIC_RALLY     = 0.05     # rebond maximal observé depuis le creux >= 5 %
EMA_TREND               = 10
HL_RECENT               = 3        # plus-bas ascendants : les 3 derniers plus-bas vs ceux des séances précédentes
HIGHER_LOW_ATR          = 0.25     # et au moins 0,25 ATR au-dessus du creux de l'événement
MACD_FAST               = 12
MACD_SLOW               = 26
MACD_SIGNAL             = 9
MACD_RISING             = 3
UPDOWN_WINDOW           = 5
MIN_CONFIRM_SCORE       = 4        # score de confirmation sur 7
MIN_CONFIRM_AGE         = 5        # pas de retournement confirmé avant 5 séances après l'événement

# Test secondaire du creux (Secondary Test, approche Wyckoff)
TEST_LOW_ATR            = 0.5      # autorise un passage intraday jusqu'à 0,5 ATR sous le creux
TEST_HIGH_ATR           = 1.0      # le test doit rester à moins de 1 ATR au-dessus du creux
TEST_VOLUME_RATIO       = 0.65     # volume du test <= 65 % du volume de capitulation
TEST_RANGE_RATIO        = 0.80     # amplitude du test <= 80 % de celle de la séance de capitulation
INVALIDATION_ATR        = 0.25     # invalidation si une clôture passe sous le creux de plus de 0,25 ATR
MAX_CONFIRMATION_AGE    = 12       # signal sans réaction après 12 séances : "SANS SUITE"

# Évaluation historique indicative, ce n'est pas une simulation de stratégie complète
BACKTEST_HORIZONS       = (5, 10, 20)
BACKTEST_HORIZON        = max( BACKTEST_HORIZONS )
BACKTEST_ENABLED        = False

CRITERIA = ["C1", "C2", "C3", "C4", "C5"]
CONFIRMATION_CRITERIA = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]

# ------------------------------------------------------------------------------

_THEME = {
    True: { # --- DARK --- #
        "BG":       "#0F1117",
        "BG_LEGEND":"#0D397A",
        "ROW":      "#171A21",
        "ROW_ALT":  "#151920",
        "BORDER":   "#252B35",
        "TEXT":     "#F4F7FB",
        "MUTED":    "#858E9E",
        "GREEN":    "#00E676",
        "RED":      "#FF4D67",
        "ORANGE":   "#FFB020",
        "BLUE":     "#4DA3FF",
    },
    False: { # --- LIGHT --- #
        "BG":       "#FFFFFF",
        "BG_LEGEND":"#F5F5F5",
        "ROW":      "#F0F0F0",
        "ROW_ALT":  "#DFE4EC",
        "BORDER":   "#173363",
        "TEXT":     "#6b7280",
        "MUTED":    "#858E9E",
        "GREEN":    "#26B6A3",
        "RED":      "#D83C49",
        "ORANGE":   "#FFB020",
        "BLUE":     "#4DA3FF",
    },
}
        
# ------------------------------------------------------------------------------

def _drop_live_bar( data: pd.DataFrame ) -> pd.DataFrame:
    """Retire la dernière bougie si c'est la séance du jour et que la place n'a pas clôturé."""
    if not EXCLUDE_LIVE_BAR or data.empty:
        return data
    now = pd.Timestamp.now( tz=data.index.tz )
    if data.index[-1].date() == now.date() and now.hour < SESSION_CLOSE_HOUR:
        return data.iloc[:-1]
    return data

def _prepare_history( df: pd.DataFrame ) -> pd.DataFrame | None:
    """Nettoie l'historique OHLCV et prépare les références sans fuite de données futures."""
    if df is None or df.empty:
        return None

    data = df.copy()
    if isinstance( data.columns, pd.MultiIndex ):
        return None

    required = ["High", "Low", "Close"]
    if any( column not in data.columns for column in required ):
        return None

    if "Open" not in data.columns:
        data["Open"] = data["Close"].shift( 1 ).fillna( data["Close"] )
    if "Volume" not in data.columns:
        data["Volume"] = 0

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        data[column] = pd.to_numeric( data[column], errors="coerce" )

    data = data.dropna( subset=["Open", "High", "Low", "Close"] )
    data = data.loc[~data.index.duplicated( keep="last" )].sort_index().copy()
    data["Volume"] = data["Volume"].fillna( 0 ).clip( lower=0 )
    data = _drop_live_bar( data )

    if len( data ) < MIN_BARS:
        return None

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]
    prev_close = close.shift( 1 )

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max( axis=1 )

    # Les références sont décalées d'une séance : l'événement ne gonfle pas sa propre moyenne.
    data["_TR"] = true_range
    data["_ATR_REF"] = true_range.rolling( ATR_PERIOD, min_periods=ATR_PERIOD ).mean().shift( 1 )

    # Références volume calculées sur les seules séances précédentes :
    # la médiane sert au critère C2
    data["_VOL_REF_MEDIAN"] = volume.rolling( VOL_REF, min_periods=VOL_REF ).median().shift( 1 )
    data["_VOL_Q95"] = volume.rolling(
        VOL_PERCENTILE_LOOKBACK, min_periods=VOL_PERCENTILE_LOOKBACK
    ).quantile( VOL_PERCENTILE ).shift( 1 )
    data["_TURNOVER_EST"] = close * volume
    data["_TURNOVER_REF_MEDIAN"] = data["_TURNOVER_EST"].rolling(
        VOL_REF, min_periods=VOL_REF
    ).median().shift( 1 )    
    data["_SMA50"] = close.rolling( SMA_PERIOD, min_periods=SMA_PERIOD ).mean()
    data["_RSI"] = rsi_welles_wilder( data, RSI_PERIOD )
    data["_EMA10"] = close.ewm( span=EMA_TREND, adjust=False ).mean()

    # Calcul du MACD
    _, _, hist = macd( data, histo=True, short_window=MACD_FAST, long_window=MACD_SLOW, signal_window=MACD_SIGNAL )
    data["_MACD_HIST"] = hist
    data["_EARNINGS"] = False

    return data

def _earnings_mask( ticker_obj, data: pd.DataFrame ) -> numpy.ndarray:
    """Marque la séance de publication des résultats et la suivante (publication avant ou après clôture)."""
    mask = numpy.zeros( len( data ), dtype=bool )
    if not EXCLUDE_EARNINGS_EVENTS:
        return mask
    try:
        dates = ticker_obj.get_earnings_dates( limit=EARNINGS_LIMIT )
    except Exception as error:
        print( f"[{ticker_obj.ticker}] Dates de résultats indisponibles : {error}" )
        return mask
    if dates is None or dates.empty:
        return mask
    days = data.index.tz_localize( None ).normalize()
    for date in dates.index.tz_localize( None ).normalize():
        if date >= days[0]:
            position = days.searchsorted( date )
            mask[position:position + 2] = True
    return mask

# ------------------------------------------------------------------------------
# Détection d'un pic vendeur (Selling Climax potentiel)
# ------------------------------------------------------------------------------

def _measure_climax_at( data: pd.DataFrame, index: int ) -> dict | None:
    """Mesure les critères d'une séance, même si elle ne qualifie pas comme capitulation."""
    min_index = max( DROP_LOOKBACK, SMA_PERIOD, ATR_PERIOD, VOL_PERCENTILE_LOOKBACK, VOL_REF )
    if index < min_index or index >= len( data ):
        return None

    row = data.iloc[index]
    start = max( 0, index - DROP_LOOKBACK )
    prior = data.iloc[start:index]
    if prior.empty:
        return None

    prior_peak = float( prior["High"].max() )
    event_low = float( row["Low"] )
    new_low = bool( event_low <= data["Low"].iloc[max( 0, index - NEW_LOW_LOOKBACK ):index].min() )
    earnings = bool( row["_EARNINGS"] )
    close = float( row["Close"] )
    event_volume = float( row["Volume"] )
    event_tr = float( row["_TR"] ) if pd.notna( row["_TR"] ) else 0.0
    atr_ref = float( row["_ATR_REF"] ) if pd.notna( row["_ATR_REF"] ) else 0.0
    vol_ref_median = float( row["_VOL_REF_MEDIAN"] ) if pd.notna( row["_VOL_REF_MEDIAN"] ) else 0.0
    vol_q95 = float( row["_VOL_Q95"] ) if pd.notna( row["_VOL_Q95"] ) else 0.0
    turnover_est = close * event_volume
    turnover_ref_median = (
        float( row["_TURNOVER_REF_MEDIAN"] )
        if pd.notna( row["_TURNOVER_REF_MEDIAN"] ) else 0.0
    )
    sma50 = float( row["_SMA50"] ) if pd.notna( row["_SMA50"] ) else numpy.nan
    drop = (prior_peak - event_low) / prior_peak if prior_peak > 0 else 0.0
    # Ratio utilisé par la règle C2 : volume / médiane des séances précédentes.
    vol_ratio = event_volume / vol_ref_median if vol_ref_median > 0 else 0.0
    turnover_ratio = turnover_est / turnover_ref_median if turnover_ref_median > 0 else 0.0
    atr_ratio = event_tr / atr_ref if atr_ref > 0 else 0.0

    candle_range = max( float( row["High"] ) - event_low, 0.0 )
    close_location = (close - event_low) / candle_range if candle_range > 0 else 0.0
    lower_wick = (min( float( row["Open"] ), close ) - event_low) / candle_range if candle_range > 0 else 0.0

    rsi_start = max( 0, index - RSI_EVENT_WINDOW + 1 )
    rsi_min = data["_RSI"].iloc[rsi_start:index + 1].min()
    rsi_extreme = bool( pd.notna( rsi_min ) and rsi_min <= RSI_OVERSOLD )

    c1 = bool( drop >= MIN_DROP and new_low and pd.notna( sma50 ) and close < sma50 )
    c2 = bool(
        vol_ref_median > 0
        and event_volume > 0
        and (vol_ratio >= VOLUME_MULTIPLIER or (vol_q95 > 0 and event_volume >= vol_q95))
    )
    c3 = bool( atr_ref > 0 and atr_ratio >= MIN_ATR_MULTIPLIER )
    c4 = bool( close_location >= CLOSE_LOCATION_MIN or lower_wick >= LOWER_WICK_MIN )
    c5 = rsi_extreme
    flags = [c1, c2, c3, c4, c5]

    return {
        "index": index,
        "date": data.index[index],
        "peak": prior_peak,
        "low": event_low,
        "high": float( row["High"] ),
        "close": close,
        "open": float( row["Open"] ),
        "drop": drop,
        "vol_ratio": vol_ratio,                 # rapport au volume médian : règle C2 active
        "turnover_ratio": turnover_ratio,       # valeur échangée estimée / médiane historique
        "atr_ratio": atr_ratio,
        "rsi_min": float( rsi_min ) if pd.notna( rsi_min ) else numpy.nan,
        "event_volume": event_volume,
        "event_tr": event_tr,
        "atr_ref": atr_ref,
        "close_location": close_location,
        "lower_wick": lower_wick,
        "flags": flags,
        "earnings": earnings,
        "score": int( sum( flags ) ),
        "qualified": bool( c1 and c2 and c3 and not earnings ),
    }


def _evaluate_climax_at( data: pd.DataFrame, index: int ) -> dict | None:
    """Retourne un événement uniquement si les trois critères de base sont validés."""
    event = _measure_climax_at( data, index )
    if event is None or not event["qualified"]:
        return None
    return event

def _find_climax_candidates(
        data: pd.DataFrame,
        start_index: int,
        end_index: int,
    ) -> list[dict]:
    """Retourne les séances qualifiées dans l'intervalle [start_index, end_index)."""
    candidates = []
    start_index = max( start_index, DROP_LOOKBACK, SMA_PERIOD, VOL_PERCENTILE_LOOKBACK, ATR_PERIOD )
    end_index = min( end_index, len( data ) )

    for index in range( start_index, end_index ):
        event = _evaluate_climax_at( data, index )
        if event is not None:
            candidates.append( event )

    return candidates

def _event_strength( event: dict ) -> tuple:
    """Classe les détections proches : score d'abord, puis volume et amplitude."""
    return (
        int( event["score"] ),
        min( float( event["vol_ratio"] ), 20.0 ),
        min( float( event["atr_ratio"] ), 10.0 ),
        int( event["index"] ),
    )

def _cluster_events( events: list[dict] ) -> list[dict]:
    """Fusionne des séances détectées à quelques jours d'intervalle en un seul épisode."""
    if not events:
        return []

    events = sorted( events, key=lambda event: event["index"] )
    clusters = []
    cluster = [events[0]]

    for event in events[1:]:
        if event["index"] - cluster[-1]["index"] <= EVENT_CLUSTER_GAP:
            cluster.append( event )
        else:
            clusters.append( max( cluster, key=_event_strength ) )
            cluster = [event]

    clusters.append( max( cluster, key=_event_strength ) )
    return clusters

# ------------------------------------------------------------------------------
# Confirmation du pic vendeur : rebond automatique, retest et structure de prix
# ------------------------------------------------------------------------------

def _evaluate_confirmation( data: pd.DataFrame, event: dict ) -> dict:
    """Évalue l'évolution postérieure au pic vendeur jusqu'à la dernière séance."""
    event_index = int( event["index"] )
    last_index = len( data ) - 1
    bars_ago = last_index - event_index
    post = data.iloc[event_index + 1:]

    last = data.iloc[-1]
    last_close = float( last["Close"] )
    last_low = float( last["Low"] )
    ema10 = float( last["_EMA10"] ) if pd.notna( last["_EMA10"] ) else numpy.nan
    event_low = float( event["low"] )
    event_close = float( event["close"] )
    event_high = float( event["high"] )
    event_volume = float( event["event_volume"] )
    event_tr = float( event["event_tr"] )
    atr_ref = float( event["atr_ref"] )

    rebound = (last_close - event_low) / event_low if event_low > 0 else 0.0
    rally_high = float( post["High"].max() ) if not post.empty else event_high
    max_rebound = (rally_high - event_low) / event_low if event_low > 0 else 0.0
    automatic_rally = bool( not post.empty and max_rebound >= MIN_AUTOMATIC_RALLY )

    # Une invalidation repose sur les clôtures, pas sur une simple mèche intraday.
    invalidation_level = event_low - INVALIDATION_ATR * atr_ref
    invalidated = bool( not post.empty and (post["Close"] < invalidation_level).any() )

    # Plus-bas ascendants : les plus-bas récents dépassent ceux des séances précédentes depuis
    # l'événement, et restent nettement au-dessus du creux de la capitulation.
    recent_low = post["Low"].iloc[-HL_RECENT:].min()
    higher_lows = bool(
        recent_low > post["Low"].iloc[:-HL_RECENT].min()
        and recent_low >= event_low + HIGHER_LOW_ATR * atr_ref
    )
    above_ema = bool( pd.notna( ema10 ) and last_close > ema10 )
    close_above_event = bool( last_close > event_close )

    hist_diff = data["_MACD_HIST"].diff().iloc[-MACD_RISING:]
    macd_rising = bool( len( hist_diff ) == MACD_RISING and (hist_diff > 0).all() )

    close_changes = data["Close"].diff().iloc[-UPDOWN_WINDOW:]
    recent_volumes = data["Volume"].iloc[-UPDOWN_WINDOW:]
    up_volume = float( recent_volumes[close_changes > 0].sum() )
    down_volume = float( recent_volumes[close_changes < 0].sum() )
    buying_volume = bool( up_volume > down_volume )

    # Test secondaire : après un rebond, le cours revisite la zone du creux avec
    # moins de volume et une amplitude plus faible que lors de la capitulation.
    test_status = "—"
    successful_test = False
    weak_test_seen = False

    if automatic_rally and not post.empty:
        # Le test peut survenir après le premier rebond significatif, même si le cours
        # inscrit ensuite un plus haut : on ne le limite donc pas aux séances suivant
        # le maximum absolu atteint depuis l'événement.
        rally_threshold = event_low * (1 + MIN_AUTOMATIC_RALLY)
        rally_offsets = numpy.flatnonzero( post["High"].to_numpy() >= rally_threshold )
        rally_start_index = event_index + 1 + int( rally_offsets[0] )
        test_bars = data.iloc[rally_start_index + 1:]

        if test_bars.empty:
            test_status = "EN ATTENTE"
        else:
            for test_index in range( rally_start_index + 1, len( data ) ):
                test = data.iloc[test_index]
                test_low = float( test["Low"] )
                test_close = float( test["Close"] )
                test_volume = float( test["Volume"] )
                test_tr = float( test["_TR"] ) if pd.notna( test["_TR"] ) else 0.0

                inside_test_zone = (
                    event_low - TEST_LOW_ATR * atr_ref <= test_low
                    <= event_low + TEST_HIGH_ATR * atr_ref
                )
                if inside_test_zone:
                    lower_volume = event_volume > 0 and test_volume <= event_volume * TEST_VOLUME_RATIO
                    narrower_range = event_tr > 0 and test_tr <= event_tr * TEST_RANGE_RATIO
                    if test_close >= event_low and lower_volume and narrower_range:
                        successful_test = True
                        break
                    weak_test_seen = True

            if successful_test:
                test_status = "RÉUSSI"
            elif invalidated:
                test_status = "ÉCHOUÉ"
            elif weak_test_seen:
                test_status = "FAIBLE"
            else:
                test_status = "EN ATTENTE"

    confirm_flags = [
        automatic_rally,
        close_above_event,
        above_ema,
        higher_lows,
        macd_rising,
        buying_volume,
        successful_test,
    ]
    confirm_score = int( sum( confirm_flags ) )

    if invalidated:
        status = "INVALIDÉE"
    elif (
        bars_ago >= MIN_CONFIRM_AGE
        and automatic_rally
        and above_ema
        and higher_lows
        and (buying_volume or successful_test)
        and confirm_score >= MIN_CONFIRM_SCORE
    ):
        status = "RETOURNEMENT CONFIRMÉ"
    elif bars_ago > MAX_CONFIRMATION_AGE and not automatic_rally:
        status = "SANS SUITE"
    elif bars_ago >= 1 and (automatic_rally or (rebound >= 0.03 and close_above_event)):
        status = "EN CONFIRMATION"
    else:
        status = "CAPITULATION POTENTIELLE"

    return {
        "bars_ago": bars_ago,
        "rebound": rebound,
        "max_rebound": max_rebound,
        "automatic_rally": automatic_rally,
        "above_ema": above_ema,
        "higher_lows": higher_lows,
        "macd_rising": macd_rising,
        "buying_volume": buying_volume,
        "successful_test": successful_test,
        "test_status": test_status,
        "confirm_flags": confirm_flags,
        "confirm_score": confirm_score,
        "status": status,
        "last_price": last_close,
        "last_low": last_low,
    }

# ------------------------------------------------------------------------------
# Scanner multi-tickers
# ------------------------------------------------------------------------------

def _diagnostic_rejection( diagnostic: dict ) -> str:
    """Explique pourquoi la meilleure séance récente n'a pas passé le filtre obligatoire."""
    failures = []
    if not diagnostic["flags"][0]:
        failures.append( "C1 BAISSE" )
    if not diagnostic["flags"][1]:
        failures.append( "C2 VOLUME" )
    if not diagnostic["flags"][2]:
        failures.append( "C3 ATR" )
    if diagnostic["earnings"]:
        failures.append( "RÉSULTATS" )
    return " + ".join( failures ) if failures else "CRITÈRES DE BASE VALIDÉS"

def _marks( names: list[str], flags=() ) -> dict[str, str]:
    """'+' / '-' pour chaque critère ; tout à '-' si aucun flag n'est fourni."""
    marks = { name: "-" for name in names }
    marks.update( { name: "+" if flag else "-" for name, flag in zip( names, flags ) } )
    return marks

@dataclass
class ScanRow:
    """Une ligne du tableau de scan (un ticker)."""
    ticker: str
    name: str
    status: str
    reason: str
    event_date: str = "—"
    score_sc: float = numpy.nan
    drop: str = "—"
    vol_ratio: float = numpy.nan
    turnover_ratio: float = numpy.nan
    atr_ratio: float = numpy.nan
    rsi_min: float = numpy.nan
    rebound: str = "—"
    age: int | str = "—"
    test: str = "—"
    score_conf: float = numpy.nan
    rebound_max: float = numpy.nan
    criteria: dict = field( default_factory=lambda: _marks( CRITERIA ) )
    confirmation: dict = field( default_factory=lambda: _marks( CONFIRMATION_CRITERIA ) )

    def __post_init__( self ):
        self.name = str( self.name )[:24]

    @staticmethod
    def _event_metrics( event: dict ) -> dict:
        """Champs communs à un événement détecté et à une séance rejetée."""
        return {
            "event_date":     str( event["date"] )[:10],
            "score_sc":       event["score"],
            "drop":           f"-{event['drop']:.0%}",
            "vol_ratio":      round( event["vol_ratio"], 2 ),
            "turnover_ratio": round( event["turnover_ratio"], 2 ),
            "atr_ratio":      round( event["atr_ratio"], 2 ),
            "rsi_min":        round( event["rsi_min"], 1 ) if pd.notna( event["rsi_min"] ) else numpy.nan,
        }

    @classmethod
    def empty( cls, ticker: str, name: str, status: str, reason: str ) -> "ScanRow":
        """Crée une ligne pour qu'un ticker reste visible même si ses données sont inutilisables."""
        return cls( ticker, name, status, reason )

    @classmethod
    def from_event( cls, ticker: str, name: str, event: dict, confirmation: dict ) -> "ScanRow":
        """Ligne d'un épisode de capitulation qualifié, avec son suivi de confirmation."""
        return cls(
            ticker, name, confirmation["status"], "ÉVÉNEMENT DÉTECTÉ",
            **cls._event_metrics( event ),
            rebound=f"{confirmation['rebound']:+.1%}",
            age=confirmation["bars_ago"],
            test=confirmation["test_status"],
            score_conf=confirmation["confirm_score"],
            rebound_max=confirmation["max_rebound"],
            criteria=_marks( CRITERIA, event["flags"] ),
            confirmation=_marks( CONFIRMATION_CRITERIA, confirmation["confirm_flags"] ),
        )

    @classmethod
    def from_rejected( cls, ticker: str, name: str, diagnostic: dict, data: pd.DataFrame ) -> "ScanRow":
        """Ligne de la meilleure séance récente rejetée, pour expliquer le rejet."""
        low = diagnostic["low"]
        rebound = (float( data["Close"].iloc[-1] ) - low) / low if low > 0 else 0.0
        return cls(
            ticker, name, "REJETÉE", _diagnostic_rejection( diagnostic ),
            **cls._event_metrics( diagnostic ),
            rebound=f"{rebound:+.1%}",
            age=len( data ) - 1 - int( diagnostic["index"] ),
            rebound_max=rebound,
            criteria=_marks( CRITERIA, diagnostic["flags"] ),
        )

    def to_dict( self ) -> dict:
        """Seul endroit où les champs sont associés aux libellés de colonnes du tableau."""
        return {
            "Ticker":         self.ticker,
            "Nom":            self.name,
            "Date SC":        self.event_date,
            "Score SC":       self.score_sc,
            "Chute":          self.drop,
            "Vol / médiane":  self.vol_ratio,
            "Valeur éch. x":  self.turnover_ratio,
            "ATR ratio":      self.atr_ratio,
            "RSI min":        self.rsi_min,
            "Rebond":         self.rebound,
            "Âge":            self.age,
            "Test":           self.test,
            "Score conf":     self.score_conf,
            "Statut":         self.status,
            "Motif":          self.reason,
            "Rebond max":     self.rebound_max,
            "Score conf max": len( CONFIRMATION_CRITERIA ),
            **self.criteria,
            **self.confirmation,
        }

# -----------------------------------------------------------------------------

def scan_capitulation( tickers: list[str] ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scanne tous les tickers et retourne une ligne de statistiques par ticker, plus le bilan historique."""
    tickers = list( dict.fromkeys( tickers ) )
    rows: list[ScanRow] = []
    backtest_rows = []
    volume_method_comparison = {
        "sessions": 0,
        "median_c2": 0,
        "mean_c2": 0,
        "disagreements": 0,
    }

    min_index = max( DROP_LOOKBACK, SMA_PERIOD, VOL_PERCENTILE_LOOKBACK, ATR_PERIOD, VOL_REF )

    for ticker in tickers:
        _short_name = ticker
        try:
            _ticker = yfinance.Ticker( ticker )
            df = _ticker.history( period=HISTORY_PERIOD, auto_adjust=False )

            try:
                _short_name = _ticker.get_info().get( "shortName" ) or ticker
            except Exception:
                _short_name = ticker

            data = _prepare_history( df )
            if data is None:
                print( f"[{ticker}] Historique insuffisant ou format OHLCV indisponible." )
                rows.append( ScanRow.empty( ticker, _short_name, "DONNÉES INSUFF.", "Historique court/invalide" ) )
                continue
            data["_EARNINGS"] = _earnings_mask( _ticker, data )

            # Détection historique pour le bilan statistique descriptif.
            if BACKTEST_ENABLED and len( data ) > MIN_BARS + BACKTEST_HORIZON:
                all_candidates = _find_climax_candidates(
                    data,
                    start_index=max( DROP_LOOKBACK, SMA_PERIOD, VOL_PERCENTILE_LOOKBACK, ATR_PERIOD ),
                    end_index=len( data ) - BACKTEST_HORIZON,
                )
                historical_events = _cluster_events( all_candidates )

                for event in historical_events:
                    event_index = int( event["index"] )
                    future = data.iloc[event_index + 1:event_index + BACKTEST_HORIZON + 1]
                    if len( future ) < BACKTEST_HORIZON:
                        continue

                    entry_price = float( data["Open"].iloc[event_index + 1] )
                    if entry_price <= 0:
                        continue

                    backtest_rows.append( {
                        "Ticker": ticker,
                        "Date": str( event["date"] )[:10],
                        "Score SC": f"{event['score']}/5",
                        "Volume / médiane x": event["vol_ratio"],
                        "Valeur échangée / médiane x": event["turnover_ratio"],
                        "C2 médiane": bool( event["flags"][1] ),
                        "ATR x": event["atr_ratio"],
                        "+5 séances": float( future["Close"].iloc[4] / entry_price - 1 ),
                        "+10 séances": float( future["Close"].iloc[9] / entry_price - 1 ),
                        "+20 séances": float( future["Close"].iloc[19] / entry_price - 1 ),
                        "+5% sous 20j": bool( future["High"].max() >= entry_price * 1.05 ),
                        "Clôture sous creux": bool(
                            (future["Close"] < event["low"] - INVALIDATION_ATR * event["atr_ref"]).any()
                        ),
                    } )

            # On calcule aussi les statistiques des séances non qualifiées sur la fenêtre récente.
            recent_start = max( min_index, len( data ) - MAX_EVENT_AGE - 1 )
            diagnostic_events = [
                event for event in (
                    _measure_climax_at( data, index )
                    for index in range( recent_start, len( data ) )
                )
                if event is not None
            ]

            # Comparaison à critères identiques : seul le dénominateur de C2 change.
            for diagnostic_event in diagnostic_events:
                median_c2 = bool( diagnostic_event["flags"][1] )
                volume_method_comparison["sessions"] += 1
                volume_method_comparison["median_c2"] += int( median_c2 )

            recent_candidates = _find_climax_candidates( data, recent_start, len( data ) )
            recent_events = _cluster_events( recent_candidates )

            if recent_events:
                # Pour le signal, on suit l'épisode qualifié le plus récent.
                event = max( recent_events, key=lambda item: item["index"] )
                confirmation = _evaluate_confirmation( data, event )
                rows.append( ScanRow.from_event( ticker, _short_name, event, confirmation ) )

                print(
                    f"{_short_name} ({ticker}) : capitulation {event['score']}/5 | "
                    f"chute -{event['drop']:.0%} | vol/médiane x{event['vol_ratio']:.2f} "
                    f"valeur éch. x{event['turnover_ratio']:.2f}) | "
                    f"amplitude x{event['atr_ratio']:.2f} ATR | confirmation "
                    f"{confirmation['confirm_score']}/7 | {confirmation['status']}"
                )
            elif diagnostic_events:
                # Le meilleur cas imparfait des dernières séances permet d'expliquer le rejet.
                diagnostic = max(
                    diagnostic_events,
                    key=lambda item: (
                        int( item["score"] ),
                        int( sum( item["flags"][:3] ) ),
                        int( item["index"] ),
                    ),
                )
                rows.append( ScanRow.from_rejected( ticker, _short_name, diagnostic, data ) )
            else:
                # Pas assez de séances utilisables dans la fenêtre de détection.
                rows.append( ScanRow.empty( ticker, _short_name, "DONNÉES INSUFF.", "Aucune séance mesurable" ) )

        except Exception as e:
            print( f"[{ticker}] Erreur : {e}" )
            rows.append( ScanRow.empty( ticker, _short_name, "ERREUR", str( e )[:28] ) )

    df_scan = pd.DataFrame( [row.to_dict() for row in rows] )
    if not df_scan.empty:
        status_order = {
            "RETOURNEMENT CONFIRMÉ": 0,
            "EN CONFIRMATION": 1,
            "CAPITULATION POTENTIELLE": 2,
            "REJETÉE": 3,
            "SANS SUITE": 4,
            "INVALIDÉE": 5,
            "DONNÉES INSUFF.": 6,
            "ERREUR": 7,
        }
        df_scan["_status_order"] = df_scan["Statut"].map( status_order ).fillna( 8 )
        df_scan = (
            df_scan.sort_values(
                ["_status_order", "Score SC", "Vol / médiane", "ATR ratio"],
                ascending=[True, False, False, False],
                na_position="last",
            )
            .drop( columns=["_status_order"] )
            .reset_index( drop=True )
        )

    df_backtest = pd.DataFrame( backtest_rows )
    counts = df_scan["Statut"].value_counts().to_dict() if not df_scan.empty else {}
    print(
        f"\nBilan du scan : {len( df_scan )}/{len( tickers )} tickers affichés | "
        f"signaux {sum( counts.get(s, 0) for s in ('CAPITULATION POTENTIELLE', 'EN CONFIRMATION', 'RETOURNEMENT CONFIRMÉ'))} | "
        f"rejetées {counts.get('REJETÉE', 0)} | données insuff. {counts.get('DONNÉES INSUFF.', 0)} | "
        f"erreurs {counts.get('ERREUR', 0)}"
    )
    print(
        "Comparaison C2 sur les fenêtres récentes : "
        f"{volume_method_comparison['sessions']} séances évaluables | "
        f"C2 médiane : {volume_method_comparison['median_c2']} | "
        f"C2 moyenne (ancienne règle) : {volume_method_comparison['mean_c2']} | "
        f"désaccords entre méthodes : {volume_method_comparison['disagreements']}"
    )
    return df_scan, df_backtest

# ------------------------------------------------------------------------------
# Bilan historique indicatif
# ------------------------------------------------------------------------------

def show_backtest_summary( df_backtest: pd.DataFrame ) -> None:
    """Imprime des statistiques descriptives sur les événements historiques détectés."""
    if df_backtest.empty:
        print( "\nBacktest descriptif : aucun événement historique exploitable." )
        return

    print( "\n--- Bilan historique descriptif des capitulations détectées ---" )
    print( f"Événements indépendants après regroupement : {len( df_backtest )}" )
    for horizon in BACKTEST_HORIZONS:
        column = f"+{horizon} séances"
        returns = pd.to_numeric( df_backtest[column], errors="coerce" ).dropna()
        if not returns.empty:
            win_rate = (returns > 0).mean()
            print(
                f"À +{horizon:2d} séances : rendement moyen {returns.mean():+.2%} | "
                f"rendement médian {returns.median():+.2%} | séances positives {win_rate:.0%}"
            )

    touch_rate = df_backtest["+5% sous 20j"].mean()
    invalidation_rate = df_backtest["Clôture sous creux"].mean()
    print( f"Hausse intrajournalière d'au moins +5 % en 20 séances : {touch_rate:.0%}" )
    print( f"Clôture ultérieure sous le creux ajusté de l'ATR : {invalidation_rate:.0%}" )
    print( "Note : entrée indicative à l'ouverture suivante - statistiques brutes, sans frais, "
           "slippage, benchmark ni contrôle de biais de sélection." )

# ------------------------------------------------------------------------------
# Rapport Matplotlib
# ------------------------------------------------------------------------------

def show_scan_report( df_scan: pd.DataFrame, nb_tickers: int ) -> Figure:
    """Rapport de capitulation
    Colonnes définies en pixels, axe Y orienté vers le bas, carte arrondie,
    en-tête en pilule, séparateurs fins, scores en barres et critères en pastilles.
    Le thème reste piloté par settings.MODE.dark et les deux palettes _THEME.
    """
    T = _THEME[ settings.MODE.dark ]  # lire le mode courant à chaque rendu
    dark_mode = bool( settings.MODE.dark )
    GRID = "#303744" if dark_mode else "#C8CDD5"
    SCORE_OFF = "#2A303A" if dark_mode else "#D5D7DA"

    counts = df_scan["Statut"].value_counts().to_dict() if not df_scan.empty else {}
    nb_potential = counts.get( "CAPITULATION POTENTIELLE", 0 )
    nb_confirming = counts.get( "EN CONFIRMATION", 0 )
    nb_confirmed = counts.get( "RETOURNEMENT CONFIRMÉ", 0 )
    nb_rejected = counts.get( "REJETÉE", 0 )
    nb_no_data = counts.get( "DONNÉES INSUFF.", 0 ) + counts.get( "ERREUR", 0 )
    nb_signals = sum(
        counts.get( status, 0 )
        for status in (
            "CAPITULATION POTENTIELLE", "EN CONFIRMATION", "RETOURNEMENT CONFIRMÉ",
            "SANS SUITE", "INVALIDÉE",
        )
    )

    # --- Colonnes : (clé, titre, largeur en pixels, alignement) ---
    columns = [
        ("Ticker",       "TICKER",       82,  "left"),
        ("Nom",          "SOCIÉTÉ",     175,  "left"),
        ("Date SC",      "DATE SC",      85,  "center"), # SC: Selling Climax
        ("Chute",        "CHUTE",        68,  "right"),
        ("Vol / médiane", "VOL / MÉD.",  78,  "right"),
        ("Vol / moyenne", "VOL / MOY.",  78,  "right"),
        ("Valeur éch. x", "VAL. ÉCH. ×", 82,  "right"),
        ("ATR ratio",     "ATR RATIO",   70,  "right"),
        ("RSI min",      "RSI MIN",      65,  "right"),
        ("Rebond",       "REBOND",       76,  "right"),
        ("Âge",          "ÂGE",          46,  "center"),
        ("Détail",       "TEST / MOTIF",140,  "center"),
        ("Score SC",     "SCORE SC",    108,  "left"),
        ("Score conf",   "SCORE CONF",  112,  "left"),
        ("C1",           "*C1",          44,  "center"),
        ("C2",           "*C2",          44,  "center"),
        ("C3",           "*C3",          44,  "center"),
        ("C4",           "*C4",          44,  "center"),
        ("C5",           "*C5",          44,  "center"),
        ("Statut",       "STATUT",      123,  "center"),
    ]

    # --- Géométrie (1 unité = 0.01 pouce) --- #
    OUT, IN, CELL_PAD = 20, 14, 8
    TITLE_H, HEADER_H, ROW_H, FOOTER_H = 110, 44, 42, 84
    W = 2 * (OUT + IN) + sum( column[2] for column in columns )
    visible_rows = max( len( df_scan ), 1 )
    card_h = HEADER_H + visible_rows * ROW_H + 10
    H = TITLE_H + card_h + FOOTER_H

    fig = Figure( figsize=(W / 100, H / 100), dpi=110, facecolor=T["BG"] )
    Watermark.apply( fig )
    ax = fig.add_axes( [0, 0, 1, 1] )
    ax.set_xlim( 0, W )
    ax.set_ylim( H, 0 )  # axe Y vers le bas : mêmes coordonnées que le tableau de référence
    ax.axis( "off" )

    def pill( x: float, y: float, w: float, h: float, color: str,
              alpha: float = 1.0, zorder: float = 1 ):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle=f"round,pad=0,rounding_size={h / 2}",
                fc=color, ec="none", alpha=alpha, zorder=zorder,
            )
        )

    # Calcul des positions de chaque colonne.
    x_pos, x = {}, OUT + IN
    for key, title, width, align in columns:
        x_pos[key] = (x, width, align)
        x += width

    def anchor( key: str ):
        x0, width, align = x_pos[key]
        if align == "left":
            return x0 + CELL_PAD, "left"
        if align == "right":
            return x0 + width - CELL_PAD, "right"
        return x0 + width / 2, "center"

    def score_value( value ) -> int | None:
        try:
            if pd.isna( value ):
                return None
            return int( float( value ) )
        except (TypeError, ValueError):
            return None

    def display_number( value, fmt: str, prefix: str = "" ) -> str:
        try:
            if pd.isna( value ):
                return "—"
            return prefix + format( float( value ), fmt )
        except (TypeError, ValueError):
            return "—"

    # --- Titre --- #
    ax.text(
        OUT, 31, "Capitulation vendeuse - Seller Climax", fontsize=17, fontweight="bold",
        color=T["TEXT"], va="center",
    )
    
    #--- Sous-titre --- #
    ax.text(
        OUT, 61,
        f"{nb_tickers} tickers analysés   ·   {datetime.today():%d/%m/%Y}   ·   "
        f"baisse ≥ {MIN_DROP:.0%}   ·   volume ≥ ×{VOLUME_MULTIPLIER:.1f} médiane ou Q{VOL_PERCENTILE:.0%}   ·   "
        f"TR ≥ {MIN_ATR_MULTIPLIER:.1f} ATR   ·   fenêtre {MAX_EVENT_AGE} séances",
        fontsize=8.5, color=T["MUTED"], va="center",
    )

    # Badge de synthèse à droite
    badge_label = f"{nb_signals} épisode(s) détecté(s)"
    badge_w = max(150, 16 + 7.0 * len( badge_label ))
    badge_color = T["GREEN"] if nb_confirmed else T["ORANGE"] if (nb_potential or nb_confirming) else GRID
    pill( W - OUT - badge_w, 17, badge_w, 30, badge_color, 0.16 )
    ax.text(
        W - OUT - badge_w / 2, 32, badge_label, fontsize=9,
        fontweight="bold", color=badge_color, ha="center", va="center",
    )

    # Résumé compact des statuts : pas de grosses cartes KPI.
    kpis = [
        ("POTENTIELS", nb_potential, T["ORANGE"]),
        ("EN CONF.", nb_confirming, T["BLUE"]),
        ("RET. OK", nb_confirmed, T["GREEN"]),
        ("REJETÉS", nb_rejected, T["RED"]),
        ("SANS DONNÉES", nb_no_data, T["MUTED"]),
    ]
    kpi_x_positions = [OUT, OUT + 205, OUT + 365, OUT + 505, OUT + 655]
    for (label, value, color), kpi_x in zip( kpis, kpi_x_positions ):
        ax.text( kpi_x, 88, label, fontsize=7, fontweight="bold", color=T["MUTED"], va="center" )
        # Décale le nombre selon la longueur du libellé pour éviter les collisions.
        value_x = kpi_x + 7.2 * len( label ) + 12
        ax.text( value_x, 88, str( value ), fontsize=9, fontweight="bold", color=color, va="center" )

    # --- Carte principale et en-tête arrondi --- #
    ax.add_patch(
        FancyBboxPatch(
            (OUT, TITLE_H), W - 2 * OUT, card_h,
            boxstyle="round,pad=0,rounding_size=14",
            fc=T["BG"], ec=GRID, lw=1.0, zorder=0,
        )
    )
    
    # --- En-tête --- #
    pill( OUT + 8, TITLE_H + 8, W - 2 * OUT - 16, HEADER_H - 12, T["BG_LEGEND"], 1.0 )
    y_head = TITLE_H + HEADER_H / 2 + 2
    for key, title, width, align in columns:
        tx, ha = anchor( key )
        ax.text(
            tx, y_head, title, fontsize=7.2, fontweight="bold",
            color=T["MUTED"], ha=ha, va="center",
        )

    status_colors = {
        "CAPITULATION POTENTIELLE": T["ORANGE"],
        "EN CONFIRMATION": T["BLUE"],
        "RETOURNEMENT CONFIRMÉ": T["GREEN"],
        "INVALIDÉE": T["RED"],
        "SANS SUITE": T["MUTED"],
        "REJETÉE": T["RED"],
        "DONNÉES INSUFF.": T["MUTED"],
        "ERREUR": T["RED"],
    }
    status_labels = {
        "CAPITULATION POTENTIELLE": "POTENTIEL",
        "EN CONFIRMATION": "CONFIRM.",
        "RETOURNEMENT CONFIRMÉ": "RET. OK",
        "INVALIDÉE": "INVALIDÉ",
        "SANS SUITE": "SANS SUITE",
        "REJETÉE": "REJETÉ",
        "DONNÉES INSUFF.": "SANS DATA",
        "ERREUR": "ERREUR",
    }

    # --- Aucune ligne --- #
    if df_scan.empty:
        ax.text(
            W / 2, TITLE_H + HEADER_H + ROW_H / 2,
            "Aucun ticker n'a pu être analysé.", fontsize=11,
            color=T["MUTED"], ha="center", va="center",
        )

    # --- Lignes  --- #
    for i in range( len( df_scan ) ):
        rec = df_scan.iloc[i]
        status = str( rec.get( "Statut", "ERREUR" ) )
        status_color = status_colors.get( status, T["MUTED"] )
        yc = TITLE_H + HEADER_H + i * ROW_H + ROW_H / 2

        # Accent subtil uniquement pour les statuts actifs ; sinon simple séparateur.
        accent = status_color if status in (
            "CAPITULATION POTENTIELLE", "EN CONFIRMATION", "RETOURNEMENT CONFIRMÉ"
        ) else None
        if accent:
            ax.add_patch(
                FancyBboxPatch(
                    (OUT + 8, yc - ROW_H / 2 + 3), W - 2 * OUT - 16, ROW_H - 6,
                    boxstyle="round,pad=0,rounding_size=8",
                    fc=accent, ec="none", alpha=0.09,
                )
            )
            ax.add_patch(
                FancyBboxPatch(
                    (OUT + 8, yc - 9), 3.5, 18,
                    boxstyle="round,pad=0,rounding_size=1.7",
                    fc=accent, ec="none",
                )
            )
        elif i < len( df_scan ) - 1:
            ax.plot(
                [OUT + IN, W - OUT - IN], [yc + ROW_H / 2] * 2,
                color=GRID, lw=0.8, alpha=0.7,
            )

        # Ticker, nom et date de l'événement / du meilleur candidat imparfait.
        tx, ha = anchor( "Ticker" )
        ax.text( tx + 2, yc, str( rec.get( "Ticker", "—" ) ), fontsize=8.7,
                 fontweight="bold", color=T["TEXT"], ha=ha, va="center" )
        tx, ha = anchor( "Nom" )
        ax.text( tx, yc, str( rec.get( "Nom", "—" ) )[:23], fontsize=7.6,
                 color=T["TEXT"], ha=ha, va="center" )
        tx, ha = anchor( "Date SC" )
        ax.text( tx, yc, str( rec.get( "Date SC", "—" ) ), fontsize=7.2,
                 color=T["MUTED"], ha=ha, va="center" )

        # Chute, volume, ATR, RSI et rebond.
        tx, ha = anchor( "Chute" )
        drop_value = str( rec.get( "Chute", "—" ) )
        ax.text( tx, yc, drop_value, fontsize=8.4, fontweight="bold",
                 color=T["RED"] if drop_value != "—" else T["MUTED"], ha=ha, va="center" )

        vol_value = rec.get( "Vol / médiane", numpy.nan )
        vol_num = float( vol_value ) if pd.notna( vol_value ) else numpy.nan
        tx, ha = anchor( "Vol / médiane" )
        ax.text(
            tx, yc, display_number( vol_num, ".2f", "×" ), fontsize=7.8,
            fontweight="bold" if pd.notna( vol_num ) and vol_num >= VOLUME_MULTIPLIER else "normal",
            color=T["GREEN"] if pd.notna( vol_num ) and vol_num >= VOLUME_MULTIPLIER else T["TEXT"] if pd.notna( vol_num ) else T["MUTED"],
            ha=ha, va="center",
        )

        mean_value = rec.get( "Vol / moyenne", numpy.nan )
        mean_num = float( mean_value ) if pd.notna( mean_value ) else numpy.nan
        tx, ha = anchor( "Vol / moyenne" )
        ax.text(
            tx, yc, display_number( mean_num, ".2f", "×" ), fontsize=7.6,
            fontweight="bold" if pd.notna( vol_num ) and vol_num >= VOLUME_MULTIPLIER else "normal",
            color=T["GREEN"] if pd.notna( mean_num ) and mean_num >= VOLUME_MULTIPLIER else T["TEXT"] if pd.notna( mean_num ) else T["MUTED"],
            ha=ha, va="center",
        )

        turnover_value = rec.get( "Valeur éch. x", numpy.nan )
        turnover_num = float( turnover_value ) if pd.notna( turnover_value ) else numpy.nan
        tx, ha = anchor( "Valeur éch. x" )
        ax.text(
            tx, yc, display_number( turnover_num, ".2f", "×" ), fontsize=7.6,
            color=T["TEXT"] if pd.notna( turnover_num ) else T["MUTED"],
            ha=ha, va="center",
        )

        atr_value = rec.get( "ATR ratio", numpy.nan )
        atr_num = float( atr_value ) if pd.notna( atr_value ) else numpy.nan
        tx, ha = anchor( "ATR ratio" )
        ax.text(
            tx, yc, display_number( atr_num, ".2f", "×" ), fontsize=7.8,
            fontweight="bold" if pd.notna( atr_num ) and atr_num >= MIN_ATR_MULTIPLIER else "normal",
            color=T["ORANGE"] if pd.notna( atr_num ) and atr_num >= MIN_ATR_MULTIPLIER else T["TEXT"] if pd.notna( atr_num ) else T["MUTED"],
            ha=ha, va="center",
        )

        tx, ha = anchor( "RSI min" )
        rsi_value = rec.get( "RSI min", numpy.nan )
        ax.text(
            tx, yc, display_number( rsi_value, ".0f" ), fontsize=8.0,
            fontweight="bold" if pd.notna( rsi_value ) and float( rsi_value ) <= RSI_OVERSOLD else "normal",
            color=T["ORANGE"] if pd.notna( rsi_value ) and float( rsi_value ) <= RSI_OVERSOLD else T["TEXT"] if pd.notna( rsi_value ) else T["MUTED"],
            ha=ha, va="center",
        )

        tx, ha = anchor( "Rebond" )
        rebound_value = str( rec.get( "Rebond", "—" ) )
        ax.text(
            tx, yc, rebound_value, fontsize=8.0, fontweight="bold",
            color=T["BLUE"] if status not in ("REJETÉE", "DONNÉES INSUFF.", "ERREUR") else T["MUTED"],
            ha=ha, va="center",
        )

        tx, ha = anchor( "Âge" )
        age_value = rec.get( "Âge", "—" )
        if isinstance( age_value, (int, numpy.integer, float, numpy.floating) ) and pd.notna( age_value ):
            age_value = str( int( age_value ) )
        ax.text( tx, yc, str( age_value ), fontsize=7.8, color=T["TEXT"], ha=ha, va="center" )

        # Le motif est visible pour un rejet sinon, on affiche l'état du retest.
        test_value = str( rec.get( "Test", "—" ) )
        reason = str( rec.get( "Motif", "—" ) )
        if status == "REJETÉE":
            detail = reason.replace( "C1 BAISSE", "C1 DROP" ).replace( "C2 VOLUME", "C2 VOL" )
            detail_color = T["RED"]
        elif status in ("DONNÉES INSUFF.", "ERREUR"):
            detail = reason
            detail_color = T["MUTED"]
        else:
            detail = test_value
            detail_color = T["GREEN"] if test_value == "RÉUSSI" else T["ORANGE"] if test_value in ("FAIBLE", "EN ATTENTE", "ÉCHOUÉ") else T["MUTED"]
        tx, ha = anchor( "Détail" )
        ax.text( tx, yc, detail[:23], fontsize=7.0, fontweight="bold" if test_value == "RÉUSSI" else "normal",
                 color=detail_color, ha=ha, va="center" )

        # Score de capitulation : cinq mini-barres + n/5.
        score_sc = score_value( rec.get( "Score SC", numpy.nan ) )
        bx, _ = anchor( "Score SC" )
        score_color = accent if accent else T["MUTED"]
        if score_sc is None:
            ax.text( bx + 30, yc, "—", fontsize=8, color=T["MUTED"], va="center" )
        else:
            for k in range( len( CRITERIA ) ):
                pill(
                    bx + k * 12, yc - 3.5, 10, 7,
                    score_color if k < score_sc else SCORE_OFF,
                    1.0 if k < score_sc else 0.55,
                )
            ax.text( bx + len( CRITERIA ) * 12 + 4, yc, f"{score_sc}/5", fontsize=7.8,
                     fontweight="bold", color=T["TEXT"], va="center" )

        # Score de confirmation : sept mini-barres + n/7.
        score_conf = score_value( rec.get( "Score conf", numpy.nan ) )
        bx, _ = anchor( "Score conf" )
        if score_conf is None:
            ax.text( bx + 31, yc, "—", fontsize=8, color=T["MUTED"], va="center" )
        else:
            conf_color = T["GREEN"] if score_conf >= MIN_CONFIRM_SCORE else T["MUTED"]
            for k in range( len( CONFIRMATION_CRITERIA ) ):
                pill(
                    bx + k * 9.2, yc - 3.5, 7.5, 7,
                    conf_color if k < score_conf else SCORE_OFF,
                    1.0 if k < score_conf else 0.55,
                )
            ax.text( bx + len( CONFIRMATION_CRITERIA ) * 9.2 + 3, yc, f"{score_conf}/7", fontsize=7.4,
                     fontweight="bold", color=T["TEXT"], va="center" )

        # Critères C1-C5 : pastille pleine = validé, anneau = non validé.
        for criterion in CRITERIA:
            cx, _ = anchor( criterion )
            if rec.get( criterion, "-" ) == "+":
                ax.scatter( [cx], [yc], s=52, marker="o", c=T["GREEN"],
                            edgecolors=T["GREEN"], linewidths=0.8, zorder=3 )
            else:
                ax.scatter( [cx], [yc], s=52, marker="o", facecolors="none",
                            edgecolors=GRID, linewidths=1.1, zorder=3 )

        # Statut sous forme de badge, selon la méthode du script de référence.
        sx, _ = anchor( "Statut" )
        label = status_labels.get( status, status )
        badge_widths = {
            "POTENTIEL": 88, "CONFIRM.": 82, "RET. OK": 68, "INVALIDÉ": 76,
            "SANS SUITE": 92, "REJETÉ": 70, "SANS DATA": 88, "ERREUR": 70,
        }
        badge_w = badge_widths.get( label, min( 110, 14 + 7 * len( label ) ) )
        pill( sx - badge_w / 2, yc - 11, badge_w, 22, status_color, 0.16 )
        ax.text( sx, yc, label, fontsize=7.6, fontweight="bold", color=status_color,
                 ha="center", va="center" )

    # --- Pied de page / légende ---
    y_foot = TITLE_H + card_h + 22
    ax.text(
        OUT, y_foot,
        "* critère de base obligatoire   ·   barres : score sur 5 / sur 7   ·   pastille pleine : critère validé",
        fontsize=8.2, color=T["MUTED"], va="center",
    )
    ax.text(
        OUT, y_foot + 19,
        "C1 Baisse préalable + plus-bas   ·   C2 Volume / médiane (ancienne C2 moyenne en colonne séparée)   ·   C3 Choc ATR   ·   C4 Rejet bas de bougie   ·   C5 RSI survente",
        fontsize=7.7, color=T["MUTED"], va="center",
    )
    ax.text(
        OUT, y_foot + 37,
        "Val. éch. = clôture × volume, rapportée à la médiane historique (proxy)   ·   CONF : rebond, EMA10, structure, MACD, volumes et retest",
        fontsize=7.4, color=T["MUTED"], va="center",
    )

    fig.subplots_adjust( left=0, right=1, top=1, bottom=0 )
    return fig

# ------------------------------------------------------------------------------
# Point d'entrée
# ------------------------------------------------------------------------------

def main():
    print( "--- Check API's parameters ---" )
    if not api.check_parameters( ["tickers"] ):
        exit( 1 )

    tickers = api.tickers
    print( "--- Scanner de capitulation vendeuse ---" )
    print(
        f"Baisse ≥ {MIN_DROP:.0%} sous SMA{SMA_PERIOD} | volume ≥ ×{VOLUME_MULTIPLIER:.1f} médiane "
        f"ou percentile {VOL_PERCENTILE:.0%} | TR ≥ {MIN_ATR_MULTIPLIER:.1f} ATR | "
        f"événement dans les {MAX_EVENT_AGE} dernières séances.\n"
    )

    df_scan, df_backtest = scan_capitulation( tickers )
    if not df_scan.empty:
        print( "\n--- Statistiques de tous les tickers scannés ---" )
        print( df_scan.to_string( index=False ) )

    if BACKTEST_ENABLED:
        show_backtest_summary( df_backtest )

    fig = show_scan_report( df_scan, len( set( tickers ) ) )

    if not run_standalone:
        api.show_figure( fig, 'table', 'Capitulation Scanner' )  # rendu thread-safe dans le thread Tkinter principal
    else:
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        root = tk.Tk()
        root.title( f"Capitulation Scanner - {len( tickers )} tickers analysés" )

        fig_w_px = int( fig.get_figwidth() * fig.dpi )
        fig_h_px = int( fig.get_figheight() * fig.dpi )
        win_h = min( fig_h_px + 20, root.winfo_screenheight() - 100 )
        root.geometry( f"{fig_w_px}x{win_h}" )

        outer = tk.Frame( root )
        outer.pack( fill="both", expand=True )
        v_scroll = tk.Scrollbar( outer, orient="vertical" )
        v_scroll.pack( side="right", fill="y" )
        scroll_area = tk.Canvas( outer, yscrollcommand=v_scroll.set, highlightthickness=0 )
        scroll_area.pack( side="left", fill="both", expand=True )
        v_scroll.config( command=scroll_area.yview )

        inner = tk.Frame( scroll_area )
        scroll_area.create_window( (0, 0), window=inner, anchor="nw" )
        canvas = FigureCanvasTkAgg( fig, master=inner )
        canvas.draw()
        canvas.get_tk_widget().pack()

        inner.bind( "<Configure>", lambda e: scroll_area.configure( scrollregion=scroll_area.bbox( "all" ) ) )
        scroll_area.bind_all( "<MouseWheel>", lambda e: scroll_area.yview_scroll( int( -1 * (e.delta / 120) ), "units" ) )
        root.mainloop()

if run_standalone:
    api_context = {
        "tickers": [
            "HO.PA", "SAF.PA", "RKLB", "AM.PA", "AIR.PA", "GE", "AAPL", "MSFT", "GOOGL", "AMZN",
            "NVDA", "IT", "BABA", "AMD", "BA", "BIDU", "BP", "RACE", "FTNT",
            "F", "GM", "INTC", "JD", "BMW.DE", "CROX",
        ],
    }

    api = UserScriptAPI()
    api.update( **api_context )
    main()
