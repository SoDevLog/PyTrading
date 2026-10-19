"""Régression #40 - signaux portés par le nuage Kumo projeté (mode « Futuriste »).

En mode Futuriste, `draw_main_graph` ajoute `MA2` barres futures sans OHLC puis
calcule l'Ichimoku sur l'ensemble. Les Senkou spans sont donc bien projetés dans
le futur, mais `generate_signals()` n'y produit que des zéros : `Close` est NaN,
donc toutes ses comparaisons sont fausses. Résultat : aucun signal dans la zone
projetée et `Signal_display` figé sur la moyenne du prix.

`generate_leading_signals()` complète ces lignes en comparant la dernière
clôture connue (prolongée à plat) aux deux bornes du nuage projeté.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

INTERNAL = Path(__file__).resolve().parents[1] / "TradingInPython" / "_internal"
if str(INTERNAL) not in sys.path:
    sys.path.insert(0, str(INTERNAL))

from digitsignalprocessing import ichimoku_kinko_hyo  # noqa: E402


def _projected_frame(rows):
    """Frame minimal d'une zone projetée : clôture inconnue, spans connus.

    rows = [(span_a, span_b), ...] une entrée par barre future.
    La dernière clôture connue est 10.
    """
    index = pd.bdate_range("2024-01-01", periods=len(rows) + 1)
    data = pd.DataFrame(
        {
            "Close": [10.0] + [np.nan] * len(rows),
            "Senkou_span_A": [9.0] + [r[0] for r in rows],
            "Senkou_span_B": [8.0] + [r[1] for r in rows],
        },
        index=index,
    )
    return data, data["Close"].isna()


# ---------------------------------------------------------------------------
# Comportement attendu du nuage projeté
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rows, expected",
    [
        # clôture 10 sous les deux bornes -> baissier
        ([(11.0, 12.0), (14.0, 15.0)], -1),
        # clôture 10 au-dessus des deux bornes -> haussier
        ([(8.0, 9.0), (7.0, 8.0)], 1),
        # clôture 10 entre les deux bornes -> neutre
        ([(9.0, 11.0), (8.0, 12.0)], 0),
    ],
)
def test_projected_rows_get_a_signal_from_the_projected_cloud(rows, expected):
    data, mask = _projected_frame(rows)

    out = ichimoku_kinko_hyo.generate_leading_signals(data, mask)

    assert out.loc[mask, "Signal"].tolist() == [float(expected)] * len(rows)


def test_historical_signal_values_are_not_overwritten():
    data, mask = _projected_frame([(11.0, 12.0)])
    data["Signal"] = [1.0, np.nan]
    data["Signal_display"] = [101.0, np.nan]

    out = ichimoku_kinko_hyo.generate_leading_signals(data, mask)

    assert out["Signal"].iloc[0] == 1.0
    assert out.loc[mask, "Signal"].tolist() == [-1.0]


def test_signal_display_is_offset_by_the_price_mean():
    data, mask = _projected_frame([(11.0, 12.0), (14.0, 15.0)])

    out = ichimoku_kinko_hyo.generate_leading_signals(data, mask)

    mean = out["Close"].mean()
    assert out.loc[mask, "Signal_display"].tolist() == [mean - 1, mean - 1]


def test_rows_without_a_projected_cloud_stay_unsignalled():
    data, mask = _projected_frame([(np.nan, np.nan)])

    out = ichimoku_kinko_hyo.generate_leading_signals(data, mask)

    assert out.loc[mask, "Signal"].isna().all()


# ---------------------------------------------------------------------------
# Chaîne réelle : projection -> ichimoku_modernise -> signaux
# ---------------------------------------------------------------------------

def _futurist_pipeline(last_close, n=180, ma1=9, ma2=26, ma3=52):
    """Même contrat que strategy_ichimoku.draw_main_graph (mode Futuriste)."""
    index = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(3)
    data = pd.DataFrame(
        {
            "High": 100 + rng.uniform(0, 2, n),
            "Low": 98 + rng.uniform(0, 2, n),
            "Close": np.full(n, last_close),
        },
        index=index,
    )
    future = pd.date_range(index[-1] + pd.Timedelta(days=1), periods=ma2, freq="B")
    data = pd.concat(
        [data, pd.DataFrame(np.nan, index=future, columns=data.columns)]
    )
    mask = data["Close"].isna()

    data = ichimoku_kinko_hyo.ichimoku_modernise(data, ma1, ma2, ma3)
    return data, mask


def test_futurist_pipeline_leaves_no_signal_without_the_fix():
    """Le bug d'origine : la zone projetée reste muette malgré un nuage baissier."""
    data, mask = _futurist_pipeline(last_close=50.0)  # très sous le nuage (~99)

    data = ichimoku_kinko_hyo.generate_signals(data)

    assert data.loc[mask, "Senkou_span_A"].notna().all()   # le nuage est projeté
    assert data.loc[mask, "Signal"].eq(0).all()            # mais aucun signal


def test_futurist_pipeline_signals_the_projected_cloud():
    data, mask = _futurist_pipeline(last_close=50.0)

    data = ichimoku_kinko_hyo.generate_signals(data)
    data = ichimoku_kinko_hyo.generate_leading_signals(data, mask)

    assert data.loc[mask, "Signal"].eq(-1).all()
    mean = data["Close"].mean()
    assert data.loc[mask, "Signal_display"].round(6).eq(round(mean - 1, 6)).all()
