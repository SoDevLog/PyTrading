"""Les métriques de performance doivent mesurer ce qu'elles annoncent.

`generer_rapport_performance` calculait :

    'max_drawdown': np.min(rendements_array)   # la pire opération
    'sharpe_ratio': mean / std                 # ratio par opération, non annualisé

Un drawdown est la pire baisse depuis un sommet de la courbe de capital, pas la
pire opération : trois opérations à +10 %, -5 %, -6 % donnent un drawdown de
-10,7 %, pas -6 %. Le Sharpe doit être annualisé dès que l'historique porte des
dates.

Série de référence (7 points, 6 jours ouvrés d'intervalle, 1 461 jours au total
soit exactement 4 ans) :

    prix    100 -> 110 -> 104.5 -> 98.23
    opérations  +10 %   -5 %     -6 %
    capital  1.10 -> 1.045 -> 0.9823
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

INTERNAL = Path(__file__).resolve().parents[1] / "TradingInPython" / "_internal"
if str(INTERNAL) not in sys.path:
    sys.path.insert(0, str(INTERNAL))

# `yfinance` n'est pas installé hors de l'application empaquetée, et `debug`
# n'y est livré que compilé : deux doubles suffisent pour importer la stratégie.
if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = types.ModuleType("yfinance")
if "debug.func" not in sys.modules:
    sys.modules["debug"] = types.ModuleType("debug")
    sys.modules["debug.func"] = types.ModuleType("debug.func")
    sys.modules["debug.func"].print = lambda *a, **k: None

from strategies.kalman_filter import IndicateurTendanceKalman  # noqa: E402
from strategies.kalman_filter_forcast import (  # noqa: E402
    IndicateurTendanceKalman as IndicateurTendanceKalmanForcast,
)

MODULES = [
    pytest.param(IndicateurTendanceKalman, id="kalman_filter"),
    pytest.param(IndicateurTendanceKalmanForcast, id="kalman_filter_forcast"),
]

RETURNS = [10.0, -5.0, -6.0]


def _historique(with_dates=True):
    prix = np.array([100.0, 0.0, 110.0, 0.0, 104.5, 0.0, 98.23])
    signaux = np.array([1, 0, 1, 0, 1, 0, 1])
    historique = {"prix": prix, "signaux": signaux}
    if with_dates:
        # 2020-01-01 -> 2024-01-01 : 1 461 jours, soit exactement 4 ans
        historique["dates"] = pd.date_range("2020-01-01", "2024-01-01", periods=7)
    return historique


def _indicateur(cls, with_dates=True):
    obj = object.__new__(cls)
    obj.historique = _historique(with_dates=with_dates)
    return obj


@pytest.mark.parametrize("cls", MODULES)
def test_max_drawdown_is_measured_from_the_equity_peak(cls):
    perf = _indicateur(cls).generer_rapport_performance()

    assert perf["max_drawdown"] == pytest.approx(-10.7, abs=1e-9), (
        "le drawdown doit être la pire baisse depuis un sommet de la courbe de "
        f"capital (-10,7 %), pas la pire opération (-6 %) : {perf['max_drawdown']}"
    )


@pytest.mark.parametrize("cls", MODULES)
def test_sharpe_is_annualised_from_the_history_dates(cls):
    perf = _indicateur(cls).generer_rapport_performance()

    rendements = np.array(RETURNS)
    attendu = (rendements.mean() / rendements.std()) * np.sqrt(len(RETURNS) / 4.0)

    assert perf["sharpe_ratio"] == pytest.approx(float(attendu), rel=1e-9)


@pytest.mark.parametrize("cls", MODULES)
def test_sharpe_falls_back_to_a_per_trade_ratio_without_dates(cls):
    perf = _indicateur(cls, with_dates=False).generer_rapport_performance()

    rendements = np.array(RETURNS)
    assert perf["sharpe_ratio"] == pytest.approx(
        float(rendements.mean() / rendements.std()), rel=1e-9
    )


@pytest.mark.parametrize("cls", MODULES)
def test_other_metrics_are_preserved(cls):
    perf = _indicateur(cls).generer_rapport_performance()

    assert perf["nb_signaux"] == 3
    assert perf["taux_reussite"] == pytest.approx(100 / 3)
    assert perf["max_gain"] == pytest.approx(10.0)
