"""Issue #14 — les artistes du dessin précédent doivent disparaître au redessin.

`on_mobile_average_params_change` ne retire que les clés de `artists_ichimoku` :
la ligne de prix, la droite de tendance et les chandeliers restent sur les axes
et s'empilent à chaque changement de paramètre (ou à chaque `<FocusOut>`).

Ces tests pilotent le vrai `draw_main_graph`, sans interface : seuls les
modules livrés uniquement en .pyd (config, figure, tkinterh, settings) sont
remplacés par des doubles.
"""
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

INTERNAL = Path(__file__).resolve().parents[1] / "TradingInPython" / "_internal"
if str(INTERNAL) not in sys.path:
    sys.path.insert(0, str(INTERNAL))


# ---------------------------------------------------------------------------
# Doubles pour les modules compilés
# ---------------------------------------------------------------------------

def _stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


_stub("config", func=None)
_stub("config.func", read_configuration=lambda *a, **k: ({"stocks": []}, ""))
_stub("figure")
_stub("figure.helper", set_axe=lambda *a, **k: None)
_stub("figure.candlestick_ohlc",
      candlestick_ohlc=None, set_visibility_ohlc=lambda *a, **k: None)
_stub("figure.linedeltaselector",
      LineDeltaSelector=lambda *a, **k: types.SimpleNamespace(
          set_curve_data=lambda *x: None))
_stub("settings")
_stub("settings.default", MODE=types.SimpleNamespace(dark=False))
_stub("tkinterh")
_stub("tkinterh.helper", Tooltip=lambda *a, **k: None)
_stub("helper", format_number3f=lambda value: f"{value:.3f}")
_stub("figure.order_blocks_drawer",
      OrderBlockDrawer=lambda *a, **k: types.SimpleNamespace(do_print=lambda *x: None))
_stub("strategies")
_stub("strategies.smc_engine",
      SMC_Engine=type("SMC_Engine", (), {}),
      SMC_Params=type("SMC_Params", (), {}))

import strategy_ichimoku as ichimoku_module  # noqa: E402
import strategy_sma12e as sma12e_module  # noqa: E402
import strategy_smart_money as smart_money_module  # noqa: E402


def _real_candles(ax_main, ohlc, width, **kwargs):
    """Un artiste réel à la place des chandeliers compilés."""
    return ax_main.fill_between([0, 1], [0, 1], [1, 2], alpha=0.0)


ichimoku_module.candlestick_ohlc = _real_candles
sma12e_module.candlestick_ohlc = _real_candles
sma12e_module.set_visibility_ohlc = lambda *a, **k: None


# ---------------------------------------------------------------------------
# Faux widgets et objet stratégie
# ---------------------------------------------------------------------------

class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _MessageEntry:
    def config(self, **kwargs):
        pass


def _data(n=160, seed=5):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n)))
    frame = pd.DataFrame({
        "Open": close,
        "High": close * (1 + np.abs(rng.normal(0, 0.006, n))),
        "Low": close * (1 - np.abs(rng.normal(0, 0.006, n))),
        "Close": close,
        "Adj Close": close,
    }, index=index)
    frame["Date2num"] = np.arange(n)
    return frame


def _ichimoku():
    obj = object.__new__(ichimoku_module.strategy_ichimoku)
    obj.window = None
    obj.message_entry = _MessageEntry()
    obj.intraday = False
    obj.display = None
    obj.name = "TEST"
    obj.slope = None
    obj.data = _data()
    obj.MAKE_FORCASTING = False
    obj.artists_ichimoku = {}
    obj.candle_artists = []
    obj.line_price = None
    obj.line_tendency = None
    obj.mobile_average_1 = _Var(9)
    obj.mobile_average_2 = _Var(26)
    obj.mobile_average_3 = _Var(52)
    obj.nb_days_forcasted_entry = _Var(12)
    obj.nb_days_past_entry = _Var(26)
    obj.ichimoku_futuriste = _Var(False)
    obj.var_make_forcasting = _Var(False)
    obj.var_price = _Var(0)
    obj.var_candles = _Var(1)
    obj.var_tenkan_sen = _Var(1)
    obj.var_kijun_sen = _Var(1)
    obj.var_kijun_upper = _Var(0)
    obj.var_kijun_lower = _Var(0)
    obj.var_kijun_bands = _Var(1)
    obj.var_chikou = _Var(1)
    obj.var_kumo = _Var(1)
    obj.fig, obj.ax = plt.subplots()
    obj.set_canvas(types.SimpleNamespace(draw_idle=lambda: None))
    return obj


def _count(ax):
    return (len(ax.lines), len(ax.collections), len(ax.patches))


# ---------------------------------------------------------------------------
# Le bug
# ---------------------------------------------------------------------------

def test_ichimoku_redraw_does_not_accumulate_artists():
    obj = _ichimoku()
    obj.draw_main_graph(obj.ax, False, 0.48)
    after_first = _count(obj.ax)

    for _ in range(3):
        obj.on_mobile_average_params_change()

    assert _count(obj.ax) == after_first, (
        "chaque changement de paramètre laisse des artistes derrière lui : "
        f"{after_first} -> {_count(obj.ax)}"
    )


def test_ichimoku_draw_twice_does_not_duplicate_the_price_line():
    obj = _ichimoku()
    obj.draw_main_graph(obj.ax, False, 0.48)
    obj.draw_main_graph(obj.ax, False, 0.48)

    labels = [line.get_label() for line in obj.ax.lines]
    assert labels.count("Prix de clôture") == 1
    assert labels.count("Tendance") == 1


# ---------------------------------------------------------------------------
# strategy_sma12e : même défaut, la droite de tendance n'est même pas retenue
# ---------------------------------------------------------------------------

def _sma12e():
    obj = object.__new__(sma12e_module.strategy_sma12e)
    obj.window = None
    obj.message_entry = _MessageEntry()
    obj.intraday = False
    obj.display = None
    obj.name = "TEST"
    obj.data = _data()
    obj.slope = None
    obj.cum_price = 0.0
    obj.candle_artists = []
    obj.line_tendency = None
    obj.width = 0.5
    obj.change_chk_sates = True
    obj.mobile_average_1 = _Var(9)
    obj.mobile_average_2 = _Var(21)
    obj.mobile_average_3 = _Var(5)
    obj.mobile_average_4 = _Var(8)
    obj.mobile_average_5 = _Var(13)
    obj.mobile_average_6 = _Var(34)
    obj.mobile_average_exp = _Var(9)
    obj.var70 = _Var(False)
    obj.template = "{}"
    obj.fig, obj.ax = plt.subplots()
    obj.set_canvas(types.SimpleNamespace(draw_idle=lambda: None))
    obj.update_interface_chk = lambda: None
    obj.toggle_visibility = lambda: None
    return obj


def test_sma12e_update_does_not_accumulate_artists():
    obj = _sma12e()
    obj.draw_main_graph(obj.ax, False, 0.5)
    after_first = _count(obj.ax)

    for _ in range(3):
        obj.update_graph()

    assert _count(obj.ax) == after_first, (
        "chaque update laisse des artistes derrière lui : "
        f"{after_first} -> {_count(obj.ax)}"
    )


def test_sma12e_update_does_not_duplicate_the_tendency_line():
    obj = _sma12e()
    obj.draw_main_graph(obj.ax, False, 0.5)
    obj.update_graph()

    labels = [line.get_label() for line in obj.ax.lines]
    assert labels.count("Slope") == 1


# ---------------------------------------------------------------------------
# strategy_smart_money : la remise à zéro oubliait les artistes au lieu de
# les retirer (le moteur SMC compilé rend le dessin complet hors de portée)
# ---------------------------------------------------------------------------

def test_smart_money_reset_removes_the_previous_artists():
    obj = object.__new__(smart_money_module.strategy_smart_money)
    fig, ax = obj_ax = plt.subplots()
    line, = ax.plot([0, 1], [0, 1], label="price")
    cloud = ax.fill_between([0, 1], [0, 1], [1, 2])
    obj.ax = ax
    obj.artists = {"price": [line], "kumo": [cloud]}

    obj._reset_artists()

    assert len(ax.lines) == 0, "la ligne de prix est restée sur les axes"
    assert len(ax.collections) == 0, "le nuage est resté sur les axes"
    assert obj.artists == {"price": [], "kumo": []}
