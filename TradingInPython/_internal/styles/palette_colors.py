""" TradingInPython - Palette de couleurs
"""
import sys
from pathlib import Path
base = Path(__file__).resolve().parent.parent
sys.path.append( str(base) )
import settings.default as settings

_DARK = {
    "CANDLE_BULL": "#60DDBE",
    "CANDLE_BEAR": "#E08886",
    "UP": "#65F3E5",
    "DOWN": "#FA9191",    
    "TENKAN": "#80E581",
    "KIJUN": "#FFA585",
    "BLACK": "#DDDDDD",
    "RED": "#FF0000",
    "GREEN": "#4EDB4E",
    "BLUE":"#80D7FF",
    "LIGHT_RED": "#FFA190",
    "LIGHT_GREEN": "#BCF4BC",
    "LIGHT_BLUE":"#A9E4FF",
    "WATERMARK": "#FFFF00",
    # ... ajoute ici toutes tes autres couleurs dark
}

_LIGHT = {
    "CANDLE_BULL": "#77D879",  # 
    "CANDLE_BEAR": "#DB3F3F",  # 
    "UP": "#269B64",
    "DOWN": "#DB3F3F",
    "TENKAN": "darkgreen",
    "KIJUN": "orangered",
    "BLACK": "#000000",
    "RED": "#FF0000", 
    "GREEN": "#008000", 
    "BLUE":"navy",
    "LIGHT_RED": "#FFA190",
    "LIGHT_GREEN": "#BCF4BC",
    "LIGHT_BLUE":"#9CD3EC",
    "WATERMARK": "#006EFF9F",
    # ... ajoute ici toutes tes autres couleurs light
}

def __getattr__(name):
    palette = _DARK if settings.MODE.dark else _LIGHT
    try:
        return palette[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} n'a pas d'attribut {name!r}") from None