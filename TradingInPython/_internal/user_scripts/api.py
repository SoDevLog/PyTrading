""" user_api.py — API partagée pour les scripts utilisateur de TradingInPython.

    Usage dans un script utilisateur :
        from user_api import api

        def main():
            print( api.symbol )
            df = api.get_ohlcv()
            api.on_bar( my_callback )
"""

from __future__ import annotations
from typing import Callable, Optional
import pandas as pd

# -----------------------------------------------------------------------------
# Singleton exposé aux scripts utilisateur
# Peuplé par ScriptRunnerWindow avant d'exécuter le script.
# -----------------------------------------------------------------------------

class UserScriptAPI:
    def __init__( self ):
        self.name: str = ''
        self.ticker: str = ''
        self.period: str = ''
        self.interval: str = ''
        self.tickers: list = []
        self.df: pd.DataFrame = pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
        self.bar_callbacks:   list[Callable] = []
        self.close_callbacks: list[Callable] = []
    
    # -------------------------------------------------------------------------
    
    def update( self, **kwargs ) -> None:
        self.name = kwargs.get( "name", self.name )
        self.ticker = kwargs.get( "ticker", self.ticker )
        self.period = kwargs.get( "period", self.period )
        self.interval = kwargs.get( "interval", self.interval )
        self.tickers =  kwargs.get( "tickers", self.tickers )
        df = kwargs.get( "df" )
        if df is not None:
            self.df = df.copy()

    # -------------------------------------------------------------------------
    
    def check_parameters( self, required_params: list[str] ) -> bool:
        missing = [p for p in required_params if not getattr(self, p)]
        if missing:
            print(f"ERROR: Missing required parameters: {', '.join(missing)}")
            return False
        for p in required_params:
            print(f"{p}: {getattr( self, p )}")
        return True
    
    # -------------------------------------------------------------------------
    # Callbacks / événements
    # -------------------------------------------------------------------------

    def on_bar(self, callback: Callable[[pd.Series], None]) -> None:
        """
        Enregistre un callback appelé à chaque nouvelle barre.
        
        Le callback reçoit une pd.Series avec les colonnes OHLCV
        et un index correspondant au timestamp de la barre.

        Exemple :
            def my_handler(bar):
                print(bar['close'])
            api.on_bar(my_handler)
        """
        self.bar_callbacks.append(callback)

    def on_close(self, callback: Callable[[], None]) -> None:
        """
        Enregistre un callback appelé à la fermeture de l'app
        ou à la fin du script.

        Exemple :
            def on_exit():
                print("Script terminé proprement")
            api.on_close(on_exit)
        """
        self.close_callbacks.append(callback)

    # -------------------------------------------------------------------------
    # Méthodes internes - appelées par l'app, pas par les scripts
    # -------------------------------------------------------------------------

    def _fire_bar(self, bar: pd.Series) -> None:
        """Appelé par l'app pour notifier une nouvelle barre."""
        for cb in self.bar_callbacks:
            try:
                cb(bar)
            except Exception as e:
                print(f"[UserScriptAPI] on_bar callback error: {e}")

    def _fire_close(self) -> None:
        """Appelé par l'app ou ScriptRunner à la fin d'exécution."""
        for cb in self.close_callbacks:
            try:
                cb()
            except Exception as e:
                print(f"[UserScriptAPI] on_close callback error: {e}")

    def _reset_callbacks(self) -> None:
        """Vide les callbacks entre deux exécutions de scripts."""
        self.bar_callbacks.clear()
        self.close_callbacks.clear()

# -----------------------------------------------------------------------------
# Instance Singleton — peuplée par l'app avant l'exécution du script
# -----------------------------------------------------------------------------

api: Optional[UserScriptAPI] = None