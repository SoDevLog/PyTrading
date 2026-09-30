""" Yahoo Finance Source : ETF
    Même fonctionnement que YahooSource, avec le screener ETF de yfinance (ETFQuery, yfinance >= 1.x).
    Used by : StocksImportApp
"""

import yfinance as yf

from stocks.sources.yahoo import YahooSource

# -----------------------------------------------------------------------------

class YahooETFSource( YahooSource ):

    NAME = "Yahoo ETF"
    HAS_SECTOR = False # le screener ETF renvoie aussi quelques lignes typées EQUITY : sans secteur pour autant
    QUERY = yf.ETFQuery

    # Libellé = ( champ de tri Yahoo, ordre croissant ) ; la capitalisation n'existe pas pour les ETF
    SORTS = {
        "Actifs sous gestion ↓": ( "fundnetassets", False ),
        "Volume moyen 3 mois ↓": ( "avgdailyvol3m", False ),
        "Symbole A → Z": ( "ticker", True )
    }

    # -----------------------------------------------------------------------------

    def search( self, query, max_results=20 ):
        """ La recherche Yahoo mélange les types : on en demande plus, puis on ne garde que les ETF """
        results = super().search( query, max_results * 3 )
        return [ stock for stock in results if stock["type"] == "ETF" ][:max_results]

    # -----------------------------------------------------------------------------
