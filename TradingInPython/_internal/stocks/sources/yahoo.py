""" Yahoo Finance Source 
    Used by : StocksImportApp
"""

import yfinance as yf

from concurrent.futures import ThreadPoolExecutor

# -----------------------------------------------------------------------------

class YahooSource:

    NAME = "Yahoo Finance"
    HAS_SECTOR = True # False pour une source dont les résultats n'ont ni secteur ni industrie
    QUERY = yf.EquityQuery # surchargé par YahooETFSource

    # Classement du screener : libellé = ( champ de tri Yahoo, ordre croissant )
    SORTS = {
        "Capitalisation ↓": ( "intradaymarketcap", False ),
        "Volume moyen 3 mois ↓": ( "avgdailyvol3m", False ),
        "Symbole A → Z": ( "ticker", True )
    }

    def search( self, query, max_results=20 ):
        search = yf.Search( query, max_results=max_results )
        results = []

        for stock in search.quotes:
            symbol = stock.get( "symbol", "" )

            if not symbol:
                continue

            results.append( {
                "symbol": symbol,
                "name": stock.get( "shortname", stock.get( "longname", symbol ) ),
                "exchange": stock.get( "exchange", "" ),
                "type": stock.get( "quoteType", "" ),
                "sector": stock.get( "sectorDisp" ) or stock.get( "sector" ) or "",
                "industry": stock.get( "industryDisp" ) or stock.get( "industry" ) or "",
                "source": self.NAME
            } )

        return results

    # -----------------------------------------------------------------------------

    def list_market( self, region, exchange, offset=0, size=250, sort_field="ticker", sort_asc=False ):
        query = self.QUERY(
            "and",
            [
                self.QUERY( "eq", [ "region", region ] ),
                self.QUERY( "eq", [ "exchange", exchange ] )
            ]
        )

        result = yf.screen( query, offset=offset, size=size, sortField=sort_field, sortAsc=sort_asc )
        results = []

        for stock in result.get( "quotes", [] ):
            symbol = stock.get( "symbol", "" )

            if not symbol:
                continue

            results.append( {
                "symbol": symbol,
                "name": stock.get( "shortName", stock.get( "shortname", stock.get( "longName", symbol ) ) ),
                "exchange": stock.get( "exchange", exchange ),
                "type": stock.get( "quoteType", "" ),
                "source": self.NAME
            } )

        return results

    # -----------------------------------------------------------------------------

    def _fetch_details( self, symbol ):
        """ ( symbole, {secteur, industrie} ) ou ( symbole, None ) si Yahoo ne répond pas """
        try:
            information = yf.Ticker( symbol ).info
        except Exception:
            return symbol, None

        # Réponse vide (limitation, symbole inconnu) : pas de réponse, pour ne pas enregistrer un secteur vide
        if not ( information.get( "quoteType" ) or information.get( "sector" ) ):
            return symbol, None

        return symbol, {
            "sector": information.get( "sector", "" ),
            "industry": information.get( "industry", "" )
        }

    # -----------------------------------------------------------------------------

    def get_details( self, symbols, workers=8 ):
        """ Le screener ne fournit ni secteur ni industrie : un appel par symbole, en parallèle.
            Retourne { symbole: {"sector", "industry"} } pour les symboles qui ont répondu.
        """
        with ThreadPoolExecutor( max_workers=workers ) as pool:
            return { symbol: details for symbol, details in pool.map( self._fetch_details, symbols ) if details is not None }

    # -----------------------------------------------------------------------------

    def verify( self, symbol ):
        ticker = yf.Ticker( symbol )
        data = ticker.history( period="5d", interval="1d" )

        if data is None or data.empty:
            return None

        information = ticker.info

        return {
            "symbol": symbol,
            "name": information.get( "longName", information.get( "shortName", symbol ) ),
            "exchange": information.get( "exchange", "" ),
            "type": information.get( "quoteType", "" ),
            "sector": information.get( "sector", "" ),
            "industry": information.get( "industry", "" ),
            "source": self.NAME
        }

    # -----------------------------------------------------------------------------
