""" Yahoo Finance Source 
    Used by : StocksImportApp
"""

import yfinance as yf

# -----------------------------------------------------------------------------

class YahooSource:

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
                "source": "Yahoo Finance"
            } )

        return results

    # -----------------------------------------------------------------------------

    def list_market( self, region, exchange, offset=0, size=250 ):
        query = yf.EquityQuery(
            "and",
            [
                yf.EquityQuery( "eq", [ "region", region ] ),
                yf.EquityQuery( "eq", [ "exchange", exchange ] )
            ]
        )

        result = yf.screen( query, offset=offset, size=size )
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
                "source": "Yahoo Finance"
            } )

        return results

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
            "source": "Yahoo Finance"
        }

    # -----------------------------------------------------------------------------
