""" Nasdaq Trader Source 
    Used by : StocksImportApp
"""

import csv
import io
import urllib.request


class NasdaqSource:
    NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

    # -----------------------------------------------------------------------------

    def search( self, query, max_results=20 ):
        query = query.lower().strip()

        if not query:
            return []

        results = self.list_stocks()
        matches = []

        for stock in results:
            symbol = stock.get( "symbol", "" ).lower()
            name = stock.get( "name", "" ).lower()

            if query in symbol or query in name:
                matches.append( stock )

                if len( matches ) >= max_results:
                    break

        return matches

    # -----------------------------------------------------------------------------

    def list_stocks( self ):
        results = []
        results.extend( self._read_nasdaq_listed() )
        results.extend( self._read_other_listed() )
        return results

    # -----------------------------------------------------------------------------

    def verify( self, symbol ):
        symbol = symbol.strip().upper()

        for stock in self.search( symbol, 1 ):
            if stock.get( "symbol", "" ).upper() == symbol:
                return stock

        return None

    # -----------------------------------------------------------------------------

    def _read_nasdaq_listed( self ):
        content = self._download( self.NASDAQ_LISTED_URL )
        reader = csv.DictReader( io.StringIO( content ), delimiter="|" )
        results = []

        for row in reader:
            symbol = row.get( "Symbol", "" ).strip()
            name = row.get( "Security Name", "" ).strip()
            test_issue = row.get( "Test Issue", "" ).strip()

            if not symbol or not name or test_issue == "Y":
                continue

            if symbol.startswith( "File Creation" ):
                continue

            results.append( {
                "symbol": symbol,
                "name": name,
                "exchange": "NASDAQ",
                "type": "ETF" if row.get( "ETF", "" ).strip() == "Y" else "EQUITY",
                "source": "Nasdaq"
            } )

        return results

    # -----------------------------------------------------------------------------

    def _read_other_listed( self ):
        content = self._download( self.OTHER_LISTED_URL )
        reader = csv.DictReader( io.StringIO( content ), delimiter="|", quotechar='"' )
        results = []

        for row in reader:
            symbol = row.get( "ACT Symbol", "" ).strip()
            name = row.get( "Security Name", "" ).strip()
            test_issue = row.get( "Test Issue", "" ).strip()

            if not symbol or not name or test_issue == "Y":
                continue

            if symbol.startswith( "File Creation" ):
                continue

            results.append( {
                "symbol": symbol,
                "name": name,
                "exchange": row.get( "Exchange", "US" ).strip(),
                "type": "ETF" if row.get( "ETF", "" ).strip() == "Y" else "EQUITY",
                "source": "Nasdaq"
            } )

        return results

    # -----------------------------------------------------------------------------

    def _download( self, url ):
        request = urllib.request.Request(
            url,
            headers={ "User-Agent": "Mozilla/5.0" }
        )

        with urllib.request.urlopen( request, timeout=20 ) as response:
            return response.read().decode( "utf-8", errors="replace" )

    # -----------------------------------------------------------------------------
