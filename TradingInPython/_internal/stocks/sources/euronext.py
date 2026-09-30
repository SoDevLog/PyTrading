""" Euronext Source 
    Used by : StocksImportApp
    Stocks All Market :
    > https://live.euronext.com/fr/products/equities/list
    
"""

import csv
from pathlib import Path

# -----------------------------------------------------------------------------

class EuronextSource:
    
    HAS_SECTOR = True # for columns 'sector' and 'industry'
    
    # -----------------------------------------------------------------------------

    def __init__( self ):
        self.file_name = Path(__file__).parent / "euronext.csv"

    # -----------------------------------------------------------------------------

    def file_ok( self ):
        return self.file_name.is_file()

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

        return self._read_file()

    # -----------------------------------------------------------------------------

    def verify( self, symbol ):
        symbol = symbol.strip().upper()

        for stock in self.search( symbol, 1 ):
            if stock.get( "symbol", "" ).upper() == symbol:
                return stock

        return None

    # -----------------------------------------------------------------------------

    def _read_file( self ):
        results = []

        with open( self.file_name, "r", encoding="utf-8-sig", newline="" ) as file:
            reader = csv.DictReader( file )

            for row in reader:
                symbol = row.get( "SecurityCode", "" ).strip()
                name = row.get( "SecurityDescription", "" ).strip()

                if not symbol or not name:
                    continue

                results.append( {
                    "symbol": symbol,
                    "name": name,
                    "exchange": row.get( "Market", "" ).strip(),
                    "type": row.get( "SecuritySubType", "" ).strip(),
                    "sector": row.get( "Sector", "" ).strip(),
                    "industry": row.get( "Industry", "" ).strip(),
                    "source": "Euronext"
                } )

        return results

    # -----------------------------------------------------------------------------
