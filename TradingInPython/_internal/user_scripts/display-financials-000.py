""" Tool_Finance - Afficher les données financières de l'entreprise
    
    Avant de penser à faire des calculs financiés compliqués 
    Trouvons les bonnes infos aux bons endroits dans les différents champs du Stock
    
    FCF - Free Cash Flow
    EBIT - Earnings Before Interest and Taxes = CA - Achats - Charges Externes - Charges de personnel - Autres charges
    EBITDA = EBIT - Provisions et Dotations aux amortissements
    
"""
import yfinance

# ----------------------------------------------------------------------------
# Format big number to be readable by user
# ----------------------------------------------------------------------------
# The idea is to display big number in format like x xxx xxx xxx
# when numer are small put decimals
#
def format_number( number, max_value=1e+12, deci=0 ):
    """ If number > max_value format to scientifique number
        else put space each 3 digits
    """    
    if abs( number ) > max_value:
        return "{:.3e}".format(number)
    
    if abs( number ) < 1000:
        return "{:.{decimals}f}".format(number, decimals=3)

    return "{:,.{decimals}f}".format(number, decimals=deci).replace(",", " ")

# ---------------------------------------------------

PRINT_STOCK_CASH_FLOW = False
def print_stock_cash_flow( msg ):
    if PRINT_STOCK_CASH_FLOW:
        print( msg )
        
# ---------------------------------------------------

PRINT_STOCK_FINANCIAL = False
def print_stock_financial( msg ):
    if PRINT_STOCK_FINANCIAL:
        print( msg )

separator = f"--------------------------------------"
max_value = 1e+12 # show human's readable number
        
# ---------------------------------------------------
        
# Récupérer les données financières de l'entreprise
#
# symbol = "AM.PA" # DASSAULT AVIATION
# symbol = "AB.PA" # AB SCIENCE
# symbol = "VOW.XETR" #Volkswagen NONE
# symbol = "VOW.DE" # NONE
# symbol = "TSLA"
#ticker =  { "symbol": "MSFT" } # MICROSOFT
#ticker =  { "symbol": "LR.PA" } # LEGRAND
#ticker =  { "symbol": "TRI.PA" } # TRIGANO
#ticker =  { "symbol": "GTLB" } # GITLAB RG-A
#ticker =  { "PLTR" } # Palantir Technologies Inc.
#ticker =  { "BRZE" } # BRAZE RG-A"
#ticker =  { "ASY.PA" } # ASSYSTEM
#ticker =  { "RTX" } # RXT aeronautic and defense
ticker = 'WPM' # Wheaton Precious Metals (Gold)
ticker = 'ETL.PA' # EUTELSAT

# ----------------------------------------------------------------------------

stock = yfinance.Ticker( ticker )
stock_info = stock.info.copy() # ne faire qu'une requête et sauver tous les résultats
if len(stock_info) == 1: # astuce pour savoir que stock_info est vide ... l'objet est créé mais il n'y a rien dedans
    print( f"Le symbol { ticker } n'existe pas")
    exit()
        
# ----------------------------------------------------------------------------
        
short_name = stock_info['shortName']
industry_key = stock_info['industryKey']
stock_exchange = stock_info.get('exchange')

print( separator )
print( f"Short Name: {short_name}" )
print( f"Industry Key: {industry_key}" )
print( f"Stock market : {stock_exchange}" )
print( separator )
print( f"Beta - Indice de volatilité : {stock_info['beta']}" )
print( f"Capitalisation boursière : {format_number(stock_info['marketCap'], max_value)}" )
print( separator )

# Afficher les états financiers
# -----------------------------
stock_financials = stock.financials.copy()

print_stock_financial( 'stock.financials' )
print_stock_financial( separator )
for k in stock_financials.axes[0]:
    print_stock_financial( k )
#print( stock_financials.axes[0] )
print_stock_financial( separator )

# EBIT (Earnings Before Interest and Taxes)
# -----------------------------------------
# EBIT = EBITDA - DA (Dépréciations & Amortissements)
ebit = stock_financials.loc['EBIT']
print( ebit.name )
print( separator )
print( ebit )
print( separator )

ebit_first_year = ebit.values[-1]  # Bénéfices avant intérêts et impôts première année
print(f"Bénéfice avant intérêts et impôts la première années: {format_number(ebit_first_year, max_value)}")
ebit_last_year = ebit.values[0]
print(f"Bénéfice avant intérêts et impôts la dernière années: {format_number(ebit_last_year, max_value)}")

print( separator )
ebitda = stock_financials.loc['EBITDA']
print( ebitda.name )
print( separator )
print( ebitda )
print( separator )

ebitda_last_year = ebitda.values[0]
print(f"Bénéfice avant intérêts et impôts et dépréciation : {format_number(ebitda_last_year, max_value)}")
print( separator )

# NONE !
#stock_earnings = stock.earnings # NONE ! cf. Ticker.income_stmt

# Free Cash Flow (flux de trésorerie libre)
# -----------------------------------------
# FCF = EBIT - Investissement en capital + Amortissements

stock_cash_flow = stock.cashflow.copy() # ne faire qu'une requête

print_stock_cash_flow( separator )
print_stock_cash_flow("Stock Cash Flow Index")
print_stock_cash_flow( stock_cash_flow.index )
print_stock_cash_flow( separator )

print_stock_cash_flow( stock_cash_flow )
print_stock_cash_flow( separator )

print( stock_cash_flow.loc['Free Cash Flow'].name )
print( separator )
print( stock_cash_flow.loc['Free Cash Flow'] )
print( separator )

# Free Cash Flow la dernière année
fcf = stock_cash_flow.loc['Free Cash Flow'].values[0]
print( f"Free Cash Flow last year : {format_number( fcf, max_value )}" )
print( separator )

# Dépenses d'investissement
capex = stock_cash_flow.loc['Capital Expenditure']
print( capex.name )
print( separator )
print( capex )
print( separator )

# Amortissement
if 'Depreciation' in stock_cash_flow:
    depreciation = stock_cash_flow.loc['Depreciation']
    print( depreciation.name )
    print( separator )
    print( depreciation )
    print( separator )
else:
    print('=> NO ->Depreciation<-')
    print( separator )
    
# Balance Sheet
#-----------------
stock_balance_sheet = stock.balance_sheet.copy()

# Récupération de données supplémentaires
if 'Long Term Debt' in stock_balance_sheet:
    total_debt = stock_balance_sheet.loc['Long Term Debt']
    print( total_debt.name )
    print( separator )
    print( total_debt )
    print( separator )
else:
    print( '=> NO ->Long Term Debt<-')
    print( separator )

total_equity = stock_balance_sheet.loc['Stockholders Equity']
print( total_equity.name )
print( separator )
print( total_equity )
print( separator )

# Dividends
# ---------
stock_dividends = stock.dividends.copy()

print( stock_dividends.name )
print( separator )
print( f"Nombre d'années de dividendes : {stock_dividends.size}")
dividends_last_years = stock_dividends[ len(stock_dividends)-5: ]
print( dividends_last_years )
print( separator )


