""" Tool_Finance - Juste Valeur à partir de la projection du Free Cash Flow FCF
    
    Capitalisation Globale de l'Entreprise
    
    Avantages il n'y a pas de paramètres (ou presque pas)
    
    Utiliser plus de données financières que par le 'DCF simple'
    
    Il faut regarder les valeurs de : cash_flow.loc['Free Cash Flow'] pour certaines sotck elles n'existent pas

    beta : Indice de volatilité de cette action par rapport au marché
    
    beta = 1 : Volatilité égale à celle du marché. Si le marché augmente de 1%, on peut s'attendre à ce que l'action augmente également de 1%.
    beta > 1 : L'action est plus volatile que le marché. Par exemple, si le beta est de 1,5, cela signifie que l'action est 50 % plus volatile que le marché. 
               Si le marché monte de 1%, l'action pourrait augmenter de 1,5%, et inversement en cas de baisse.
    beta < 1 : L'action est moins volatile que le marché. Un beta de 0,5 signifie que l'action est 50% moins volatile que le marché.
    beta négatif : Cela suggère que l'action pourrait se déplacer en sens inverse du marché, bien que cela soit assez rare.
    
    Paramètres :
        risk_free_rate = 0.02 # Taux sans risque (par exemple, 2%)
        market_return = 0.08  # Rendement attendu du marché
        cost_of_debt = 0.03  # Supposons un coût de la dette de 3%
        tax_rate = 0.30  # Taux d'imposition supposé de 30%
        growth_rate = 0.03  # Taux de croissance supposé des FCF de 3%
    
"""
import yfinance
import numpy as np
import sys
from pathlib import Path
base = Path(__file__).resolve().parent.parent.parent
sys.path.append( str(base) )
import helper as h

separator = f"-----------------------------------------------------------" 

# Étape 1: Récupérer les données économiques de l'action
#symbol = "AM.PA" # DASSAULT AVIATION
#symbol = "ALO.PA" # ALSTOM
#symbol = "CAP.PA" # CAPGEMINI
#symbol= "AIR.PA" # AIRBUS
symbol = "BEN.PA" # BENETEAU
#symbol = "SOI.PA" # SOITEC
#symbol= "VOW.DE" # VOLKSWAGEN
#symbol= "TSLA" # TESLA
# NO! symbol= "GAM" # General American Investors Company, Inc.
# symbol= "IONQ" # IONQ
# symbol= "SAF.PA" # SAFRAN
symbol= "HO.PA" # THALES
# symbol = "STLAP.PA" # STELLANTIS
#symbol = "STMPA.PA" # STMICRO

stock = yfinance.Ticker( symbol )

# Display main stock infos
stock_info = stock.info.copy() # ne faire qu'une requête et sauver tous les résultats
short_name = stock_info['shortName']
industry_key = stock_info['industryKey']
shares_outstanding = stock_info['sharesOutstanding']
current_price = stock.history(period='1d')['Close'].iloc[0]

print( separator )
print( 'Estimation of Fair Value by the method Free Cash Flow (FCF)' )
print( separator )
print( f"Short Name: {short_name}" )
print( f"Industry Key: {industry_key}" )
print( separator )
print( f"Shares outstanding: {h.format_number( shares_outstanding, 1e+12 )}" )
print( separator )

# Free Cash Flow (flux de trésorerie libre)
cash_flow = stock.cashflow.copy() # pareil ne faire qu'une requête

# Free Cash Flow
#fcf = cash_flow.loc['Free Cash Flow'].values[0] # dernière année
fcf = cash_flow.loc['Free Cash Flow'].values[1] # non nulle ou non négative

print(f"Free Cahs Flow: {h.format_number(fcf, 1e+12)}")
print( separator )
print( cash_flow.loc['Free Cash Flow'] )
print( separator )

# Récupération de données supplémentaires
if 'beta' in stock_info:
    beta = stock_info['beta']  # Indice de volatilité
    print( f"Volatility Index: {beta}")
else:
    print( f"=> Pas de valeur de l'indice de volatility !")
    beta = 1 # equivalent to market
    print( separator )

stock_balance_sheet = stock.balance_sheet.copy()
total_debt = 0
if 'Long Term Debt' in stock.balance_sheet.index:
    total_debt = stock_balance_sheet.loc['Long Term Debt'].values[0]
elif 'Long Term Debt And Capital Lease Obligation' in stock.balance_sheet.index:
    total_debt = stock_balance_sheet.loc['Long Term Debt And Capital Lease Obligation'].values[0]
    print( f"=> Use Long Term Debt And Capital Lease Obligation: {h.format_number(total_debt)}")
    print( separator )
else:
    print('=> No Long Term Debt on Long Term Debt And Capital Lease Obligation IMPOSSIBLE!')
    exit()

total_equity = stock_balance_sheet.loc['Stockholders Equity'].values[0]

# Étape 2: Calcul du coût des capitaux propres (CAPM ou WACC)
# WACC - Weighted Average Cost of Capital (coût moyen pondéré du capital)
# -----------------------------------------------------------
risk_free_rate = 0.02 # Taux sans risque (par exemple, 2%)
market_return = 0.08  # Rendement attendu du marché

cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)

debt_ratio = total_debt / (total_debt + total_equity)
equity_ratio = total_equity / (total_debt + total_equity)

cost_of_debt = 0.03  # Supposons un coût de la dette de 3%
tax_rate = 0.30  # Taux d'imposition supposé de 30%

wacc = (equity_ratio * cost_of_equity) + (debt_ratio * cost_of_debt * (1 - tax_rate))

print( f"Weighted Average Cost of Capital: {h.format_number(wacc)}")
print( separator )

# Étape 3: Estimer les flux de trésorerie futurs
#
growth_rate = 0.03  # Taux de croissance supposé des FCF de 3%
years = 5  # Période de prévision
future_fcfs = []

for i in range( 1, years + 1 ):
    future_fcf = fcf * (1 + growth_rate) ** i
    future_fcfs.append( future_fcf )

# Calcul de la valeur terminale avec une croissance perpétuelle
terminal_value = future_fcfs[-1] * (1 + growth_rate) / (wacc - growth_rate)

print( f"Terminal value: {h.format_number( terminal_value, 1e+12 )}" )

# Étape 4: Actualiser les flux de trésorerie futurs
#
discounted_fcfs = []
for i, fcf in enumerate( future_fcfs ):
    discounted_value = fcf / (1 + wacc) ** (i + 1)
    discounted_fcfs.append( discounted_value )

# Ajouter la valeur terminale actualisée
terminal_value_discounted = terminal_value / (1 + wacc) ** years
discounted_fcfs.append( terminal_value_discounted )

print( f"Terminal value discounted: {h.format_number( terminal_value_discounted, 1e+12 ) }" )

# Somme des flux de trésorerie actualisés pour obtenir la juste valeur de l'entreprise
enterprise_value = np.sum( discounted_fcfs )

# Comparer avec la capitalisation boursière actuelle
market_cap = stock_info['marketCap']
print( separator )
print( f"Capitalisation boursière l'entreprise : {h.format_number( market_cap, 1e+12 )} €.")
print( f"Juste valeur de l'entreprise estimée : {h.format_number( enterprise_value, 1e+12 )} €.")
print( separator )

theo_price = market_cap/shares_outstanding
fair_price = enterprise_value/shares_outstanding

print( f"Current price: {current_price:.3f}" )
print( f"Theoric price: {theo_price:.3f}" )
print( f"   Fair price: {fair_price:.3f}" )
print( separator )