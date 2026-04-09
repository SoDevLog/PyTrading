""" Tool_Finance - Plot Analyse des données financières par ChatGpt

    Debt-to-Equity Ratio (D/E) est un ratio financier qui mesure le niveau d'endettement d'une entreprise par rapport à ses fonds propres. 
    Il est calculé comme suit : D/E = Total des dettes / Fonds propres

    Un ratio de X signifie que l'entreprise a X euros de dettes pour chaque euro de fonds propres.
"""
import yfinance
import matplotlib.pyplot as plt

from matplotlib import style

import sys
sys.path.append( 'C:\\Users\\Mabyre\\Documents\\GitHub\\PythonAdvanced\\' )
import helper as h

# ----------------------------------------------------------------------------
style.use('seaborn-v0_8-paper')
# ----------------------------------------------------------------------------

DO_PLOT_GRAPH = True

# ----------------------------------------------------------------------------

#ticker = {"name": "GITLAB RG-A", "symbol": "GTLB"}
#ticker = { "name": "MICROSOFT", "symbol": "MSFT" }
#ticker = { "symbol": "SOI.PA" } # SOITEC
#ticker = { "name": "MONGODB-A", "symbol": "MDB" }
#ticker = { "symbol": "PLTR" } # Palantir Technologies Inc.
#ticker = { "symbol": "BRZE", "name": "BRAZE RG-A" }
#ticker = { "symbol": "GAM", "name": "General American Investors Company, Inc." }
#ticker = { "name": "RIGETTI COMP", "symbol": "RGTI", }
#ticker = { "name": "QUANTUM COMP", "symbol": "QUBT", } # Quantum Computing Inc.
#ticker = { 'symbol': 'QMCO', } # Quantum Corporation 
#ticker = { 'symbol': 'QBTS' } # D-Wave Quantum Inc.
#ticker = { "symbol": "VU.PA" } 
#ticker = { "symbol": "HWT.UL" } # HUWAI non n'existe pas ! Huawei n'est pas une entreprise cotée en bourse
#ticker = { 'symbol': 'SIRI' } # SiriusXM Holdings Inc.
# ticker =  { "symbol": "ASY.PA" } # ASSYSTEM
# ticker =  { "symbol": "RTX" } # RXT aeronautic and defense
# ticker =  { "symbol": "ASY.PA" } # ASSYSTEM
#ticker =  { "symbol": "STMPA.PA" } # STMICROELECTRONICS
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
beta = stock_info['beta']  # Indice de volatilité

# Extraire et afficher des métriques clés
key_metrics = {
    "Enterprise Value": stock_info.get('enterpriseValue', 'N/A'),
    "Market Cap": stock_info.get('marketCap', 'N/A'),
    "Debt-to-Equity Ratio": stock_info.get('debtToEquity', 'N/A'),
    "Current Ratio": stock_info.get('currentRatio', 'N/A'),
    "Profit Margin": stock_info.get('profitMargins', 'N/A')
}

print(f"--- Santé financière de {short_name} ({ticker}) ---")
print(f"Industry Key: : {industry_key}")
print(f"Côté sur le marché : {stock_exchange}")
print(f"Inidice de volatilité : {beta}")

for metric, value in key_metrics.items():
    if value != 'N/A':
        print(f"{metric}: {h.format_number(value)}")
    else:
        print(f"{metric}: {value}")

# Récupérer les états financiers
stock_financials = stock.financials.copy() # compte de résultat
cash_flow = stock.cashflow.copy()
balance_sheet = stock.balance_sheet.copy() # bilan

# Afficher les revenus et bénéfices
if DO_PLOT_GRAPH:
    if not stock_financials.empty:
        revenues = stock_financials.loc['Total Revenue']
        profits = stock_financials.loc['Net Income']
        
        if 'EBITDA'not in stock_financials:
            print(f"=> Pas d'information sur l'EBITDA !")
            exit()
            
        ebitda = stock_financials.loc['EBITDA']
        assets = balance_sheet.loc['Current Assets'] # Actifs
        liabilities = balance_sheet.loc['Current Liabilities'] # Passif
        ratio = assets / liabilities

        # Visualisation des tendances
        fig = plt.figure( figsize=(10, 6) )
        fig.patch.set_facecolor('whitesmoke')
        
        ax1 = fig.add_subplot( facecolor='#FAFAFA' )
        ax1.plot(revenues.index, revenues.values, label="Revenues (Total Revenue)", marker="o")
        ax1.plot(profits.index, profits.values, label="Net Income (Profit)", marker="X")
        ax1.set_ylabel('Montants (en USD)')
        
        ax2 = ax1.twinx()
        # ebitda < 0 n'est pas significatif
        ax2.plot(ebitda.index, ebitda.values, label='EBITDA', marker='*', markersize=7, color='g')
        ax2.yaxis.label.set_color('g') 
        ax2.tick_params(axis='y', colors='g')
        
        ax3 = ax1.twinx()
        ax3.plot(ratio.index, ratio.values, label='RATIO', marker='s', markersize=7, color='b')
        ax3.spines.right.set_position(("axes", 1.06))
        ax3.yaxis.label.set_color('b') 
        ax3.tick_params(axis='y', colors='b')
        ax3.set_ylabel('RATIO')

        # Fusionner les légendes
        lns = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
        labels = [l.get_label() for l in lns]
        ax1.legend( lns, labels, loc='best' )
      
        plt.grid(linestyle='--', linewidth=0.8, alpha=0.7)
        plt.title(f"Tendances financières : {short_name} ({ticker['symbol']})")
        #plt.xlabel("Dates")
        #plt.ylabel("Montants (en USD)")
        #plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Les données financières ne sont pas disponibles pour cet actif.")

# Calculer le Free Cash Flow si possible
# Total Cash From Operating Activities
if not cash_flow.empty:
    try:
        fcf = (
            cash_flow.loc['Cash Flow From Continuing Operating Activities'].sum() -
            cash_flow.loc['Capital Expenditure'].sum()
        )
        print(f"Free Cash Flow (FCF): {h.format_number(fcf)}")
    except KeyError:
        print("Certains éléments nécessaires au calcul du Free Cash Flow sont absents.")
else:
    print("Les données de cash flow ne sont pas disponibles.")

# -----------------------------------------------------------------------------

ebitda_last_year = stock_financials.loc['EBITDA'].values[0]
ev = stock_info.get('enterpriseValue', 'N/A')

print(f"Bénéfice avant intérêts et impôts et dépréciation : {h.format_number(ebitda_last_year)}")
print(f"EV/EBITDA : {h.format_number(ev/ebitda_last_year, deci=3)}")

# Extraire les informations du bilan
# ----------------------------------
# if 'Total Current Assets' does not exist calculate it
if 'Total Current Assets' in balance_sheet.index:
    total_current_assets = balance_sheet.loc['Total Current Assets'].iloc[0]
elif 'Current Assets' in balance_sheet.index:
    total_current_assets = balance_sheet.loc['Current Assets'].sum()
else:    
    print( balance_sheet.index )

# Take last asset (actif)
current_assets = balance_sheet.loc['Current Assets'].iloc[0]

if 'Total Current Liabilities' in balance_sheet.index:
    total_current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0]
elif 'Current Assets' in balance_sheet.index:
    total_current_liabilities = balance_sheet.loc['Current Liabilities'].sum()
else:    
    print( balance_sheet.index )

# Take last liabilities (passif)
current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0]

total_assets = balance_sheet.loc['Total Assets'].iloc[0]
total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]
shareholders_equity = balance_sheet.loc['Ordinary Shares Number'].iloc[0]
cash = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]

# Calculs financiers
current_ratio = current_assets / current_liabilities
total_current_ratio = total_current_assets / total_current_liabilities

debt_to_equity = total_liabilities / shareholders_equity

# Affichage des résultats
print("--- Analyse du bilan comptable :")
print(f"Actifs à court terme : {h.format_number(current_assets)}")
print(f"Passifs à court terme : {h.format_number(current_liabilities)}")
print(f"Actifs totaux : {h.format_number(total_assets)}")
print(f"Passifs totaux : {h.format_number(total_liabilities)}")
print(f"Fonds propres : {h.format_number(shareholders_equity)}")
print(f"Cash & équivalents : {h.format_number(cash)}")
print(f"Current Ratio : {h.format_number(current_ratio)}")
print(f"Total Current Ratio : {h.format_number(total_current_ratio)}")
print(f"Debt-to-Equity Ratio : {h.format_number(debt_to_equity)}")
