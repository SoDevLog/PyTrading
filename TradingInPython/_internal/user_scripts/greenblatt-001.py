""" Tool_Finance - Greenblatt Magic Formula

    La formule magique de Greenblatt est une stratégie d'investissement qui combine deux indicateurs financiers:        
        - Le rendement des bénéfices (Earnings Yield)
        - Le rendement du capital investi (Return on Capital)
    
    La formule magique de Greenblatt classe les actions en fonction de ces deux indicateurs, en attribuant un rang à chaque action pour chaque indicateur.
    Ensuite, les rangs sont additionnés pour obtenir un score global, et les actions avec les scores les plus bas sont considérées comme les meilleures opportunités d'investissement selon cette stratégie. 
        
"""
import pandas as pd
import yfinance
from user_scripts.api import api

# Check parameters for the script
if not api.check_parameters( ['tickers'] ):
    exit(1)

# Retreive data from API
tickers = api.tickers

# Retrieve financial data from Yahoo Finance
def get_financial_data( ticker ):
    stock = yfinance.Ticker( ticker )
    try:
        stock_info = stock.info.copy()
        industry_key = stock_info.get( 'industryKey' )
        short_name = stock_info.get( 'shortName' )
        sector = stock_info.get( 'sector' )
        print( f'{short_name} Sector: {sector} Industry key: {industry_key}' )
            
        financials = stock.financials.copy()
        financials_transposed = financials.transpose()
        balance_sheet = stock.balance_sheet.copy()
        history_6mo = stock.history( period='6mo' ).copy()  # 6 derniers mois
        
        # Capitalisation boursière
        market_cap = stock_info['marketCap']  
        
        # EBIT (bénéfice avant impôts et intérêts)
        if 'EBIT' in financials.index:
            ebit = financials.loc['EBIT'].iloc[0]
        elif 'Net Income' in financials.index:
            ebit = financials.loc['Net Income'].iloc[0]  # Pour se secteur Bancaire Revenu net = EBIT
        else:
            ebit = None
        
        # Trésorerie différentes clés possibles
        cash_keys = ['Cash', 'Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments']
        cash = next( (balance_sheet.loc[key].iloc[0] for key in cash_keys if key in balance_sheet.index), 0)

        # Dette différentes clés possibles
        debt_keys = ['Total Debt', 'Long Term Debt', 'Short Long Term Debt']
        total_debt = next((balance_sheet.loc[key].iloc[0] for key in debt_keys if key in balance_sheet.index), 0)
        
        # Total des actifs et passifs
        total_assets = balance_sheet.loc['Total Assets'].iloc[0]  # Actifs totaux
        total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]  # Passifs

        # Calcul de la valeur d'entreprise (EV = Market Cap + Debt - Cash)
        enterprise_value = market_cap + total_debt - cash

        # Calcul des indicateurs
        earnings_yield = ebit / enterprise_value if enterprise_value > 0 else None  # EY = EBIT / EV
        return_on_capital = ebit / (total_assets - total_liabilities) if total_assets > total_liabilities else None  # ROC = EBIT / Capital Investi

        # Calcule du momentum basé sur le rendement des 6 derniers mois
        historical_data = history_6mo  
        momentum = ( historical_data["Close"].iloc[-1] - historical_data["Close"].iloc[0] ) / historical_data["Close"].iloc[0]  # Rendement sur 6 mois

        # Croissance du chiffre d'affaires
        revenue_growth = None
        if 'Total Revenue' in financials_transposed.columns:
            revenue_current_year = financials.loc['Total Revenue'].iloc[0]  # Chiffre d'Affaires de l'année la plus récente
            revenue_previous_year = financials.loc['Total Revenue'].iloc[1]  # Chiffre d'Affaires de l'année précédente
            
            # Calcul de la croissance du chiffre d'affaires
            revenue_growth = (revenue_current_year - revenue_previous_year) / revenue_previous_year * 100

        return {
            'Short name': short_name,
            'Ticker': ticker,
            'Earnings Yield': earnings_yield, # Rendement des bénéfices
            'Return on Capital': return_on_capital, # ROC
            'Momentum (6 months)': momentum,
            'Revenue growth': revenue_growth
        }
        
    except Exception as e:
        print(f'Erreur avec le symbol {ticker}: {e}')
        return None

# Récupérer les données pour toutes les actions
data = [get_financial_data(ticker) for ticker in tickers]
data = [d for d in data if d]  # Supprimer les valeurs None

# Convertir en DataFrame
df = pd.DataFrame(data)

# Classement des actions selon EY et ROC
df['EY Rank'] = df['Earnings Yield'].rank(ascending=False)
df['ROC Rank'] = df['Return on Capital'].rank(ascending=False)

# Score final = Somme des rangs
df['Magic Formula Score'] = df['EY Rank'] + df['ROC Rank']

# Trier par meilleur score (plus bas = mieux classé)
df = df.sort_values(by='Magic Formula Score')

# Options pour afficher tout le 'df'
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Afficher les résultats
print( df )
