""" Tool_Finance
"""
import yfinance
import pandas as pd
import numpy as np
from datetime import datetime

def calculate_dcf_valuation( ticker, forecast_years=5, terminal_growth_rate=0.02 ):
    """
    Calcule la valeur d'une entreprise par la méthode DCF en utilisant les données de yfinance
    
    Args:
        ticker_symbol (str): Le symbole boursier de l'entreprise
        forecast_years (int): Nombre d'années de prévision
        terminal_growth_rate (float): Taux de croissance perpétuel
        
    Returns:
        dict: Résultats de l'évaluation DCF
    """
    
    # Options pour afficher tout le 'df'
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    
    # Récupération des données
    stock = yfinance.Ticker( ticker )
    
    # Données financières
    financials = stock.financials.copy()
    cashflow = stock.cashflow.copy()
    balance = stock.balance_sheet.copy()
    info = stock.info.copy()
    short_name = info.get( 'shortName' )
    
    # Vérification des données disponibles
    if financials.empty or cashflow.empty or balance.empty:
        return {"error": "Données financières insuffisantes pour l'analyse"}
    
    # Calcul des FCF historiques
    try:
        # Récupération des éléments nécessaires sur les 4 dernières années disponibles
        ebit = financials.loc['EBIT']
        tax_rate = 0.25  # Taux d'imposition estimé (à ajuster selon le pays)
        depreciation = cashflow.loc['Depreciation']
        capex = -cashflow.loc['Capital Expenditure']  # Négatif dans les données yfinance
        
        # Calcul du besoin en fonds de roulement (BFR)
        try:
            current_assets = balance.loc['Total Current Assets']
            current_liabilities = balance.loc['Total Current Liabilities']
            working_capital = current_assets - current_liabilities
            
            # Calcul de la variation du BFR
            wc_delta = working_capital.diff()
            wc_delta.iloc[0] = 0  # Pas de variation pour la première année
        except:
            # Si les données sont incomplètes, on estime le BFR à 0
            wc_delta = pd.Series(0, index=ebit.index)
        
        # Calcul du FCF historique
        historical_fcf = ebit * (1 - tax_rate) + depreciation + capex - wc_delta
        
        # Analyse de la croissance historique moyenne des FCF
        if len( historical_fcf ) >= 2:
            avg_growth_rate = (historical_fcf.iloc[-1] / historical_fcf.iloc[0]) ** (1 / (len(historical_fcf) - 1)) - 1
            
            # Limiter le taux de croissance à des valeurs raisonnables
            growth_rate = max(min(avg_growth_rate, 0.15), 0.01)
        else:
            growth_rate = 0.05  # Taux par défaut si historique insuffisant
        
        # Initialisation avec le dernier FCF connu
        last_fcf = historical_fcf.iloc[-1]
        
        # Calcul du WACC (Weighted Average Cost of Capital)
        # Simplification: on peut utiliser un WACC par défaut ou le calculer si les données nécessaires sont disponibles
        try:
            market_cap = info.get('marketCap', 0)
            total_debt = balance.loc['Total Debt'].iloc[-1] if 'Total Debt' in balance.index else 0
            
            equity_weight = market_cap / (market_cap + total_debt)
            debt_weight = total_debt / (market_cap + total_debt) if total_debt > 0 else 0
            
            cost_of_equity = 0.08  # Coût des capitaux propres estimé (béta × prime de risque + taux sans risque)
            cost_of_debt = 0.04 * (1 - tax_rate)  # Coût de la dette après impôt
            
            wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt
        except:
            wacc = 0.08  # WACC par défaut si calcul impossible
        
        # Projection des FCF futurs
        projected_fcf = []
        
        for year in range(1, forecast_years + 1):
            projected_fcf.append(last_fcf * (1 + growth_rate) ** year)
        
        # Calcul de la valeur terminale
        terminal_value = projected_fcf[-1] * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)
        
        # Actualisation des flux de trésorerie
        discount_factors = [(1 + wacc) ** -i for i in range(1, forecast_years + 1)]
        pv_fcf = sum(fcf * df for fcf, df in zip(projected_fcf, discount_factors))
        pv_terminal = terminal_value * discount_factors[-1]
        
        # Valeur totale de l'entreprise
        enterprise_value = pv_fcf + pv_terminal
        
        # Ajustements pour obtenir la valeur des capitaux propres
        try:
            cash = balance.loc['Cash'].iloc[-1] if 'Cash' in balance.index else balance.loc['Cash And Cash Equivalents'].iloc[-1]
            debt = balance.loc['Total Debt'].iloc[-1] if 'Total Debt' in balance.index else 0
            equity_value = enterprise_value + cash - debt
        except:
            equity_value = enterprise_value
        
        # Calcul du prix par action
        try:
            shares_outstanding = info.get('sharesOutstanding', 0)
            price_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0
        except:
            price_per_share = 0
        
        # Résultats
        current_price = info.get('currentPrice', 0)
        
        return {
            "ticker": ticker,
            "name": short_name,
            "date": datetime.now().strftime('%Y-%m-%d'),
            "historical_fcf": historical_fcf.to_dict(),
            "growth_rate": growth_rate,
            "wacc": wacc,
            "terminal_growth_rate": terminal_growth_rate,
            "projected_fcf": projected_fcf,
            "enterprise_value": enterprise_value,
            "equity_value": equity_value,
            "estimated_price_per_share": price_per_share,
            "current_market_price": current_price,
            "discount_premium": (price_per_share - current_price) / current_price if current_price > 0 else 0
        }
        
    except Exception as e:
        return {"error": f"Erreur lors du calcul: {str(e)}"}

# -----------------------------------------------------------------------------
# Exécution pour Safran (SAF.PA pour Euronext Paris)
#
def main():    
    symbol = "HO.PA" # "SAF.PA"
    result = calculate_dcf_valuation( symbol )
    
    # Affichage des résultats
    if "error" in result:
        print(f"Erreur: {result['error']}")
    else:
        print(f"\nÉvaluation DCF: {result['name']} ({result['ticker']}) au {result['date']}")
        print(f"Taux de croissance utilisé: {result['growth_rate']:.2%}")
        print(f"WACC: {result['wacc']:.2%}")
        print(f"Taux de croissance terminal: {result['terminal_growth_rate']:.2%}")
        
        print("\nFlux de trésorerie historiques:")
        for year, fcf in result['historical_fcf'].items():
            print(f"{year}: {fcf/1e6:.2f} millions €")
        
        print("\nFlux de trésorerie projetés:")
        for i, fcf in enumerate(result['projected_fcf']):
            print(f"Année {i+1}: {fcf/1e6:.2f} millions €")
        
        print(f"\nValeur d'entreprise: {result['enterprise_value']/1e9:.2f} milliards €")
        print(f"Valeur des capitaux propres: {result['equity_value']/1e9:.2f} milliards €")
        print(f"Prix par action estimé: {result['estimated_price_per_share']:.2f} €")
        print(f"Prix de marché actuel: {result['current_market_price']:.2f} €")
        
        discount = result['discount_premium']
        if discount > 0:
            print(f"L'action est sous-évaluée de {discount:.2%}")
        else:
            print(f"L'action est surévaluée de {-discount:.2%}")

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
