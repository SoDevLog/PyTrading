""" Tool_Finance - Juste Valeur à partir de la projection du Free Cash Flow FCF

    Capitalisation Globale de l'Entreprise

    Avantages : peu de paramètres manuels, utilise plus de données financières que le DCF simple.

    Améliorations v2 :
        - Sélection automatique du premier FCF positif (plus de hardcode sur values[1])
        - Taux sans risque dynamique via le 10Y US Treasury (^TNX)
        - Coût de la dette calculé depuis les intérêts réels (Interest Expense / Long Term Debt)
        - Guard : wacc > growth_rate vérifié avant le calcul de la valeur terminale
        - Guard : total_equity négatif détecté et signalé
        - Deux scénarios : conservateur (3%) + optimiste (croissance historique FCF plafonnée à 15%)
        - Affichage upside/downside en % pour chaque scénario
        - Correction bug : stock_balance_sheet utilisé partout (plus d'appel double à stock.balance_sheet)
        - Correction typo "Free Cahs Flow"
        - Séparateur sans f-string inutile
        - print(separator) ajouté après affichage du beta

    beta : Indice de volatilité de l'action par rapport au marché
        beta = 1   : Volatilité égale au marché.
        beta > 1   : Plus volatile que le marché (ex. beta 1.5 → 50% plus volatile).
        beta < 1   : Moins volatile que le marché (ex. beta 0.5 → 50% moins volatile).
        beta < 0   : Se déplace en sens inverse du marché (rare).

    Paramètres fixes :
        market_return = 0.08   # Rendement attendu du marché (8%)
        tax_rate      = 0.30   # Taux d'imposition supposé (30%)
        growth_rate_conservative = 0.03  # Scénario conservateur (3%)

"""
import yfinance
import numpy as np
import sys
from pathlib import Path
base = Path(__file__).resolve().parent.parent.parent
sys.path.append( str(base) )
import helper as h
from user_scripts.api import api

separator = "-----------------------------------------------------------"

# -- Paramètres fixes ----------------------------------
market_return = 0.08  # Rendement attendu du marché
tax_rate = 0.30       # Taux d'imposition supposé de 30%
years = 5             # Période de projection (années)
growth_rate_conservative = 0.03  # Scénario conservateur
MAX_GROWTH_RATE = 0.15           # Plafond scénario optimiste

# -- Vérification paramètres ----------------------------------
if not api.check_parameters( ['ticker'] ):
    exit(1)

stock = yfinance.Ticker( api.ticker )

# -- Données de base ----------------------------------
stock_info = stock.info.copy()
short_name = stock_info['shortName']
industry_key = stock_info['industryKey']
shares_outstanding = stock_info['sharesOutstanding']
market_cap = stock_info['marketCap']
current_price = stock.history(period='1d')['Close'].iloc[0]

print( separator )
print( 'Estimation de la Juste Valeur - Méthode Free Cash Flow (FCF)' )
print( separator )
print( f"Société       : {short_name}" )
print( f"Secteur       : {industry_key}" )
print( f"Actions       : {h.format_number( shares_outstanding, 1e+12 )}" )
print( f"Prix actuel   : {current_price:.3f}" )
print( f"Market Cap    : {h.format_number( market_cap, 1e+12 )}" )
print( separator )

# -- Free Cash Flow : sélection automatique du premier FCF positif -----------
cash_flow = stock.cashflow.copy()

if 'Free Cash Flow' not in cash_flow.index:
    print("=> ERREUR : 'Free Cash Flow' absent des données yfinance pour ce ticker.")
    exit(1)

fcf_series = cash_flow.loc['Free Cash Flow'].dropna()
print( "Free Cash Flow historique :" )
print( fcf_series )
print( separator )

positive_fcf = fcf_series[fcf_series > 0]
if positive_fcf.empty:
    print("=> ERREUR : Aucun FCF positif disponible, calcul impossible.")
    exit(1)

fcf = positive_fcf.iloc[0]
fcf_year = positive_fcf.index[0].year
print( f"FCF retenu ({fcf_year}) : {h.format_number( fcf, 1e+12 )}" )
print( separator )

# -- Croissance historique FCF (scénario optimiste) -----------
if len(fcf_series) >= 2:
    fcf_oldest = fcf_series.iloc[-1]
    n_years = len(fcf_series) - 1
    if fcf_oldest > 0:
        historical_growth = (fcf / fcf_oldest) ** (1 / n_years) - 1
        growth_rate_optimistic = min(historical_growth, MAX_GROWTH_RATE)
        print( f"Croissance FCF historique ({n_years} ans) : {historical_growth:.1%}" )
        print( f"Scénario optimiste plafonné à           : {growth_rate_optimistic:.1%}" )
    else:
        growth_rate_optimistic = growth_rate_conservative
        print( f"=> FCF le plus ancien négatif, scénario optimiste = conservateur ({growth_rate_conservative:.1%})" )
else:
    growth_rate_optimistic = growth_rate_conservative
    print( f"=> Données FCF insuffisantes, scénario optimiste = conservateur ({growth_rate_conservative:.1%})" )
print( separator )

# -- Beta -----------
if 'beta' in stock_info and stock_info['beta'] is not None:
    beta = stock_info['beta']
    print( f"Beta (volatilité) : {beta}" )
else:
    beta = 1.0
    print( f"=> Beta absent, valeur par défaut : {beta} (équivalent marché)" )
print( separator )

# -- Bilan : dette et capitaux propres -----------
stock_balance_sheet = stock.balance_sheet.copy()

if 'Long Term Debt' in stock_balance_sheet.index:
    total_debt = stock_balance_sheet.loc['Long Term Debt'].values[0]
elif 'Long Term Debt And Capital Lease Obligation' in stock_balance_sheet.index:
    total_debt = stock_balance_sheet.loc['Long Term Debt And Capital Lease Obligation'].values[0]
    print( f"=> Dette : Long Term Debt And Capital Lease Obligation utilisé : {h.format_number(total_debt)}" )
    print( separator )
else:
    print( '=> ERREUR : Aucune donnée de dette long terme disponible.' )
    exit(1)

total_equity = stock_balance_sheet.loc['Stockholders Equity'].values[0]

if total_equity <= 0:
    print( f"=> AVERTISSEMENT : Stockholders Equity négatif ({h.format_number(total_equity)}), WACC peu fiable." )

print( f"Long Term Debt      : {h.format_number( total_debt, 1e+12 )}" )
print( f"Stockholders Equity : {h.format_number( total_equity, 1e+12 )}" )
print( separator )

# -- Taux sans risque : 10Y US Treasury dynamique -----------
try:
    treasury = yfinance.Ticker("^TNX")
    tnx_price = treasury.history(period='5d')['Close'].dropna()
    if not tnx_price.empty:
        risk_free_rate = tnx_price.iloc[-1] / 100
        print( f"Taux sans risque (10Y Treasury, dynamique) : {risk_free_rate:.2%}" )
    else:
        raise ValueError("Données TNX vides")
except Exception:
    risk_free_rate = 0.02
    print( f"=> Taux sans risque (défaut, TNX indisponible) : {risk_free_rate:.2%}" )
print( separator )

# -- Coût de la dette : calculé depuis Interest Expense si disponible -----------
stock_financials = stock.financials.copy()
cost_of_debt = None

if 'Interest Expense' in stock_financials.index and total_debt > 0:
    interest_expense = abs(stock_financials.loc['Interest Expense'].values[0])
    if interest_expense > 0:
        cost_of_debt = interest_expense / total_debt
        print( f"Coût de la dette (calculé via Interest Expense) : {cost_of_debt:.2%}" )

if cost_of_debt is None:
    cost_of_debt = 0.03
    print( f"Coût de la dette (défaut) : {cost_of_debt:.2%}" )
print( separator )

# -- WACC -----------
cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)

total_capital = total_debt + total_equity
debt_ratio = total_debt / total_capital
equity_ratio = total_equity / total_capital

wacc = (equity_ratio * cost_of_equity) + (debt_ratio * cost_of_debt * (1 - tax_rate))

print( f"Coût des capitaux propres (CAPM) : {cost_of_equity:.2%}" )
print( f"WACC                             : {wacc:.2%}" )
print( separator )

# -- Fonction de calcul DCF -----------
def _compute_fair_price(base_fcf, growth_rate, wacc, years, shares):
    """Calcule la juste valeur par action via DCF + valeur terminale."""
    if wacc <= growth_rate:
        return None, None, None  # invalide

    future_fcfs = [base_fcf * (1 + growth_rate) ** i for i in range(1, years + 1)]
    terminal_value = future_fcfs[-1] * (1 + growth_rate) / (wacc - growth_rate)

    discounted = [fcf_val / (1 + wacc) ** (i + 1) for i, fcf_val in enumerate(future_fcfs)]
    terminal_value_discounted = terminal_value / (1 + wacc) ** years

    enterprise_value = np.sum(discounted) + terminal_value_discounted
    fair_price = enterprise_value / shares
    return enterprise_value, fair_price, terminal_value_discounted

# -- Résultats -----------
theo_price = market_cap / shares_outstanding

print( f"Prix actuel   : {current_price:.3f}" )
print( f"Prix théorique (market cap / actions) : {theo_price:.3f}" )
print( separator )

for label, growth_rate in [
    ("Conservateur", growth_rate_conservative),
    ("Optimiste   ", growth_rate_optimistic),
]:
    ev, fair_price, tv_disc = _compute_fair_price(
        fcf, growth_rate, wacc, years, shares_outstanding
    )

    if ev is None:
        print( f"Scénario {label} (g={growth_rate:.1%}) : INVALIDE - growth_rate >= WACC ({wacc:.2%})" )
        continue

    upside = (fair_price - current_price) / current_price * 100
    direction = "SOUS-ÉVALUÉ" if upside > 0 else "SUR-ÉVALUÉ"

    print( f"-- Scénario {label} (g={growth_rate:.1%}) --" )
    print( f"   Valeur terminale actualisée  : {h.format_number( tv_disc, 1e+12 )}" )
    print( f"   Enterprise Value estimée     : {h.format_number( ev, 1e+12 )}" )
    print( f"   Juste valeur par action      : {fair_price:.3f}" )
    print( f"   Upside / Downside            : {upside:+.1f}% DIRECTION : {direction}" )
    print( separator )