""" Tool_Finance - Trailing EPS Calcul du PER en comparant au marché

	EPS - Earnings Per Share (ou Bénéfice Par Action (BPA) in french) : Il s'agit du bénéfice net de l'entreprise divisé 
 	par le nombre d'actions en circulation.

 	Il donne une idée du montant des bénéfices que chaque action génère pour les actionnaires.
  	
   	PER - Price Earnings Ratio : Prix actuel de l'action divisé par le bénéfice par action
    
	PER Trailing < PER Forward : 
 	Si le PER forward est plus élevé que le PER trailing, cela peut indiquer que les bénéfices futurs sont attendus à la baisse,
  	ou que le marché est optimiste quant aux perspectives de croissance malgré des bénéfices récents faibles.

	PER Trailing > PER Forward :
 	Si le PER Forward est plus faible, cela peut suggérer que les bénéfices futurs sont attendus en augmentation, ce qui peut 
  	indiquer un potentiel de croissance pour l'entreprise.
"""
import yfinance
import numpy as np

print_separator1 = f"----------------------------------"
print_separator2 = f"----------------------------------------------------------------------------"

# --------------------------------------------------------------------------------------
# Choix Utilisateur
# --------------------------------------------------------------------------------------
#
# Informations sur l'action à étudier dans son marché
#

#
# Estimation du PER pour l'industrie
# Trouver des stock dans un marché identique
#
#ticker = 'BEN.PA' # BENETEAU
#industry_symbols = [ 'TRI.PA', 'CATG.PA', '9638.HK', 'ALFPC.PA', 'BELL.MI'] # BENETEAU

# ticker = 'CAP.PA'  # CAPGEMINI
# industry_symbols = [ 'ATE.PA', 'SOP.PA', 'ACN'  ] # CAPGEMINI Non évalué : 'AL2SI.PA' 'D6H.DE'

# Sector: Industrials Industry key: aerospace-defense Aéronautique et défense
ticker = 'SAF.PA' # SAFRAN
industry_symbols = ['HO.PA', 'AM.PA', 'AIR.PA', 'GE'  ] # THALES - DASSAULT AVIATION - AIR BUS - GE aerospace

#ticker = 'ASY.PA' # ASSYSTEM
#industry_symbols = [ 'EN.PA', 'FGR.PA', 'TKTT.PA', 'ALPJT.PA'] # EN.PA Bouygues - FGR.PA Eiffage - TKTT.PA Tarkett - ALPJT.PA Poujoulat

# --------------------------------------------------------------------------------------

stock = yfinance.Ticker( ticker )
stock_info = stock.info.copy() # ne faire qu'une requête et sauver tous les résultats
if len( stock_info ) == 1: # astuce pour savoir que stock_info est vide ... l'objet est créé mais il n'y a rien dedans
    print( f"Le symbol {ticker} n'existe pas")
    exit()
    
# --------------------------------------------------------------------------------------

short_name = stock_info['shortName']
industry_key = stock_info['industryKey']
eps_trailing = stock_info['trailingEps']  # EPS sur les 12 derniers mois
eps_forward = stock_info['forwardEps']

current_price = stock.history(period='1d')['Close'].iloc[0]

print( print_separator1 )
print( f"   Short Name: {short_name}" )
print( f" Industry Key: {industry_key}" )
print( f"Current Price: {current_price:.2f}" )
print( print_separator1 )
print( f"EPS Trailing: {eps_trailing} Forward: {eps_forward}" )
print( f"PER Trailing: {current_price/eps_trailing:.3f} Forward: {current_price/eps_forward:.3f}" )
print( print_separator1 )

# Liste pour stocker les PER trailing des entreprises comparables
per_trailing_values = []
per_forward_values = []

# Market evaluation
#
for symbol in industry_symbols:
	stock = yfinance.Ticker( symbol )

	stock_info = stock.info
	short_name = stock_info['shortName']
	if 'trailingEps' in stock_info:
		_eps_trail = stock_info['trailingEps']  # EPS sur les 12 derniers mois
	else:
		_eps_trail = 0.0
	if 'forwardEps' in stock_info:
		_eps_forwd = stock_info['forwardEps']  # EPS du futur
	else:
		_eps_forwd = 0.0

	_cur_price = stock.history(period='1d')['Close'].iloc[0] # dernière valeur de cotation
	
 	# Calculer le PER si le trailing EPS est disponible
	_p2 = ""
	if _eps_trail and _eps_trail > 0.5: # seuil d'exclusion du calcul
		_per_trailing = _cur_price / _eps_trail
		per_trailing_values.append( _per_trailing )
	else:
		_per_trailing = 0.0
		_p2 += " <<< PER TRAIL"
  	
 	# Calculer le PER si le forward EPS est disponible et supérieur à un seuil
	if _eps_forwd and _eps_forwd > 0.5: # seuil d'exclusion du calcul
		_per_forward = _cur_price / _eps_forwd
		per_forward_values.append( _per_forward )
	else:
		_per_forward = 0.0
		_p2 += " <<< PER FORWRD"
  
	if _p2 != "":
		_p2 += "<<< NON PRIS EN COMPTE >>>"
   
	_p1 = f"{short_name}: {_cur_price:.3f} € - EPS Trailing: {_eps_trail} Forward: {_eps_forwd} - PER Trailing: {_per_trailing:.3f} Forward: {_per_forward:.3f}"
	print( _p1 + _p2 )
	
	print( print_separator2 )

# Calculer le PER trailing moyen de cette industrie
#
if per_trailing_values:
    per_trailing_value = np.mean( per_trailing_values )
    print(f"PER Trailing estimé pour cette industrie est de : {per_trailing_value:.2f}")
else:
    print("Impossible de calculer le PER trailing moyen, données insuffisantes.")

# Calculer le PER forward moyen de l'industrie
if per_forward_values:
    per_forward_value = np.mean( per_forward_values )
    print(f"PER Forward estimé pour cette industrie est de : {per_forward_value:.2f}")
else:
    print("Impossible de calculer le PER forward moyen, données insuffisantes.")

print( print_separator2 )

# Calcul de la juste valeur de l'action
#
juste_valeur_trailing = eps_trailing * per_trailing_value
juste_valeur_forward = eps_forward * per_forward_value

print(f"La juste valeur Trailing de l'action est estimée à : {juste_valeur_trailing:.2f} €")
print(f"La juste valeur Forward de l'action est estimée à : {juste_valeur_forward:.2f} €")

print( print_separator2 )