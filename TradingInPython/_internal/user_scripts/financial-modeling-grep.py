""" Tool_New - Financial Modeling Prep

    API-REST depuis https://financialmodelingprep.com
    
"""
import requests
from tabulate import tabulate

# Remplacer par ta clé API
API_KEY = 'YOUR_API_KEY_HERE'

# URL de l'API pour obtenir les entreprises par secteur (exemple : Engineering & Construction)
url = f'https://financialmodelingprep.com/api/v3/sector-performance?apikey={API_KEY}'

# Requête GET pour récupérer les données
response = requests.get(url)
data = response.json()

if response.status_code != 200:
    print("Erreur de récupération des données")
    exit()

print( data )

# Transformation en liste de listes pour tabulate
table = [[d['sector'], d['changesPercentage']] for d in data]

# Affichage avec tabulate
print( tabulate( table, headers=['Sector', 'Change (%)'], tablefmt='pretty' ) )
  
# Affichage des données récupérées
# if response.status_code == 200:
#     for sector in data:
#         print(f"Sector: {sector['sector']}, Change: {sector['changesPercentage']}")
# else:
#     print("Erreur de récupération des données")

# +------------------------+------------+
# |         Sector         | Change (%) |
# +------------------------+------------+
# |       Materials        | -1.88952%  |
# | Communication Services |  0.6939%   |
# |   Consumer Cyclical    | -0.20717%  |
# |   Consumer Defensive   | -0.59142%  |
# |         Energy         |  0.50746%  |
# |       Financials       | -1.15647%  |
# |      Health Care       | -1.64203%  |
# |      Industrials       | -2.03524%  |
# |      Real Estate       | -0.52564%  |
# | Information Technology | -0.61166%  |
# |       Utilities        | -0.37184%  |
# +------------------------+------------+
#
# Sector: Materials, Change: -1.88952%
# Sector: Communication Services, Change: 0.6939%
# Sector: Consumer Cyclical, Change: -0.20717%
# Sector: Consumer Defensive, Change: -0.59142%
# Sector: Energy, Change: 0.50746%
# Sector: Financials, Change: -1.15647%
# Sector: Health Care, Change: -1.64203%
# Sector: Industrials, Change: -2.03524%
# Sector: Real Estate, Change: -0.52564%
# Sector: Information Technology, Change: -0.61166%
# Sector: Utilities, Change: -0.37184%