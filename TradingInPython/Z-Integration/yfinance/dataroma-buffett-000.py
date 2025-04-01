""" Retrouver les actions tradées par Warren Buffett - Berkshire Hathaway sur le Site DATAROMA
    Fabriquer un indice Buffett

    - returnOnEquity = Résultat Net / Capitaux propres
        - ROE élevé signifie que l'entreprise génère un bon rendement sur l'argent investi par ses actionnaires.
        - ROE faible peut indiquer une inefficacité dans l'utilisation des capitaux propres.
        - ROE négatif indique des pertes.
    
    - priceEarningsRatio = Prix de l'action / Bénéfices par actions (EPS)
        - PER élevé peut indiquer que les investisseurs s'attendent à une forte croissance future des bénéfices et sont donc prêts à payer un prix élevé pour l'action.
        - PER faible peut suggérer que l'entreprise est sous-évaluée, ou bien qu'elle rencontre des difficultés financières.
        - PER très bas pourrait aussi indiquer que l'entreprise est en difficulté ou dans un secteur en déclin.
    
    - priceToBookRatio = Cours de l'action / Valeur Comptable par Action (Book Value per Share BVPS)
        - P/B > 1 : L'entreprise est valorisée au-dessus de sa valeur comptable. Cela peut indiquer une forte rentabilité ou une anticipation de croissance future.
        - P/B < 1 : L'entreprise est sous-évaluée par rapport à ses actifs, ce qui peut être une opportunité d'investissement ou le signe d'un problème structurel.
        - P/B ≈ 1 : L'entreprise est valorisée proche de sa valeur comptable.
"""
import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import yfinance

def get_berkshire_holdings():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    }
    url = "https://www.dataroma.com/m/holdings.php?m=BRK"
    response = requests.get( url, headers=headers )
    if response.status_code == 200:
        soup = BeautifulSoup( response.text, 'html.parser' )
        symbols = []
        for row in soup.select("#grid tr")[1:]:  # ignorer l'en-tête
            cols = row.find_all("td")
            if cols:
                s = cols[1].text.strip()
                print( s )
                symbols.append( s )  # champ = symbole
        return symbols
    else:
        print( f"get_berkshire_holdings: status code: {response.status_code}: {response.text}" )
        return []

def get_financial_ratios( symbol ):
    s = symbol.split(" - ")[0]
    try:
        stock = yfinance.Ticker( s )
        info = stock.info.copy()
    except Exception as e:
        print(f'Erreur avec le symbol {s}: {e}')
        return None
    
    ratios = {
        "name" : symbol,
        "returnOnEquity": info.get("returnOnEquity", 0),
        "priceEarningsRatio": info.get("trailingPE", 100),
        "priceToBookRatio": info.get("priceToBook", 10)
    }
    return ratios

def filter_buffett_stocks( symbols, roe=0.1, per=30, pbr=10 ):
    selected_stocks = []
    finances = []
    for symbol in symbols:
        ratios = get_financial_ratios( symbol )
        finances.append( ratios )
        if ratios:
            _roe = ratios.get( 'returnOnEquity', 0 )
            _per = ratios.get( 'priceEarningsRatio', 100 )
            _pbr = ratios.get( 'priceToBookRatio', 10 )
            if _roe > roe and _per < per and _pbr < pbr:
                selected_stocks.append( symbol )
    return selected_stocks, finances

def get_historical_prices( symbols ):
    prices = {}
    for symbol in symbols:
        s = symbol.split(" - ")[0]
        stock = yfinance.Ticker( s )
        data = stock.history( period="max" )  # Récupérer toutes les données historiques
        if not data.empty:
            prices[ symbol ] = data['Close']
        else:
            print(f"Aucune donnée pour {symbol}")
    return pd.DataFrame(prices)

# Nomarlised signal upon 'n'
#
def normalise( data, n ):
    norm = (data - data.min()) / (data.max() - data.min()) * n
    return norm
    
historical_sp500 = None

def plot_buffett_index( price_data ):
    buffett_indice = price_data.mean( axis=1 ) # moyenne de toutes les actions
    historical_sp500_close = pd.DataFrame( historical_sp500['Close'] )
    
    historical_sp500_close_norm = normalise( historical_sp500_close, 1000 )
    buffett_indice_norm = normalise( buffett_indice, 1000 )
    
    plt.plot( figsize=(30, 15) )
    plt.plot( buffett_indice_norm.index, buffett_indice_norm.values, color='blue', label="Buffett" )
    plt.plot( historical_sp500_close_norm.index, historical_sp500_close_norm.values, color="green", label="S&P500" )
    
    plt.title( "Buffett Indice vs S&P500" )
    plt.legend()
    plt.grid( True )
    # plt.xlabel("Date")
    # plt.ylabel("Index Value")
    plt.show()

def main():
    global historical_sp500
    
    # Extraire les symbols de la page HTML
    symbols = get_berkshire_holdings()
    if not symbols:
        print("Impossible de récupérer les données de Berkshire Hathaway")
        return

    # Filtrer les symbols en fonction de leurs données financières
    selected_stocks, finances = filter_buffett_stocks( symbols )
    
    # Options pour afficher tout le 'df'
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    # Afficher les résultats
    df_finances = [item for item in finances if item is not None]
    df = pd.DataFrame( df_finances )
    df_sorted = df.sort_values(by="returnOnEquity", ascending=False)
    print( df_sorted )
    
    print( "Actions sélectionnées selon les critères Buffett:", selected_stocks )
    
    price_data = get_historical_prices( selected_stocks )
    if price_data.empty:
        print("Aucune donnée de prix disponible.")
        return
    
    sp500 = yfinance.Ticker( "^GSPC" ) # indice S&P 500
    historical_sp500 = sp500.history( period="max" )
    
    plot_buffett_index( price_data )

if __name__ == "__main__":
    main()
