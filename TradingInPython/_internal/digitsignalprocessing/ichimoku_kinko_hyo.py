""" Puisqu'il y a un Ichimoku moderne ...

	L'indicateur Ichimoku Kinko Hyo modernisé repose sur une combinaison de plusieurs éléments de l'indicateur Ichimoku traditionnel, 
 	ajustés par l'Average True Range (ATR) pour tenir compte de la volatilité.

	1. Signal d'Achat (Buy Signal)
    ------------------------------
	Condition :
		La ligne Tenkan-sen (ligne de conversion) traverse au-dessus de la Kijun-sen (ligne de base).
		Le prix de clôture se situe au-dessus des deux limites du nuage Ichimoku (Senkou Span A et Senkou Span B).
	Interprétation :
		Cela indique une force haussière dans le marché. Un croisement haussier entre Tenkan-sen et Kijun-sen, accompagné de la position du prix au-dessus du nuage, 
        suggère que les acheteurs contrôlent le marché. 
        L'ajout de bandes basées sur l'ATR permet d'estimer la volatilité et d'ajuster la probabilité d'un mouvement continu à la hausse.
	Action :
		Les traders peuvent envisager d'entrer en position longue (achat) ou de maintenir des positions existantes si ces conditions sont remplies.
	
 	2. Signal de Vente (Sell Signal)
    --------------------------------
	Condition :
		La ligne Tenkan-sen traverse en dessous de la Kijun-sen.
		Le prix de clôture se situe en dessous des deux limites du nuage Ichimoku (Senkou Span A et Senkou Span B).
	Interprétation :
		Ce signal indique une faiblesse du marché et une possible tendance baissière. Le croisement baissier entre Tenkan-sen et Kijun-sen, en conjonction avec un prix sous le nuage,
        suggère que les vendeurs dominent. 
        L'ajustement avec l'ATR donne une idée de la volatilité, ce qui est utile pour évaluer la solidité de la tendance baissière.
	Action :
		Les traders peuvent envisager d'entrer en position courte (vente) ou de sortir des positions longues existantes.
  
    La force de la tendance est plus importante si ces signaux se produisent lorsque le prix est bien au-dessus (pour une tendance haussière) 
    ou bien en-dessous (pour une tendance baissière) du nuage.
    
	3. Zones Neutres ou Indécises
    -----------------------------
	Condition :
		Le prix est à l'intérieur du nuage Ichimoku.
		Les croisements entre Tenkan-sen et Kijun-sen sont peu clairs ou répétés sur une courte période.
	Interprétation :
		Lorsque le prix est dans le nuage, cela indique une zone d'indécision où ni les acheteurs ni les vendeurs ne prennent le contrôle. Le marché 
  		peut être dans une phase de consolidation ou de transition vers une nouvelle tendance.
	Action :
		Les traders peuvent choisir de rester en dehors du marché jusqu'à ce qu'une direction plus claire émerge. 
  		Les ordres peuvent être placés à l'extérieur du nuage pour capter un éventuel breakout.
    
	4. Signal de Confirmation et Volatilité (via l'ATR et Kijun_sen_upper et Kijun_sen_lower)
    -----------------------------------------------------------------------------------------
	Condition :
		Les bandes basées sur l'ATR (Kijun_sen_upper et Kijun_sen_lower) peuvent être utilisées pour confirmer les signaux ou pour gérer les risques.
	Interprétation :
		Si le prix s'éloigne fortement des bandes, cela pourrait indiquer un mouvement survolatilité. Les traders pourraient interpréter cela
  		comme un signal d'épuisement de tendance ou utiliser l'information pour ajuster leur stop-loss.
	Action :
		Ajuster les positions ou les ordres stop en fonction de l'élargissement ou du rétrécissement de ces bandes, en tenant compte de la volatilité actuelle du marché.

        - Kijun_sen_upper : Cette bande supérieure représente une résistance dynamique. 
        Plus la volatilité (ATR) est élevée, plus cette bande sera éloignée de la Kijun-sen. 
        Lorsque le prix atteint cette bande, cela peut indiquer une zone de surachat ou un niveau de résistance 
        où le prix pourrait se renverser ou corriger à la baisse.

        - Kijun_sen_lower : Cette bande inférieure représente un support dynamique. 
        Si le prix descend jusqu'à cette bande, cela peut indiquer une zone de survente ou un niveau de support
        où le prix pourrait rebondir à la hausse.

    Volatilité du Marché :
        Lorsque les bandes sont larges (c'est-à-dire que l'ATR est élevé), cela indique que le marché est volatile. Une telle situation suggère de la prudence, 
        car des mouvements de prix plus importants peuvent se produire.
        Lorsque les bandes se rapprochent, cela indique une baisse de la volatilité, ce qui peut suggérer un marché plus stable.

    Points de Renversement Potentiels :
        Si le prix dépasse la bande supérieure (Kijun_sen_upper), cela peut signaler une condition de surachat, et une correction à la baisse pourrait suivre.
        Si le prix descend en dessous de la bande inférieure (Kijun_sen_lower), cela peut indiquer une condition de survente, et un rebond à la hausse pourrait suivre.
    
    Confirmation de Tendance :
        Tant que le prix évolue entre les bandes, cela confirme la tendance actuelle.
        Si le prix casse les bandes, cela peut être un signal de changement de tendance ou d'une forte accélération du mouvement.        

    5. Kumo (nuage)
    ---------------
    Interprétation : 
        Le nuage est l'élément central de l'indicateur Ichimoku. Il représente la zone de support et de résistance. 
        Plus le nuage est épais, plus le support ou la résistance est fort. 
        Lorsque le prix est au-dessus du nuage, la tendance est considérée comme haussière, et lorsqu'il est en dessous, la tendance est baissière. 
        Si le prix est à l'intérieur du nuage, cela indique une tendance indécise ou neutre.
        
        - Twist du nuage : c'est quand Senkou Span A passe en dessous ou au dessus de Senkou Span B
        Lorsque le nuage Ichimoku twist après une tendance cela indique un affaiblisement de la tendance et que le prix entre en range. Il est alors peut-être temps 
        de prendre ses bénéfices.
        
 	6. Ligne Chikou Span ligne retardée (Lagging Span)
    --------------------------------------------------
	Condition :
		La Chikou Span est souvent utilisée comme confirmation. Elle montre le prix actuel projeté en arrière de 26 périodes.
	Interprétation :
		Si Chikou Span est au-dessus du prix actuel, cela confirme une tendance haussière. Si elle est en dessous, cela confirme une tendance baissière.
	Action :
	    Utiliser Chikou Span pour confirmer les signaux d'achat ou de vente, ou pour éviter de prendre des positions contre la tendance dominante.

	7. Philosophie de l'indicateur
    ------------------------------
    L'Ichimoku se base sur une analyse visuelle et la théorie de la tendance. 
 	L'indicateur suppose que les prix reflètent déjà toutes les informations disponibles sur le marché, 
  	y compris les volumes, même si ce n’est pas explicitement pris en compte.

	- Avantages et limitations sans les volumes
		Avantages :
			Simplicité : L'absence de données sur les volumes rend l'Ichimoku plus simple à utiliser. Les traders n'ont pas à analyser des 
   			données supplémentaires, ce qui peut rendre la prise de décision plus rapide et moins encombrée.

			Identification des tendances : Il est particulièrement efficace pour identifier les tendances, les niveaux de support et de résistance, 
   			et pour générer des signaux de trading dans des conditions de marché variées.

		Limitations :
			Absence de confirmation par le volume : Le volume peut souvent fournir des indices importants sur la force ou la faiblesse d'une tendance. 
   			Par exemple, un volume élevé pendant une tendance haussière peut confirmer la solidité de cette tendance, 
      		tandis qu'un volume faible pourrait indiquer une faiblesse ou un manque d'engagement du marché.

			Difficulté à détecter les retournements : Les volumes peuvent aider à repérer les retournements de tendance en montrant si les mouvements 
   			de prix sont soutenus par une participation significative. Sans ces données, certains retournements peuvent être plus difficiles à identifier.		
      	
 	8. Conclusion
    -------------
		L'interprétation des signaux dans l'Ichimoku modernisé doit tenir compte de plusieurs facteurs simultanément, notamment les croisements de lignes, 
  		la position par rapport au nuage, et les ajustements de volatilité via l'ATR. En combinant ces éléments, les traders peuvent obtenir une vue d'ensemble 
    	plus complète de la situation du marché, ce qui peut aider à prendre des décisions plus éclairées sur les entrées, sorties et la gestion des risques.
"""
import warnings
import numpy as np
import pandas
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .indicators import atr

# 1. Calculer les composantes de l'Ichimoku avec l'ATR
# ----------------------------------------------------
# - True Range (TR)
# TR = max( ∣High − Low∣, ∣High − Previous Close∣, ∣Low − Previous Close∣ )
#
# L'idée est de capturer les écarts de prix les plus importants, 
# en tenant compte non seulement de la plage intra-journalière (High - Low), 
# mais aussi des gaps d'ouverture par rapport à la clôture de la période précédente.
#
# - ATR : Moyenne mobile sur un certain nombre de périodes
#
def ichimoku_modernise( data, period1=9, period2=26, period3=52, multiplier=1.5 ):
    # Tenkan-sen (ligne de conversion)
    data['Tenkan_sen'] = (data['High'].rolling(window=period1).max() + data['Low'].rolling(window=period1).min()) / 2

    # Kijun-sen (ligne de base)
    data['Kijun_sen'] = (data['High'].rolling(window=period2).max() + data['Low'].rolling(window=period2).min()) / 2

    # Senkou Span A (première limite du nuage)
    data['Senkou_span_A'] = ((data['Tenkan_sen'] + data['Kijun_sen']) / 2).shift(period2)

    # Senkou Span B (deuxième limite du nuage)
    data['Senkou_span_B'] = ((data['High'].rolling(window=period3).max() + data['Low'].rolling(window=period3).min()) / 2).shift(period2)

    # Chikou Span (ligne retardée)
    data['Chikou_span'] = data['Close'].shift(-period2)

    # Average True Range (ATR)
    data['ATR'] = atr( data, period2 )
    
    # Ajouter des bandes basées sur l'ATR
    data['Kijun_sen_upper'] = data['Kijun_sen'] + (data['ATR'] * multiplier)
    data['Kijun_sen_lower'] = data['Kijun_sen'] - (data['ATR'] * multiplier)
    
    return data

# 2. Générer des signaux basés sur Ichimoku et entraîner un modèle de régression logistique
#
def generate_signals( data ):
    data['Signal'] = 0

    # Conditions pour les signaux d'achat
    data.loc[(data['Tenkan_sen'] > data['Kijun_sen']) & (data['Close'] > data['Senkou_span_A']) & (data['Close'] > data['Senkou_span_B']), 'Signal'] = 1

    # Conditions pour les signaux de vente
    data.loc[(data['Tenkan_sen'] < data['Kijun_sen']) & (data['Close'] < data['Senkou_span_A']) & (data['Close'] < data['Senkou_span_B']), 'Signal'] = -1
    
    # Pour afficher ce signal en bas du graph
    mean = data['Close'].mean()
    data['Signal_display'] = data['Signal'] + mean
    
    return data

def train_predictive_model_0000(data):
    # On crée des features pour entraîner le modèle
    X = data[['Tenkan_sen', 'Kijun_sen', 'Senkou_span_A', 'Senkou_span_B', 'ATR', 'Kijun_sen_upper', 'Kijun_sen_lower']]
    y = data['Signal']
    
    # On élimine les lignes où le signal n'est pas défini
    X = X.dropna()
    y = y[ X.index ]

    if X.shape[0] == 0:
        raise ValueError("Pas assez de données pour entraîner le modèle après suppression des NaN.")
    
    if X.shape[0] < 10:
        raise ValueError("Pas assez de données pour diviser en ensembles d'entraînement et de test.")
    
    # Split les données en jeu d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Créer et entraîner le modèle
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Prédictions et évaluation
    y_pred = model.predict( X_test )
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Précision du modèle: {accuracy:.2f}")
    
    return model

# La variable y = data['Signal'] est ce que le modèle de machine learning essaie de prédire.
# Le modèle de machine learning, par exemple LogisticRegression, est un modèle supervisé. 
# Cela signifie qu'il apprend en observant les relations entre les variables 
# d'entrée (X - les caractéristiques ou features) 
# et la variable cible (y - les signaux d'achat/vente).
#
def train_predictive_model( data ):
    global y_test, y_pred
    
    # On crée des features pour entraîner le modèle
    X = data[['Tenkan_sen', 'Kijun_sen', 'Senkou_span_A', 'Senkou_span_B', 'ATR', 'Kijun_sen_upper', 'Kijun_sen_lower']]
    y = data['Signal']
    
    # Vérification des NaN
    X = X.dropna()
    y = y[ X.index ]
    
    if X.shape[0] == 0:
        raise ValueError("Pas assez de données pour entraîner le modèle après suppression des NaN.")
    
    if X.shape[0] < 10:
        raise ValueError("Pas assez de données pour diviser en ensembles d'entraînement et de test.")
    
    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reconvertir en DataFrame pour conserver les noms des colonnes
    X_scaled_df = pandas.DataFrame( X_scaled, columns=X.columns, index=X.index )
    
    # Split les données en jeu d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split( X_scaled_df, y, test_size=0.3, random_state=42 )
    
    # Créer et entraîner le modèle avec un nombre d'itérations plus élevé et en capturant les avertissements
    with warnings.catch_warnings():
        warnings.simplefilter( "ignore", category=ConvergenceWarning )
        model = LogisticRegression( max_iter=1000, solver='lbfgs')  # Augmentation de max_iter à 1000 autres solveurs 'liblinear', 'sag', 'saga'
        model.fit( X_train, y_train )
    
    # Prédictions et évaluation
    y_pred = model.predict( X_test )
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Précision du modèle: {accuracy:.2f}")
    
    return model

# Appliquer le modèle à de nouvelles données pour obtenir des signaux prédictifs
# franchement ... je ne vois pas à quoi sert cette fontion
def apply_model( model, data ):
    
    X_new = data[['Tenkan_sen', 'Kijun_sen', 'Senkou_span_A', 'Senkou_span_B', 'ATR', 'Kijun_sen_upper', 'Kijun_sen_lower']].dropna()
    data['Signal_Forcasted'] = np.nan
    data.loc[ X_new.index, 'Signal_Forcasted'] = model.predict( X_new )

    # Pourvoir afficher sur le graphique
    mean = data['Close'].mean()
    data['Signal_Forcasted'] = data['Signal_Forcasted'] + mean
        
    return data

def plot_forcasts( df, y_test, y_pred ):
    # Créer une colonne dans le DataFrame pour les prédictions, initialement avec des NaN
    df['Forcasts'] = float('nan')
    
    # Aligner les prédictions avec les indices dans df
    df.loc[ y_test.index, 'Forcasts'] = y_pred

    # Tracer le Kijun-sen
    #plt.plot( df['Kijun_sen'], label='Kijun-sen')

    # Tracer les prédictions d'achat et de vente
    plt.scatter( df.index, df['Forcasts'], label='Forcasts', marker='o', color='red')

def plot_future_predictions(df, y_pred):
    # Dernier index de la série historique
    last_index = df.index[-1]
    
    # Calcul du nombre de prédictions futures
    future_indices = pandas.date_range( start=last_index, periods=len( y_pred ) + 1, freq='D')[1:]
    
    # Créer une série pour les prédictions futures
    future_predictions = pandas.Series( y_pred, index=future_indices )
    
    # Tracer le Kijun-sen
    #plt.plot( df.index, df['Kijun_sen'], label='Kijun-sen' )
    
    # Tracer les prédictions futures
    plt.plot( future_predictions.index, future_predictions, label='Prédictions Futures', marker='o', linestyle='--', color='red')

def predict_future_signals(df, model, days_in_future, days_in_past):
    """
    Prédire les signaux futurs en utilisant un modèle basé sur des fenêtres de données passées.

    Parameters:
    df (pd.DataFrame): DataFrame contenant les données historiques.
    model: Modèle de machine learning entraîné.
    days_in_future (int): Nombre de jours pour lesquels prédire les signaux futurs.
    days_in_past (int): Nombre de jours de données passées à utiliser pour chaque prédiction.

    Returns:
    np.array: Prédictions des signaux pour les jours futurs.
    """
    
    # Assumer que df a une colonne 'Signal' qui est l'étiquette à prédire
    if 'Signal' not in df.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'Signal'.")

    # Ne prendre que les 7 valeurs de création du model
    features = ['Tenkan_sen', 'Kijun_sen', 'Senkou_span_A', 'Senkou_span_B', 'ATR', 'Kijun_sen_upper', 'Kijun_sen_lower']
    _df = df[ features ].copy()
    _df.fillna( 0, inplace=True )
    
    # Normaliser les données historiques
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform( _df )
    
    # Préparer les fenêtres de données passées pour les prédictions
    X = np.array([data_scaled[i:i + days_in_past] for i in range(len(data_scaled) - days_in_past)])
    
    # Assurer que le modèle s'attend à la bonne forme des données
    X = np.reshape(X, (X.shape[0], days_in_past, X.shape[2]))
    
    # Utiliser la dernière fenêtre de données pour commencer les prédictions futures
    last_window = pandas.DataFrame(X[-1], columns=features)

    predictions = []
    
    for _ in range( days_in_future ):
                
        # Prédire le signal futur
        prediction = model.predict( last_window  )
        predictions.append( prediction[0] )
        
        # Préparer l'entrée pour la prochaine prédiction
        X_prime = np.roll( last_window, shift=-1, axis=0)  # Déplacer la fenêtre
        last_window = pandas.DataFrame( X_prime, columns=features )
    
    # Convertir les prédictions en tableau numpy
    predictions = np.array( predictions )
    
    # Mise à l'échelle pour l'affichage
    mean = df['Close'].mean()
    predictions = predictions + mean

    # Inverser la transformation pour obtenir les valeurs réelles du signal si besoin
    # Ici, nous n'utilisons pas de `scaler.inverse_transform` car nous avons prédit directement les signaux
    return predictions