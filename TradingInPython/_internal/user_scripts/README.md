# User's script runnable by plateforme's Script-runner

Ces scripts sont exéctuables par le **Script-runner** de la plateforme **TradingInPython**.

Des données comme :

- le symbol du stock
- son nom
- les data fetchées par le data provider
- une liste de ticker de stocks
- ...

Sont transmisent au script par l'API : api.py (Application Porgramming Interface).

Pour accéder à la documentation du Script Runner de **TradingInPython** :

- [Interpréteur de script python](https://trading-in-python.sodevlog.com/script-interpreter/)

Pour savoir comment utiliser ces données dans votre script régardez dans le fichier :

- [use_api.py](use_api.py)

## SuperTrend Indicator

Un indicateur technique de tendance basé sur l'**ATR (Average True Range)** permettant d'identifier la direction de la tendance et les retournements haussiers ou baissiers.

Le script utilise l'implémentation `super_trend()` de `digitsignalprocessing.indicators` et affiche simultanément le cours de clôture et le SuperTrend.

### Fonctionnement

Le SuperTrend est représenté différemment selon la direction de la tendance :

- **SuperTrend haussier** : affiché lorsque la tendance est haussière.
- **SuperTrend baissier** : affiché lorsque la tendance est baissière.
- **Retournement haussier** : signalé par un marqueur `▲`.
- **Retournement baissier** : signalé par un marqueur `▼`.

Les paramètres utilisés dans cet exemple sont :

- **ATR Period** : `10`
- **Multiplier** : `2.0`

### Utilisation

Le script peut être exécuté directement depuis le **Script-runner de TradingInPython** et utilise les paramètres `ticker`, `period` et `interval` transmis par l'API de la plateforme.

Il peut également être exécuté de manière autonome avec `yfinance`. L'exemple fourni utilise **NVDA** sur une période de six mois avec des données journalières.

- [super-trend-indicator.py](super-trend-indicator.py)

### Exemple

Le script peut également servir de base pour développer ou adapter une stratégie de trading utilisant le SuperTrend comme :

- filtre de tendance ;
- détection de retournement ;
- signal d'entrée ou de sortie ;
- composant d'une stratégie combinant plusieurs indicateurs techniques.

> Ce script est fourni comme exemple et peut être adapté à vos propres besoins.

## Bull/Bear Strength Index

Un stratégie de trading complète qui utilise les indicateurs techniques :

- Cloud BSI basé sur deux EMA,
- Stop BSI basé sur un ATR Trailing Stop,
- Synergie permettant de rechercher un alignement entre court et moyen terme
- deux histogrammes BSI permettant de distinguer le momentum court terme CT et moyen terme MT.
- [strategy-bull-bear-strengh-index.py](strategy-bull-bear-strengh-index.py)

### Documentation sur le BSI

- [Bull/Bear Strength Index - Python Trading Strategy](https://www.trading-et-data-analyses.com/2026/09/bullbear-strength-index-python-trading.html)


## Filtre des actions à forte croissance

- [strong-growth.py](strong-growth.py)

## Script sur la finance

Trailing EPS (Earnings Per Share) Calcul du PER (Price Earnings Ratio) en comparant au marché :

- [evaluation-PER-000.py](evaluation-PER-000.py)

Retrouver les actions tradées par Warren Buffett - Berkshire Hathaway sur le Site DATAROMA :

- [dataroma-buffett-000.py](dataroma-buffett-000.py)

Juste Valeur à partir de la projection du Free Cash Flow FCF :

- [juste-valeur-FCF.py](juste-valeur-FCF.py)

Comparaison des Indices entre eux - S&P Standard & Poor's SP 500 :

- [indice-market-000.py](indice-market-000.py)

La fameuse formule magique de Greenblatt Magic Formula :

- [greenblatt-000.py](greenblatt-000.py)

Calendrier des jours d'ouverture des bourses mondiales :

- [market-calendar-000.py](market-calendar-000.py)

Agent IA de trading technique - MLP (Multi-Layer Perceptrons) Keras (backend JAX) :

- [agent_ia.py](agent_ia.py)

Vous trouverez d'autres scripts que vous pourrez exécuter, adapter pour vos besoins, vous en inspirez.

**Pensez à "starer" le projet PyTrading**, mettez un étoile pour suivre les évolutions des Scripts et de leur utilisation dans **TradingInPython**.

## Mise en garde

_Ces scripts ne sont que des exemples, ils sont délivrés tels quels. Ils peuvent ne plus fonctionner à cause de certains changements et doivent parfois être adaptés pour fonctionner à nouveau._
