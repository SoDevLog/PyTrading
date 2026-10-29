# 🐍 TradingInPython - Bibliothèque de scripts Python

Des exemples de scripts pour **analyser les marchés, tester des indicateurs et explorer les données financières** avec [TradingInPython](https://trading-in-python.sodevlog.com/).

Cette bibliothèque rassemble des scripts que vous pouvez exécuter depuis le **Script Runner** de la plateforme. Étudiez les pour les adapter à vos propres besoins.

> **Vous souhaitez écrire vos scripts personnalisés ?** Vous pouvez commencer par l'API utilisateur : [`use_api.py`](use_api.py) pour comprendre comment votre script peut exploiter les données transmises par TradingInPython.

---

## 🚀 Démarrer avec le Script Runner

Le Script Runner peut transmettre à un script python des données telles que :

- le symbole et le nom de l'action sélectionnée,
- les données de marché récupérées par le fournisseur de données,
- une liste de tickers.

Ces informations sont accessibles via l'API de la plateforme (`api.py`). Les paramètres disponibles et leur utilisation sont illustrés dans le script [`use_api.py`](use_api.py).

**Pour aller plus loin :** [Documentation - Interpréteur de scripts Python](https://trading-in-python.sodevlog.com/script-interpreter/interface/)

_Quelques uns de ces scripts fonctionnent en standalone si vous avez un environnement de développement Python installé sur votre machine et sinon pour exéctuer ces scripts :_

- [Téléchargez TradingInPython](https://www.trading-et-data-analyses.com/p/plateforme-de-trading-technique.html)

---

## 📈 Analyse technique et stratégies de trading

### Comment détecter une capitulation vendeuse ?

Algorithme de détection de la capitulation des vendeurs, comment détecter la fin de la prise de contrôle des vendeurs et un retour à la hausse ?

C'est un condensé d'analyse technique, à l'aide de différents indicateurs techniques l'algorithme va détecter les évènements suivants :

- Une tendance baissière déjà bien établie,
- Une nouvelle accélération de la baisse,
- Un volume de transactions nettement supérieur à la normale,
- Une amplitude de bougie inhabituelle, signe d'un marché particulièrement agité,
- Une clôture avec une longue mèche qui montre éventuellement un rejet des prix les plus bas,
- Un RSI qui atteint une zone de survente,
- Puis, dans les séances suivantes, une diminution de la pression vendeuse et l'apparition d'un rebond.

Ce Filtre repère une séance particulière au cours de laquelle la baisse du cours s'accompagne d'un volume exceptionnel et d'une amplitude de variation inhabituelle (ATR).

- ➡️ [`scanner-seller-capitulation.py`](scanner-seller-capitulation.py)
- 📚 [Documentation : Comment détecter une capitulation des vendeurs](https://www.trading-et-data-analyses.com/2026/10/comment-detecter-une-capitulation.html)

### Détection des Supports et des Résistances

Algorithme de détection des niveaux de Supports et de Résistances dynamiques guidée par le profil de volume. Les supports et les résistances occupent une place importante dans l'analyse technique des marchés financiers.

- ➡️ [`support-resistance-levels.py`](support-resistance-levels.py)  
- 📚 [Documentation : Comment détecter les supports et les résistances dynamiques avec Python ?](https://www.trading-et-data-analyses.com/2026/10/detecter-les-supports-et-les-resistances-en-python.html)

Cette détection des niveaux de supports et de résistance est maintenant intégrée dans la plateforme :

- [TradingInPython](https://www.trading-et-data-analyses.com/p/plateforme-de-trading-technique.html)

### SuperTrend Indicator

Un indicateur de tendance fondé sur l'**ATR (Average True Range)**. Il aide à visualiser la direction de la tendance et ses retournements potentiels.

Le script utilise `super_trend()` de `digitsignalprocessing.indicators` et affiche le cours de clôture avec le SuperTrend :

- **Tendance haussière ou baissière** : le tracé s'adapte à la direction détectée,
- **Retournement haussier** : marqueur `▲`,
- **Retournement baissier** : marqueur `▼`.

Paramètres de l'exemple : `ATR Period = 10` et `Multiplier = 2.0`.

Le script peut être lancé depuis le Script Runner avec les paramètres `ticker`, `period` et `interval`. Il propose également un exemple autonome utilisant `yfinance` sur le titre NVDA, avec six mois de données journalières.

- ➡️ [`super-trend-indicator.py`](super-trend-indicator.py)

Ce script peut servir de point de départ pour étudier un filtre de tendance, des retournements ou des signaux d'entrée et de sortie.

Cet indicateur est maintenant intégré dans la plateforme.

### Bull/Bear Strength Index (BSI)

Une stratégie complète de trading qui combine plusieurs éléments d'analyse technique :

- un **Cloud BSI**, construit à partir de deux EMA ;
- un **Stop BSI**, fondé sur un ATR Trailing Stop ;
- une synergie destinée à rechercher un alignement entre le court et le moyen terme ;
- deux histogrammes pour distinguer les dynamiques de momentum court terme (CT) et moyen terme (MT).

Comment lire la force du marché pour répondre à une question fondamentale, le marché est-il vraiment haussier ? Ou est-il seulement en train de faire semblant sur un horizon de temps trop court pour être fiable.

- ➡️ [`strategy-bull-bear-strengh-index.py`](strategy-bull-bear-strengh-index.py)  
- 📚 [Lire l'article : Bull/Bear Strength Index - Python Trading Strategy](https://www.trading-et-data-analyses.com/2026/09/bullbear-strength-index-python-trading.html)

### Filtre des actions à forte croissance - Strong Growth

Un exemple de filtre pour rechercher des actions dont la hausse s'accompagne d'une dynamique de volume intéressante.

- ➡️ [`strong-growth-table.py`](strong-growth-table.py)  
- 📚 [Lire l'article : Filtre Strong Growth - détecter une hausse soutenue par les volumes avec Python](https://www.trading-et-data-analyses.com/2026/09/filtre-strong-growth-python.html)

---

## 🌍 Macroéconomie et données FRED

Ces scripts montrent comment exploiter des séries de la **Federal Reserve Bank of St. Louis (FRED)** pour construire des indices économiques.

### Avant de commencer : configurer la clé API

Une clé API FRED est nécessaire pour faire fonctionner les scripts de cette section. [Créer ou récupérer une clé API FRED](https://fred.stlouisfed.org/docs/api/api_key.html).

Ajoutez ensuite votre clé dans le fichier de configuration :

`C:\Users\<Nom d'utilisateur>\AppData\Local\TradingInPython\config\.env`

avec la ligne suivante :

```dotenv
FRED_API_KEY=xxx
```

Remplacez `xxx` par votre propre clé.

### Scripts disponibles

| Script | Objectif |
| --- | --- |
| [`indice-fred-eurozone.py`](indice-fred-eurozone.py) | Construire un indice économique pour la zone euro. |
| [`indice-fred-france.py`](indice-fred-france.py) | Construire un indice économique pour la France. |
| [`macro_indice_engine.py`](macro_indice_engine.py) | Généraliser la construction d'indices économiques à partir d'une configuration. |

Dans `macro_indice_engine.py`, la configuration utilisée dans l'exemple est importée ici :

```python
from macro_indices.france import INDEX_CONFIG
```

Vous pouvez adapter cette configuration pour explorer un autre ensemble de données.

📚 [Lire l'article : Comment construire un indice économique](https://www.trading-et-data-analyses.com/2026/04/construire-un-indicateur-macro-econimque-fred.html)

---

## 🏢 Analyse financière et données d'entreprises

Ces exemples explorent différentes approches d'analyse financière et de recherche sur les marchés.

| Script | Description |
| --- | --- |
| [`evaluation-PER-000.py`](evaluation-PER-000.py) | Étudier le bénéfice par action (EPS) et calculer un PER (*Price/Earnings Ratio*) en le comparant au marché. |
| [`dataroma-buffett-000.py`](dataroma-buffett-000.py) | Explorer les positions attribuées à Berkshire Hathaway sur le site Dataroma. |
| [`juste-valeur-FCF.py`](juste-valeur-FCF.py) | Estimer une juste valeur à partir de projections de *Free Cash Flow* (FCF). |
| [`indice-market-000.py`](indice-market-000.py) | Comparer des indices boursiers, avec un exemple autour du S&P 500. |
| [`greenblatt-000.py`](greenblatt-000.py) | Explorer la *Magic Formula* de Joel Greenblatt. |
| [`market-calendar-000.py`](market-calendar-000.py) | Consulter un calendrier des jours d'ouverture des marchés boursiers. |

---

## 🤖 Intelligence artificielle et trading

### Agent IA de trading technique

Un exemple d'agent reposant sur un **MLP (Multi-Layer Perceptron)** avec Keras et le backend JAX, appliqué au trading technique.

- ➡️ [`agent_ia.py`](agent_ia.py)

---

## 🧩 Créez et adaptez vos propres scripts

Les exemples de cette bibliothèque peuvent vous aider à découvrir les possibilités offertes par Python dans TradingInPython, puis à développer vos propres analyses.

1. Consultez [`use_api.py`](use_api.py) pour comprendre comment accéder aux données transmises par la plateforme.
2. Choisissez un script proche de votre besoin et étudiez son fonctionnement.
3. Adaptez-le progressivement à votre approche et aux bibliothèques disponibles dans votre environnement.
4. Consultez la [documentation du Script Runner](https://trading-in-python.sodevlog.com/script-interpreter/) pour les modalités d'exécution.

⭐ **Vous trouvez ces scripts utiles ?** Pensez à ajouter une étoile au [projet PyTrading sur GitHub](https://github.com/SoDevLog/PyTrading) pour suivre ses évolutions.

---

## ⚠️ Important

Ces scripts sont fournis **à titre d'exemple, en l'état**. Les sources de données, les API ou les bibliothèques utilisées peuvent évoluer, certains scripts peuvent donc nécessiter des adaptations pour continuer à fonctionner.

_Ils sont proposés à des fins d'exploration et de développement. Ils ne constituent pas des recommandations d'achat ou de vente._
