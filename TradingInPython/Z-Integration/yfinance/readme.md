# Comment trouver des actions sous-évaluées

Pour le trader une part importante de son travail est de trouver des actions à trader.

## Check YahooFinance

C'est très rare mais parfois cela arrive, le serveur de datas de YahooFinance est down et là, si votre script ne le prévois pas, vous ne voyez rien.

Alors pour détailler les erreurs de **yfinance** voici le script **check-yfinance.py**

## FCF - Free Cash Flow

Juste Valeur d'une action à partir de la projection du Free Cash Flow FCF

Capitalisation Globale de l'Entreprise

Avantages il n'y a pas de paramètres (ou presque pas)

Utiliser plus de données financières que par le 'DCF simple' (Discount Cash FLow)

Il faut regarder les valeurs de : cash_flow.loc['Free Cash Flow'] pour certaines sotck elles n'existent pas

**beta** : Indice de volatilité de cette action par rapport au marché

- beta = 1 : Volatilité égale à celle du marché. Si le marché augmente de 1%, on peut s'attendre à ce que l'action augmente également de 1%.
- beta > 1 : L'action est plus volatile que le marché. Par exemple, si le beta est de 1,5, cela signifie que l'action est 50 % plus volatile que le marché. Si le marché monte de 1%, l'action pourrait augmenter de 1,5%, et inversement en cas de baisse.
- beta < 1 : L'action est moins volatile que le marché. Un beta de 0,5 signifie que l'action est 50% moins volatile que le marché.
- beta négatif : Cela suggère que l'action pourrait se déplacer en sens inverse du marché, bien que cela soit assez rare.

### Paramètres

risk_free_rate = 0.02 # Taux sans risque (par exemple, 2%)
market_return = 0.08  # Rendement attendu du marché
cost_of_debt = 0.03  # Supposons un coût de la dette de 3%
tax_rate = 0.30  # Taux d'imposition supposé de 30%
growth_rate = 0.03  # Taux de croissance supposé des FCF de 3%

## DATAROMA Warren Buffett

**Return On Equity (ROE)** = Résultat Net / Capitaux propres
        - ROE élevé signifie que l'entreprise génère un bon rendement sur l'argent investi par ses actionnaires.
        - ROE faible peut indiquer une inefficacité dans l'utilisation des capitaux propres.
        - ROE négatif indique des pertes.

**Price Earnings Ratio** = Prix de l'action / Bénéfices par actions (EPS)
        - PER élevé peut indiquer que les investisseurs s'attendent à une forte croissance future des bénéfices et sont donc prêts à payer un prix élevé pour l'action.
        - PER faible peut suggérer que l'entreprise est sous-évaluée, ou bien qu'elle rencontre des difficultés financières.
        - PER très bas pourrait aussi indiquer que l'entreprise est en difficulté ou dans un secteur en déclin.

**Price To Book Ratio** = Cours de l'action / Valeur Comptable par Action (Book Value per Share BVPS)
        - P/B > 1 : L'entreprise est valorisée au-dessus de sa valeur comptable. Cela peut indiquer une forte rentabilité ou une anticipation de croissance future.
        - P/B < 1 : L'entreprise est sous-évaluée par rapport à ses actifs, ce qui peut être une opportunité d'investissement ou le signe d'un problème structurel.
        - P/B ≈ 1 : L'entreprise est valorisée proche de sa valeur comptable.

## EPS et PER

Le **PER** (Price Earning Ratio) est le cours de l'action divisé pas le bénéfice par action (EPS). Il permet de comparer la valorisation d'une entreprise par rapport à ses bénéfices. Plus le PER est élevé, plus l'entreprise est chère par rapport à ses bénéfices.

Le **EPS** (Earning Per Share) est le bénéfice net de l'entreprise divisé par le nombre d'actions en circulation. Il permet de mesurer la rentabilité de l'entreprise par action. Plus l'EPS est élevé, plus l'entreprise est rentable.

## Formule Magique de Greenblatt

C'est une stratégie d'investissement simple créée par Joel Greenblatt (présentée dans son livre The Little Book That Beats the Market, 2005). L'idée est de trouver des entreprises de qualité achetées à bon prix, en combinant deux indicateurs.

Les deux piliers de la formule

1. Earnings Yield (EY) — "Est-ce que c'est bon marché ?"

EY = EBIT / EV

EBIT = bénéfice avant intérêts et impôts (la vraie capacité bénéficiaire)
Enterprise Value (EV) = capitalisation boursière + dettes − trésorerie (le "vrai prix" de l'entreprise)

- Plus l'EY est élevé, plus l'action est bon marché par rapport à ses bénéfices

1. Return on Capital (ROC) — "Est-ce que c'est une bonne entreprise ?"

ROC = EBIT / Capital Investi

Capital Investi = Actifs totaux − Passifs totaux

- Plus le ROC est élevé, plus l'entreprise génère de valeur avec ses ressources

Le score le plus bas = meilleur investissement potentiel.
