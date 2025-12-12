# Smart Money Concept (SMC) trading methode ICT (Inner Circle Trading)

Conception réalisation d'un moteur SMC Interface Tkinter

- with scenario generator

## main python script

- smc.py :
    - generate signal
    - run Tkinter UI

## SMC engine

- smc_engine.py : tag les différents concept ICT dans le dataframe [ 'Open', 'High', 'Low', 'Close' ]

- CHoCH : Change of Character
- BOS : Break Of Structure
- Swings : préstructure HH HL LH LL
- OB : OderBlocks
- FVG : Faire Value Gap
- OTE : Optimal Trade Entry / Premium-Discount
- Liquidity (sweep)

Utilise plot_overlays pour venir tagger un 'ax' du graphe MatPlotLib

## Market Stock State Machine

- détection des BOS (Break Of Structure)

Retrouver cette méthode de trading dans l'application :

- [TradingInPython - Comment décoder la strcuture du marché](https://www.trading-et-data-analyses.com/2025/12/comment-decoder-la-structure-du-marche.html)
