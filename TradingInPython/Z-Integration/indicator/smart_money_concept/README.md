# Smart Money Concept (SMC) trading methode and ICT

Conception réalisation d'un moteur SMC

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

