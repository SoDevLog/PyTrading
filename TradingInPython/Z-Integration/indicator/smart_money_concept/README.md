# Smart Money Concept (SMC) trading methode ICT (Inner Circle Trading)

Design and implementation of an SMC engine with a Tkinter interface

- with scenario generator
- sate machine for BoS detection

## Main python script

- smc.py # main entry point
    - generate signal
    - run Tkinter UI

## SMC engine

- smc_engine.py : tag the different ICT concepts in the dataframe [ 'Open', 'High', 'Low', 'Close' ]

- Swings : prestructure HH HL LH LL
- CHoCH : Change of Character
- BOS : Break Of Structure
- Swings : préstructure HH HL LH LL
- OB : OderBlocks
- FVG : Faire Value Gap
- OTE : Optimal Trade Entry / Premium-Discount
- Liquidity (sweep)

Use plot_overlays to tag an 'ax' of the MatPlotLib graph

## Market Stock State Machine

- detection of BOS (Break Of Structure)

## Build

- install **Python**
- install **PyInstaller**

- Create windows executable :

>pyinstaller smc.spec --clean

- [SMC.exe](./dist/)

## Formation

Retreive this trading method in the blog TradingInPython :

- [TradingInPython - Comment décoder la strcuture du marché](https://www.trading-et-data-analyses.com/2025/12/comment-decoder-la-structure-du-marche.html)
