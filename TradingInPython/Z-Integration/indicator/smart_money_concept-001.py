"""
    Module Smart Money Concepts (SMC)
    - Détection : Swings, Structure (HH/HL/LH/LL), CHoCH, BOS
    - Liquidity (EQH/EQL), FVG, Order Blocks (OB)
    - Plotting mplfinance + overlays Matplotlib
    - Simple intégration Tkinter : cases à cocher pour activer/désactiver les overlays

    - detect_liquidity
    
    - generate_sample_data
    - overlay
    - SmartMoneyConcepts
    - SMC_Tkinter_UI
    
    redraw : with new random data
    plot : with current data
    
    Auteur : TradingInPython
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tkinter as tk
import mplfinance as mpf
from tkinter import ttk

# -----------------------------------------------------------------------------

def generate_sample_data():
    
    NB_DATA = 120
    
    # Générer NB_DATA jours de données
    dates = pd.date_range(start="2024-01-01", periods=NB_DATA, freq="D")

    prices = []
    price = 100 # start price

    for i in range( NB_DATA ):
        # hausse
        if i < 19:
            drift = 0.25
        # baisse
        elif i < 38:
            drift = -0.2
        # range
        elif i < 50:
            drift = 0.01            
        # hausse
        if i < 60:
            drift = 0.25
        # range
        elif i < 90:
            drift = -0.01
        # baisse
        else:
            drift = -0.3

        price += drift + np.random.normal( 0, 0.6 )
        prices.append( price )

    # Créer OHLC réalistes
    opens = np.array(prices) + np.random.normal( 0, 0.3, NB_DATA )
    highs = opens + np.abs(np.random.normal( 0.8, 0.3, NB_DATA ))
    lows = opens - np.abs(np.random.normal( 0.8, 0.3, NB_DATA ))
    closes = lows + (highs - lows) * np.random.rand( NB_DATA )

    data = {
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes
    }
    df = pd.DataFrame( data, index=dates )
    return df

# -----------------------------------------------------------------------------
    
class SmartMoneyConcepts:
    def __init__( self, lookback=3, fvg_lookback=2, liquidity_threshold=1e-8 ):
        self.lookback = lookback
        self.fvg_lookback = fvg_lookback
        self.liquidity_threshold = 0.15 #liquidity_threshold

    # ---------------------------------------
    # 1) Swings (pivots) - simple local pivot
    # ---------------------------------------
    def detect_swings( self, df ):
        df = df.copy()
        L = self.lookback
        df['SwingHigh'] = False
        df['SwingLow'] = False

        for i in range(L, len(df)-L):
            high = df['High'].iloc[i]
            low = df['Low'].iloc[i]
            if high == df['High'].iloc[i-L:i+L+1].max():
                df.at[df.index[i], 'SwingHigh'] = True
            if low == df['Low'].iloc[i-L:i+L+1].min():
                df.at[df.index[i], 'SwingLow'] = True

        return df

    # -------------------------------
    # 2) Market Structure HH/HL/LH/LL
    # -------------------------------
    def market_structure( self, df ):
        df = df.copy()
        df['Structure'] = None
        last_high = None
        last_low = None

        for i in range(len(df)):
            if df.get('SwingHigh').iloc[i]:
                if last_high is None or df['High'].iloc[i] > last_high:
                    df.at[df.index[i], 'Structure'] = 'HH'
                else:
                    df.at[df.index[i], 'Structure'] = 'LH'
                last_high = df['High'].iloc[i]

            elif df.get('SwingLow').iloc[i]:
                if last_low is None or df['Low'].iloc[i] > last_low:
                    df.at[df.index[i], 'Structure'] = 'HL'
                else:
                    df.at[df.index[i], 'Structure'] = 'LL'
                last_low = df['Low'].iloc[i]

        return df

    # --------------
    # 3) CHoCH & BOS
    # --------------
    def detect_bos_choch( self, df ):
        df = df.copy()
        df['CHoCH'] = None
        df['BOS'] = None

        last = None
        for i in range(len(df)):
            curr = df['Structure'].iloc[i]
            if pd.isna(curr) or curr is None:
                continue

            if last is not None:
                # CHoCH heuristics
                if (last == 'HH' and curr in ('HL', 'LL')) or (last == 'HL' and curr in ('LH','LL')) or \
                   (last == 'LL' and curr in ('LH','HH')) or (last == 'LH' and curr in ('HL','HH')):
                    df.at[ df.index[i], 'CHoCH' ] = 'CHoCH'

                # BOS up
                if curr == 'HH' and last in ('LH', 'LL'):
                    df.at[df.index[i], 'BOS'] = 'BOS_UP'

                # BOS down
                if curr == 'LL' and last in ('HL', 'HH'):
                    df.at[df.index[i], 'BOS'] = 'BOS_DOWN'

            last = curr

        return df

    # ----------------------
    # 4) Liquidity detection
    # ----------------------
    # EQH (Equal High) = probabilité d’aller chercher une liquidité haute → ligne horizontale sur le niveau de High.
    # EQL (Equal Low) = probabilité d’aller chercher une liquidité basse → ligne horizontale sur le niveau de Low.
    #
    def detect_liquidity( self, df, threshold=None, bars_range=2 ):
        df = df.copy()
        th = self.liquidity_threshold if threshold is None else threshold

        df['EQH'] = False
        df['EQL'] = False
        df['BSL'] = False # Buy-Side Liquidity
        df['SSL'] = False # Sell-Side Liquidity
        df['LIQ_TAKEN'] = None  # 'BSL' ou 'SSL'

        # Détection ICT
        for i in range(bars_range, len(df)):
            hi = df['High'].iloc[i]
            lo = df['Low'].iloc[i]

            # Equal High : égalité avec une barre "bars_range" en arrière
            if abs(hi - df['High'].iloc[i - bars_range]) <= th:
                df.at[df.index[i], 'EQH'] = True
                df.at[df.index[i], 'BSL'] = True

            # Equal Low : égalité avec une barre "bars_range" en arrière
            if abs(lo - df['Low'].iloc[i - bars_range]) <= th:
                df.at[df.index[i], 'EQL'] = True
                df.at[df.index[i], 'SSL'] = True

        # Détection des prises de liquidité
        for i in range(1, len(df)):
            # Liquidité haute prise → mèche qui dépasse l'EQH précédent
            if df['High'].iloc[i] > df['High'].iloc[i-1] and df['EQH'].iloc[i-1]:
                df.at[df.index[i], 'LIQ_TAKEN'] = 'BSL'

            # Liquidité basse prise → mèche qui dépasse l'EQL précédent
            if df['Low'].iloc[i] < df['Low'].iloc[i-1] and df['EQL'].iloc[i-1]:
                df.at[df.index[i], 'LIQ_TAKEN'] = 'SSL'

        return df


    # ----------------
    # 5) FVG detection
    # ----------------
    def detect_fvg( self, df ):
        df = df.copy()
        df['FVG_UP'] = False
        df['FVG_DOWN'] = False
        for i in range(2, len(df)):
            if df['Low'].iloc[i] > df['High'].iloc[i-2]:
                df.at[df.index[i], 'FVG_UP'] = True
            if df['High'].iloc[i] < df['Low'].iloc[i-2]:
                df.at[df.index[i], 'FVG_DOWN'] = True
        return df

    # ------------------------------
    # 6) Order Blocks
    # ------------------------------
    def detect_order_blocks( self, df ):
        df = df.copy()
        df['Bullish_OB_low'] = pd.NA
        df['Bullish_OB_high'] = pd.NA
        df['Bearish_OB_low'] = pd.NA
        df['Bearish_OB_high'] = pd.NA

        for i in range( 2, len(df) ):
            if df['BOS'].iloc[i] == 'BOS_UP':
                prev = i-1
                if df['Close'].iloc[prev] < df['Open'].iloc[prev]:
                    df.at[df.index[prev], 'Bullish_OB_low'] = df['Low'].iloc[prev]
                    df.at[df.index[prev], 'Bullish_OB_high'] = df['High'].iloc[prev]
            if df['BOS'].iloc[i] == 'BOS_DOWN':
                prev = i-1
                if df['Close'].iloc[prev] > df['Open'].iloc[prev]:
                    df.at[df.index[prev], 'Bearish_OB_low'] = df['Low'].iloc[prev]
                    df.at[df.index[prev], 'Bearish_OB_high'] = df['High'].iloc[prev]

        return df

    # ------------------------------
    # 7) Overlay sur un axe existant
    # ------------------------------
    def overlay( self, df, ax, 
            show_structure=True, 
            show_bos=True, 
            show_choch=True, 
            show_liquidity=True, 
            show_ob=True, 
            show_fvg=True, 
            show_labels=True
        ):
        """Dessine les overlays SMC sur un axes Matplotlib existant (ax)."""
        
        # Créer un mapping index -> position pour mplfinance
        idx_to_pos = {idx: pos for pos, idx in enumerate(df.index)}

        # Structure labels
        if show_structure and 'Structure' in df.columns:
            for idx in df.index:
                s = df.loc[ idx, 'Structure' ]
                if s in ('HH','HL','LH','LL'):
                    pos = idx_to_pos[ idx ]
                    y = df.loc[ idx, 'High'] if s in ('HH','LH') else df.loc[ idx, 'Low']
                    if show_labels:
                        _facecolor = 'lightgreen' if s in ('HH','HL') else 'lightcoral'
                        ax.text( pos, y, s, fontsize=8,
                                 ha='center',
                                 va='bottom' if 'H' in s else 'top',
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor=_facecolor, alpha=0.2))

        # BOS lines
        if show_bos and 'BOS' in df.columns:
            for idx in df.index:
                b = df.loc[idx, 'BOS']
                pos = idx_to_pos[idx]
                if b == 'BOS_UP':
                    y = df.loc[idx, 'High']
                    #ax.axhline( y=y, linestyle='--', linewidth=1.5, color='green', alpha=0.7)
                    ax.hlines( y=y, xmin=pos, xmax=pos+20, linestyle='--', linewidth=0.9, color='green', alpha=0.7)
                    if show_labels:
                        ax.text( pos, y+0.5, 'BOS ↑', fontsize=8, color='green', 
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.2))
                elif b == 'BOS_DOWN':
                    y = df.loc[idx, 'Low']
                    #ax.axhline( y=y, linestyle='--', linewidth=1.5, color='red', alpha=0.7)
                    ax.hlines( y=y, xmin=pos, xmax=pos+20, linestyle='--', linewidth=0.9, color='red', alpha=0.7)
                    if show_labels:
                        ax.text( pos, y-0.5, 'BOS ↓', fontsize=8, color='red',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.2))

        # CHoCH markers
        if show_choch and 'CHoCH' in df.columns:
            for idx in df.index:
                choch = df.loc[ idx, 'CHoCH' ]
                if choch is not None:
                    pos = idx_to_pos[idx]

                    # Selon ICT, un CHoCH se place sur le pivot
                    struct = df.loc[idx, 'Structure']

                    # pivot HIGH
                    if struct in ('HH','LH'):
                        y = df.loc[idx, 'High'] + 0.5
                        va = 'bottom'
                        arrow = '↧' # flèche vers le bas (top → reversal)
                        
                        ax.text(
                            pos, y,
                            f"CHoCH {arrow}",
                            color='orange',
                            fontsize=9,
                            fontweight='bold',
                            ha='center',
                            va=va,
                            bbox=dict( boxstyle='square,pad=0.25', linewidth=0.5, facecolor='lightcoral', alpha=0.25 )
                        )
                    
                    # pivot LOW
                    if struct in ('HL', 'LL'):
                        y = df.loc[idx, 'Low'] - 0.5
                        va = 'top'
                        arrow = '↥'  # flèche vers le haut (bottom → reversal)

                        ax.text(
                            pos, y,
                            f"CHoCH {arrow}",
                            color='orange',
                            fontsize=9,
                            fontweight='bold',
                            ha='center',
                            va=va,
                            bbox=dict( boxstyle='square,pad=0.25', linewidth=0.5, facecolor='lightgreen', alpha=0.25 )
                        )

        # Liquidity TAKEN
        if show_liquidity and 'LIQ_TAKEN' in df.columns:
            for idx in df.index:
                liq = df.loc[idx, 'LIQ_TAKEN']
                if liq is None:
                    continue

                pos = idx_to_pos[idx]

                if liq == 'BSL': # Buy-Side Liquidity
                    y = df.loc[idx, 'High']
                    ax.scatter( pos, y+0.2, color='purple', s=40, marker='v' )
                    ax.text( pos, y+0.7, 'BSL taken', fontsize=8, color='purple' )

                elif liq == 'SSL': # Sell-Side Liquidity
                    y = df.loc[idx, 'Low']
                    ax.scatter( pos, y-0.2, color='brown', s=40, marker='^' )
                    ax.text( pos, y-0.7, 'SSL taken', fontsize=8, color='brown' )
                        
        # FVG rectangles
        if show_fvg and 'FVG_UP' in df.columns:
            for i, idx in enumerate(df.index[2:], start=2):
                if df.loc[idx, 'FVG_UP']:
                    top = df.loc[idx, 'Low']
                    bot = df.iloc[i-2]['High']
                    pos = idx_to_pos[idx]
                    ax.add_patch( patches.Rectangle((pos-2, bot), 2, top-bot,
                                                    facecolor='green', alpha=0.2, 
                                                    edgecolor='green', linewidth=1))
                if df.loc[idx, 'FVG_DOWN']:
                    top = df.iloc[i-2]['Low']
                    bot = df.loc[idx, 'High']
                    pos = idx_to_pos[idx]
                    ax.add_patch( patches.Rectangle((pos-2, bot), 2, top-bot,
                                                    facecolor='red', alpha=0.2,
                                                    edgecolor='red', linewidth=1))

        # Order Blocks
        if show_ob:
            for idx in df.index:
                pos = idx_to_pos[ idx ]
                if pd.notna( df.loc[idx, 'Bullish_OB_low']):
                    low = df.loc[idx, 'Bullish_OB_low']
                    high = df.loc[ idx, 'Bullish_OB_high']
                    ax.add_patch(patches.Rectangle((pos-0.4, low), 0.8, high-low,
                                                   facecolor='green', edgecolor='darkgreen',
                                                   alpha=0.3, linewidth=1.5))
                if pd.notna(df.loc[idx, 'Bearish_OB_high']):
                    low = df.loc[idx, 'Bearish_OB_low']
                    high = df.loc[idx, 'Bearish_OB_high']
                    ax.add_patch(patches.Rectangle((pos-0.4, low), 0.8, high-low,
                                                   facecolor='red', edgecolor='darkred',
                                                   alpha=0.3, linewidth=1.5))

    # -----------------
    # 8) Pipeline apply
    # -----------------
    def apply( self, df ):
        df1 = self.detect_swings( df )
        df2 = self.market_structure( df1 )
        df3 = self.detect_bos_choch( df2 )
        df4 = self.detect_liquidity( df3 )
        df5 = self.detect_fvg( df4 )
        df6 = self.detect_order_blocks( df5 )
        return df6

# -----------------------------------
# Tkinter UI for Smart Money Concepts
# -----------------------------------
class SmartMoneyApp:
    def __init__( self, df, smc ):
        self.df = df
        self.smc = smc
        self.root = tk.Tk()
        self.root.title('SMC Overlay Control')
        self.root.geometry('+20+50') # left top
        #self.root.geometry( "260x290" ) # width height
    
        # Variables Checkbuttons
        self.var_structure = tk.BooleanVar( value=True )
        self.var_bos = tk.BooleanVar( value=True )
        self.var_choch = tk.BooleanVar( value=True )
        self.var_liquidity = tk.BooleanVar( value=True )
        self.var_ob = tk.BooleanVar( value=True )
        self.var_fvg = tk.BooleanVar( value=True )

        # Label
        # -----
        ttk.Label(self.root, text="Smart Money Concepts", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, pady=10, sticky='w', padx=10)
        
        # Checkbuttons
        # ------------
        row = 1
        ttk.Checkbutton(self.root, text='Structure (HH/HL/LH/LL)', 
                       variable=self.var_structure).grid(row=row, column=0, sticky='w', padx=10)
        row = row + 1
        ttk.Checkbutton(self.root, text='Break of Structure (BOS)',
                       variable=self.var_bos).grid(row=row, column=0, sticky='w', padx=10)
        row = row + 1
        ttk.Checkbutton(self.root, text='Change of Character (CHoCH)', 
                       variable=self.var_choch).grid(row=row, column=0, sticky='w', padx=10)        
        row = row + 1
        ttk.Checkbutton(self.root, text='Liquidity (EQH/EQL)', 
                       variable=self.var_liquidity).grid(row=row, column=0, sticky='w', padx=10)        
        row = row + 1
        ttk.Checkbutton(self.root, text='Order Blocks (OB)', 
                       variable=self.var_ob).grid(row=row, column=0, sticky='w', padx=10)
        row = row + 1
        ttk.Checkbutton(self.root, text='Fair Value Gaps (FVG)', 
                       variable=self.var_fvg).grid(row=row, column=0, sticky='w', padx=10)
        
        # Buttons
        # -------
        row = row + 1
        ttk.Button(self.root, text='📊 Plot Chart', 
                  command=self.plot).grid(row=row, column=0, pady=(15,5), padx=10)
        
        row = row + 1
        ttk.Button(self.root, text='📊 Redraw', 
            command=self.redraw ).grid(row=row, column=0, pady=(5,15), padx=10)

        self.plot()
        
    def plot(self):
        """Plot avec mplfinance + overlays SMC"""
        fig, ax = plt.subplots( figsize=(14, 7) )
        
        # 1) Afficher les chandeliers avec mplfinance
        mpf.plot(
            self.df,
            type='candle',
            ax=ax,
            style='charles',
            show_nontrading=False
        )
        
        # 2) Superposer les overlays SMC
        self.smc.overlay(
            self.df, 
            ax,
            show_structure=self.var_structure.get(),
            show_bos=self.var_bos.get(),
            show_choch=self.var_choch.get(),
            show_liquidity=self.var_liquidity.get(),
            show_ob=self.var_ob.get(),
            show_fvg=self.var_fvg.get(),
            show_labels=True
        )
        
        ax.set_title("Smart Money Concepts (SMC) Analysis", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Price", fontsize=11)
        plt.tight_layout()
        plt.show()
    
    def redraw(self):
        _df = generate_sample_data()
        self.df = self.smc.apply( _df )
        self.plot()
        
    def run(self):
        self.root.mainloop()

# ------------------------------
# Example usage
# ------------------------------
if __name__ == '__main__':

    df = generate_sample_data()
    smc = SmartMoneyConcepts( lookback=2 )
    df = smc.apply( df )

    # Lancer l'UI   
    ui = SmartMoneyApp( df, smc )
    ui.run()