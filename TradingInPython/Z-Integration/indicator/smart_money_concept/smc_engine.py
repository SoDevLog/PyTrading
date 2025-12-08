"""
    SMC Engine (ICT-style) - version complète pour TradingInPython
    - detect_swings
    - detect_structure
    - is_displacement
    - detect_bos_choch
    - detect_liquidity
    - detect_fvg : Fair Value Gap
    - detect_order_blocks : Order Blocks : Zones où de grands ordres ont été exécutés, pouvant agir comme des niveaux de support/résistance.
    - detect_ote : OTE Optimal Trade Entry / Premium-Discount
    
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -------------------------
# Utilities
# -------------------------
def ensure_datetime_index( df, date_col='Date' ):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        else:
            # try to coerce index
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError("Le DataFrame doit avoir un DatetimeIndex ou une colonne 'Date' convertible.") from e
    return df

def atr( df, period=14 ):
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift()).abs()
    tr3 = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

# -------------------------
# Core Engine
# -------------------------
@dataclass
class SMC_Params:
    swing_width: int = 3           # pivot detection window half-size (3 -> 3 left + 3 right)
    swing_min_width: int = 3
    swing_max_width: int = 5
    atr_period: int = 14
    fvg_atr_multiplier: float = 0.33
    displacement_body_ratio: float = 0.35 # 0.6  # body/total-range threshold for displacement
    ob_mitigation_bars: int = 6

class SMC_Engine:
    def __init__(self, params: SMC_Params = None):
        self.params = params or SMC_Params()

    # ----------------
    # 1) Detect swings
    # ----------------
    def detect_swings( self, df ):
        df = df.copy()
        L = self.params.swing_width
        df['SwingHigh'] = False
        df['SwingLow'] = False

        n = len(df)
        highs = df['High'].values
        lows = df['Low'].values

        for i in range(L, n-L):
            window_high = highs[ i-L:i+L+1 ]
            window_low = lows[ i-L:i+L+1 ]
            if highs[i] == window_high.max():
                # stricter: must be strictly greater than neighbors to avoid equal ties
                if ( highs[i] > window_high.max() - 1e-12) and (np.sum(window_high == highs[i]) == 1):
                    df.iat[i, df.columns.get_loc('SwingHigh')] = True
            if lows[i] == window_low.min():
                if (lows[i] < window_low.min() + 1e-12) and (np.sum(window_low == lows[i]) == 1):
                    df.iat[i, df.columns.get_loc('SwingLow')] = True
        return df

    # -------------------
    # 2) Market Structure
    # -------------------
    def detect_structure(self, df):
        df = df.copy()
        df['Structure'] = None
        last_high_val = None
        last_low_val = None
        last_high_idx = None
        last_low_idx = None

        for i in range(len(df)):
            if df['SwingHigh'].iat[i]:
                h = df['High'].iat[i]
                if last_high_val is None or h > last_high_val:
                    df.at[df.index[i], 'Structure'] = 'HH'
                else:
                    df.at[df.index[i], 'Structure'] = 'LH'
                last_high_val = h
                last_high_idx = i
            elif df['SwingLow'].iat[i]:
                l = df['Low'].iat[i]
                if last_low_val is None or l > last_low_val:
                    df.at[df.index[i], 'Structure'] = 'HL'
                else:
                    df.at[df.index[i], 'Structure'] = 'LL'
                last_low_val = l
                last_low_idx = i
        return df

    # ---------------------------
    # helpers: displacement check
    # ---------------------------
    def is_displacement(self, df, i):
        # large body relative to range and body > displacement_body_ratio * (high-low)
        o = df['Open'].iat[i]
        c = df['Close'].iat[i]
        h = df['High'].iat[i]
        l = df['Low'].iat[i]
        body = abs(c - o)
        rng = h - l if (h - l) > 0 else 1e-9
        return (body / rng) >= self.params.displacement_body_ratio

    # ------------------------------------
    # 3) CHoCH ICT strict + BOS ICT strict
    # ------------------------------------
    def detect_bos_choch(self, df):
        df = df.copy()
        df['CHoCH'] = None
        df['BOS'] = None
        df['DISPLACEMENT'] = False  # mark displacement candles

        last_swing_high_val = None
        last_swing_low_val = None
        last_swing_high_idx = None
        last_swing_low_idx = None

        # track purge flags
        purge_low = False
        purge_high = False

        _atr = atr(df, period=self.params.atr_period).bfill() # fillna(method='bfill')

        for i in range(len(df)):
            _high = df['High'].iat[i]
            _low = df['Low'].iat[i]
            _open = df['Open'].iat[i]
            _close = df['Close'].iat[i]

            # update last swing values
            if df['SwingHigh'].iat[i]:
                last_swing_high_val = df['High'].iat[i]
                last_swing_high_idx = i
            if df['SwingLow'].iat[i]:
                last_swing_low_val = df['Low'].iat[i]
                last_swing_low_idx = i

            # displacement detection (mark the aggressive candles)
            if self.is_displacement(df, i):
                df.at[df.index[i], 'DISPLACEMENT'] = True

            # ---------- CHoCH UP logic (ICT strict)
            # Step 1: purge below last swing low (liquidity grab)
            if last_swing_low_val is not None and _low < last_swing_low_val:
                purge_low = True

            # Step 2: shift - closing break of a previous swing high AND displacement confirmation
            if purge_low and last_swing_high_val is not None:
                # require closing break (ICT) AND displacement candle (either this close or a previous displacement within 2 bars)
                close_breaks = _close > last_swing_high_val
                recent_disp = df['DISPLACEMENT'].iat[i] or (i-1 >= 0 and df['DISPLACEMENT'].iat[i-1])
                if close_breaks and recent_disp:
                    # validate structure: pivot low must be HL/LL
                    struct_0 = df['Structure'].iat[i]
                    struct_1 = df['Structure'].iat[i-1]
                    if struct_0 in ('HL', 'LL'):
                        df.at[df.index[i], 'CHoCH'] = 'CHoCH_UP'
                        purge_low = False
                        # CHoCH supersedes BOS on this bar
                        continue
                    if struct_1 in ('HL', 'LL'):
                        df.at[df.index[i - 1], 'CHoCH'] = 'CHoCH_UP'
                        purge_low = False
                        # CHoCH supersedes BOS on this bar
                        continue
                    else:
                        # reset if structure incompatible
                        purge_low = False

            # ---------- CHoCH DOWN logic (ICT strict)
            if last_swing_high_val is not None and _high > last_swing_high_val:
                purge_high = True

            if purge_high and last_swing_low_val is not None:
                close_breaks = _close < last_swing_low_val
                recent_disp = df['DISPLACEMENT'].iat[i] or (i-1 >= 0 and df['DISPLACEMENT'].iat[i-1])
                if close_breaks and recent_disp:
                    struct_0 = df['Structure'].iat[i]
                    struct_1 = df['Structure'].iat[i-1]
                    if struct_0 in ('HH', 'LH'):
                        df.at[df.index[i], 'CHoCH'] = 'CHoCH_DOWN'
                        purge_high = False
                        continue
                    if struct_1 in ('HH', 'LH'):
                        df.at[df.index[i-1], 'CHoCH'] = 'CHoCH_DOWN'
                        purge_high = False
                        continue
                    else:
                        purge_high = False

            # ---------- BOS (ICT strict) - closing break only, requires displacement OR non-spike confirmation
            # BOS up
            if last_swing_high_val is not None:
                if _close > last_swing_high_val:
                    # require either displacement or close > last_swing_high by ATR*factor
                    if df['DISPLACEMENT'].iat[i] or (_close - last_swing_high_val) > 0.25 * _atr.iat[i]:
                        df.at[df.index[i], 'BOS'] = 'BOS_UP'
                        # update last swing high as the break price to avoid repeated BOS on same level
                        last_swing_high_val = _close

            # BOS down
            if last_swing_low_val is not None:
                if _close < last_swing_low_val:
                    if df['DISPLACEMENT'].iat[i] or (last_swing_low_val - _close) > 0.25 * _atr.iat[i]:
                        df.at[df.index[i], 'BOS'] = 'BOS_DOWN'
                        last_swing_low_val = _close

        return df

    # --------------------------------------------
    # 4) Liquidity detection (EQH/EQL + LIQ_TAKEN)
    # --------------------------------------------
    def detect_liquidity( self, df, bars_range=2 ):
        df = df.copy()
        th = atr(df, period=self.params.atr_period).mean() * 0.02
        df['EQH'] = False
        df['EQL'] = False
        df['BSL'] = False  # Buy-Side Liquidity
        df['SSL'] = False  # Sell-Side Liquidity
        df['LIQ_TAKEN'] = None

        for i in range(bars_range, len(df)):
            hi = df['High'].iat[i]
            lo = df['Low'].iat[i]
            if abs(hi - df['High'].iat[i - bars_range]) <= th:
                df.at[df.index[i], 'EQH'] = True
                df.at[df.index[i], 'BSL'] = True
            if abs(lo - df['Low'].iat[i - bars_range]) <= th:
                df.at[df.index[i], 'EQL'] = True
                df.at[df.index[i], 'SSL'] = True

        for i in range(1, len(df)):
            if df['EQH'].iat[i-1] and df['High'].iat[i] > df['High'].iat[i-1]:
                df.at[df.index[i], 'LIQ_TAKEN'] = 'BSL'
            if df['EQL'].iat[i-1] and df['Low'].iat[i] < df['Low'].iat[i-1]:
                df.at[df.index[i], 'LIQ_TAKEN'] = 'SSL'
        return df

    # --------------------------
    # 5) FVG v2 (impulse driven)
    # --------------------------
    def detect_fvg(self, df):
        df = df.copy()
        atv = atr(df, period=self.params.atr_period)
        df['FVG_UP'] = False
        df['FVG_DOWN'] = False
        for i in range(2, len(df)):
            # original rule: Low[i] > High[i-2] (up imbalance)
            if df['Low'].iat[i] > df['High'].iat[i-2]:
                # require displacement: the impulse that created it must be large enough vs ATR
                impulse_body = abs(df['Close'].iat[i-1] - df['Open'].iat[i-1])
                if impulse_body > 0.25 * atv.iat[i-1]:
                    df.at[df.index[i], 'FVG_UP'] = True
            if df['High'].iat[i] < df['Low'].iat[i-2]:
                impulse_body = abs(df['Close'].iat[i-1] - df['Open'].iat[i-1])
                if impulse_body > 0.25 * atv.iat[i-1]:
                    df.at[df.index[i], 'FVG_DOWN'] = True
        return df

    # -------------------
    # 6) Order Blocks PRO
    # -------------------
    def detect_order_blocks_old( self, df ):
        df = df.copy()
        df['Bullish_OB_low'] = pd.NA
        df['Bullish_OB_high'] = pd.NA
        df['Bearish_OB_low'] = pd.NA
        df['Bearish_OB_high'] = pd.NA
        
        # OB detection: last opposing candle before DISPLACEMENT that has opposite body
        for i in range(2, len(df)):
            if df['DISPLACEMENT'].iat[i]:
                # if displacement up, find previous bearish candle as OB candidate
                if df['Close'].iat[i] > df['Open'].iat[i]:
                    # search back up to ob_mitigation_bars for a bearish candle
                    j = max(0, i - self.params.ob_mitigation_bars)
                    found = None
                    for k in range(i-1, j-1, -1):
                        if df['Close'].iat[k] < df['Open'].iat[k]:
                            found = k
                            break
                    if found is not None:
                        df.at[df.index[found], 'Bullish_OB_low'] = df['Low'].iat[found]
                        df.at[df.index[found], 'Bullish_OB_high'] = df['High'].iat[found]
                # displacement down -> bullish candle as OB
                if df['Close'].iat[i] < df['Open'].iat[i]:
                    j = max(0, i - self.params.ob_mitigation_bars)
                    found = None
                    for k in range(i-1, j-1, -1):
                        if df['Close'].iat[k] > df['Open'].iat[k]:
                            found = k
                            break
                    if found is not None:
                        df.at[df.index[found], 'Bearish_OB_low'] = df['Low'].iat[found]
                        df.at[df.index[found], 'Bearish_OB_high'] = df['High'].iat[found]
        return df

    def detect_order_blocks( self, df ):
        df = df.copy()
        df['Bullish_OB_low'] = pd.NA
        df['Bullish_OB_high'] = pd.NA
        df['Bearish_OB_low'] = pd.NA
        df['Bearish_OB_high'] = pd.NA
        
        for i in range(2, len(df)):
            if not df['DISPLACEMENT'].iat[i]:
                continue
            
            displacement_up = df['Close'].iat[i] > df['Open'].iat[i]
            
            # Chercher la dernière bougie opposée (plus proche du displacement)
            for k in range( i-1, max(0, i-self.params.ob_mitigation_bars ), -1 ):
                candle_up = df['Close'].iat[k] > df['Open'].iat[k]
                
                # Bullish OB: dernière bougie DOWN avant displacement UP
                if displacement_up and not candle_up:
                    df.at[df.index[k], 'Bullish_OB_low'] = df['Low'].iat[k]
                    df.at[df.index[k], 'Bullish_OB_high'] = df['High'].iat[k]
                    break
                
                # Bearish OB: dernière bougie UP avant displacement DOWN
                elif not displacement_up and candle_up:
                    df.at[df.index[k], 'Bearish_OB_low'] = df['Low'].iat[k]
                    df.at[df.index[k], 'Bearish_OB_high'] = df['High'].iat[k]
                    break
        
        return df

    # ---------------------------------------------
    # 7) OTE Optimal Trade Entry / Premium-Discount
    # ---------------------------------------------
    def detect_ote( self, df ):
        df = df.copy()
        df['OTE_low'] = pd.NA
        df['OTE_high'] = pd.NA
        # For each OB, create 50% and 62% zones as OTE (if present)
        for idx in df.index:
            if pd.notna(df.loc[idx, 'Bullish_OB_low']):
                low = df.loc[idx, 'Bullish_OB_low']
                high = df.loc[idx, 'Bullish_OB_high']
                mid = 0.5*(low+high)
                df.at[idx, 'OTE_low'] = mid - 0.5*(high-low)*0.12
                df.at[idx, 'OTE_high'] = mid + 0.5*(high-low)*0.12
            if pd.notna(df.loc[idx, 'Bearish_OB_high']):
                low = df.loc[idx, 'Bearish_OB_low']
                high = df.loc[idx, 'Bearish_OB_high']
                mid = 0.5*(low+high)
                df.at[idx, 'OTE_low'] = mid - 0.5*(high-low)*0.12
                df.at[idx, 'OTE_high'] = mid + 0.5*(high-low)*0.12
        return df

    # -------------------------
    # 8) Pipeline apply
    # -------------------------
    def apply(self, df):
        df = ensure_datetime_index(df)
        df1 = self.detect_swings(df)
        df2 = self.detect_structure(df1)
        df3 = self.detect_bos_choch(df2)
        df4 = self.detect_liquidity(df3)
        df5 = self.detect_fvg(df4)
        df6 = self.detect_order_blocks(df5)
        df7 = self.detect_ote(df6)
        return df7

    # -------------------------
    # 9) Plotting overlays (safe)
    # -------------------------
    def plot_overlays(
            self, ax, df, 
            show_structure=True, 
            show_bos=True, 
            show_choch=True,
            show_liquidity=True, 
            show_ob=True, 
            show_fvg=True, 
            show_ote=True, 
            show_labels=True
        ):
        """
        Draw overlays directly on a provided matplotlib Axes (safe for mplfinance usage).
        This function always recreates artists.
        """
        idx_to_pos = {idx: pos for pos, idx in enumerate(df.index)}

        # Structure
        if show_structure and 'Structure' in df.columns:
            for idx in df.index:
                s = df.loc[idx, 'Structure']
                if s in ('HH','HL','LH','LL'):
                    pos = idx_to_pos[idx]
                    y = df.loc[idx, 'High'] if s in ('HH','LH') else df.loc[idx, 'Low']
                    if show_labels:
                        face = 'lightgreen' if s in ('HH','HL') else 'lightcoral'
                        ax.text(pos, y, s, fontsize=8, ha='center', va='bottom' if 'H' in s else 'top',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor=face, alpha=0.2))

        # BOS
        if show_bos and 'BOS' in df.columns:
            for idx in df.index:
                bos = df.loc[idx, 'BOS']
                if bos is None: continue
                pos = idx_to_pos[idx]
                if bos == 'BOS_UP':
                    y = df.loc[idx, 'High']
                    ax.hlines(y=y, xmin=pos, xmax=pos+20, linestyle='--', linewidth=0.5, color='green', alpha=0.7)
                    if show_labels: ax.text(pos, y+0.5, 'BOS ↑', fontsize=8, color='green',
                                            bbox=dict(boxstyle='square,pad=0.3', facecolor='lightgreen', alpha=0.2))
                elif bos == 'BOS_DOWN':
                    y = df.loc[idx, 'Low']
                    ax.hlines(y=y, xmin=pos, xmax=pos+20, linestyle='--', linewidth=0.5, color='red', alpha=0.7)
                    if show_labels: ax.text(pos, y-0.5, 'BOS ↓', fontsize=8, color='red',
                                            bbox=dict(boxstyle='square,pad=0.3', facecolor='lightcoral', alpha=0.2))

        # CHoCH
        if show_choch and 'CHoCH' in df.columns:
            for idx in df.index:
                choch = df.loc[idx, 'CHoCH']
                if choch is None: continue
                
                pos = idx_to_pos[idx]
                if choch == 'CHoCH_UP':
                    y = df.loc[idx, 'High']
                    ax.scatter(pos, y, color='orange', s=40, marker='v')
                    if show_labels:
                        ax.text( pos, y+0.5, 'CHoCH ↧', fontsize=9, ha='center', va='bottom', color='green',
                                bbox=dict(boxstyle='square,pad=0.25', facecolor='gold', alpha=0.25))
                elif choch == 'CHoCH_DOWN':
                    y = df.loc[idx, 'Low']
                    ax.scatter(pos, y, color='orange', s=40, marker='^')
                    if show_labels:
                        ax.text( pos, y-0.5, 'CHoCH ↥', fontsize=9, ha='center', va='top', color='red',
                                bbox=dict(boxstyle='square,pad=0.25', facecolor='gold', alpha=0.25))

        # Liquidity taken
        if show_liquidity and 'LIQ_TAKEN' in df.columns:
            for idx in df.index:
                lt = df.loc[idx, 'LIQ_TAKEN']
                if lt is None: continue
                
                pos = idx_to_pos[idx]
                if lt == 'BSL':
                    y = df.loc[idx, 'High']
                    ax.scatter(pos, y+0.2, color='purple', s=35, marker='v')
                    ax.text( pos, y+0.7, 'BSL', ha='center', fontsize=8, color='purple' )
                elif lt == 'SSL':
                    y = df.loc[idx, 'Low']
                    ax.scatter(pos, y-0.2, color='brown', s=35, marker='^')
                    ax.text( pos, y-0.7, 'SSL', ha='center', fontsize=8, color='brown' )

        # FVG
        if show_fvg and 'FVG_UP' in df.columns:
            for i, idx in enumerate(df.index[2:], start=2):
                if df.loc[idx, 'FVG_UP']:
                    top = df.loc[idx, 'Low']
                    bot = df.iloc[i-2]['High']
                    pos = idx_to_pos[idx]
                    ax.add_patch( patches.Rectangle((pos-2, bot), 2.5, top-bot, facecolor='green', alpha=0.2, edgecolor='green') )
                if df.loc[idx, 'FVG_DOWN']:
                    top = df.iloc[i-2]['Low']
                    bot = df.loc[idx, 'High']
                    pos = idx_to_pos[idx]
                    ax.add_patch( patches.Rectangle((pos-2, bot), 2.5, top-bot, facecolor='red', alpha=0.2, edgecolor='red') )

        # Order Blocks - detect_bos_choch
        if show_ob:
            for idx in df.index:
                if pd.notna( df.loc[idx, 'Bullish_OB_low'] ):
                    pos = idx_to_pos[idx]
                    low = df.loc[idx, 'Bullish_OB_low']
                    high = df.loc[idx, 'Bullish_OB_high']
                    ax.add_patch( patches.Rectangle((pos+0.4, low), -self.params.ob_mitigation_bars, high-low, facecolor='green', alpha=0.25, edgecolor='darkgreen'))
                if pd.notna(df.loc[idx, 'Bearish_OB_high']):
                    pos = idx_to_pos[idx]
                    low = df.loc[idx, 'Bearish_OB_low']
                    high = df.loc[idx, 'Bearish_OB_high']
                    ax.add_patch( patches.Rectangle((pos+0.4, low), -self.params.ob_mitigation_bars, high-low, facecolor='red', alpha=0.25, edgecolor='darkred'))

        # OTE zones
        if show_ote and ('OTE_low' in df.columns and 'OTE_high' in df.columns):
            for idx in df.index:
                if pd.notna(df.loc[idx, 'OTE_low']) and pd.notna(df.loc[idx, 'OTE_high']):
                    pos = idx_to_pos[idx]
                    low = df.loc[idx, 'OTE_low']
                    high = df.loc[idx, 'OTE_high']
                    ax.add_patch( patches.Rectangle((pos-0.5, low), 2, high-low, facecolor='yellow', alpha=0.12, edgecolor='gold'))
