"""
    SMC Engine (ICT) - TradingInPython
    - detect_swings
    - detect_structure
    - is_displacement
    - detect_bos_choch : BoS (Break of Stucture) - CHoCH (CHange of CHaracters) changement de Structure ou MSS pour Market Structure Shift
    - detect_liquidity
    - detect_fvg : Fair Value Gap
    - detect_order_blocks : Order Blocks : Zones où de grands ordres ont été exécutés, pouvant agir comme des niveaux de support/résistance.
    - detect_ote : OTE Optimal Trade Entry / Premium-Discount
    
    - overlays_structure
    - overlays_segments
    - overlays_displacement
    - overlays_market_state
    - overlays_bos
    - overlays_choch
    - overlays_liquidity
    - overlays_fvg
    - overlays_order_blocs
    - overlays_ote

    - bos_atr_confirm :
        0.15 - 0.20 : marchés très volatils (crypto, small caps) ou TF courts (1-15m) — détecte plus de BOS, plus sensible.
        0.20 - 0.30 : réglage « standard » pour la plupart des actions/FX / TF intraday → bon compromis.
        0.30 - 0.40 : marchés moins volatils (large caps, indices) ou TF daily/weekly — exige un break plus net.
        0.40 - 0.50 : usages très conservateurs — limite fortement les faux-positifs, peut rater certains BOS.
    
"""
import numpy
import pandas as pd
import matplotlib.patches as patches

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from numpy.lib.stride_tricks import sliding_window_view
from dataclasses import dataclass
from state_machine_bos import MarketStockStateMachine, MarketState

# -------------------------
# Utilities
# -------------------------
def ensure_datetime_index( df, date_col='Date' ):
    df = df.copy()
    if not isinstance( df.index, pd.DatetimeIndex ):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime( df[date_col] )
            df = df.set_index( date_col )
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
    swing_width: int = 3 # pivot detection ( 3 left + pivot + 3 right)
    liquidity_threshold: float = 0.001
    bos_atr_confirm: float = 0.25
    swing_min_width: int = 3
    swing_max_width: int = 5
    atr_period: int = 14
    fvg_atr_multiplier: float = 0.33
    displacement_body_ratio: float = 0.35 # 0.6  # body/total-range threshold for displacement
    ob_mitigation_bars: int = 6

class SMC_Engine:
    def __init__(self, params: SMC_Params = None):
        self.params = params or SMC_Params()
        self.idx_to_pos = None

        # Colors for overlays_market_state
        self.colors = {
            'RANGE': ("#adadad", 0.15),              # Gris transparent
            'UPTREND': ("#7cff7c", 0.15),         # Vert très clair transparent
            'DOWNTREND': ("#ff8686", 0.15),       # Rouge très clair transparent
            'POTENTIAL_REVERSAL': ("#7c7cff", 0.15)  # Bleu très clair transparent
        }
        
        self.state_labels = {
            'RANGE': 'RANGE',
            'UPTREND': 'UP',
            'DOWNTREND': 'DOWN',
            'POTENTIAL_REVERSAL': 'REVERSAL'
        }    
    # ---------------------------------------------
    # 1) Detect swings
    # ---------------------------------------------
    # Vectoriée NumPy natives (10-100x plus rapide)
    #
    def detect_swings( self, df ):
        df = df.copy()
        L = self.params.swing_width
        n = len(df)
        
        highs = df['High'].values
        lows = df['Low'].values
        
        # Taille de la fenêtre :  2 * L + 1
        window_size = 2 * L + 1
        high_windows = sliding_window_view( highs, window_size )
        low_windows = sliding_window_view( lows, window_size )
        
        # Indices valides (centrer sur L)
        valid_indices = numpy.arange(L, n - L)
        center_highs = highs[valid_indices]
        center_lows = lows[valid_indices]
        
        # Détecter les swings
        is_swing_high = (
            ( center_highs == high_windows.max(axis=1) ) &  # Est le max
            ( numpy.sum( high_windows == center_highs[:, None], axis=1 ) == 1 )  # Unique
        )
        
        is_swing_low = (
            ( center_lows == low_windows.min(axis=1) ) &  # Est le min
            ( numpy.sum(low_windows == center_lows[:, None], axis=1) == 1 )  # Unique
        )
        
        # Assigner les résultats avec iloc au lieu de loc
        df['SwingHigh'] = False
        df['SwingLow'] = False
        df.iloc[ valid_indices[is_swing_high], df.columns.get_loc('SwingHigh')] = True
        df.iloc[ valid_indices[is_swing_low], df.columns.get_loc('SwingLow')] = True
        
        return df
    
    # -------------------
    # 2) Market Structure
    # -------------------
    def detect_structure( self, df ):
        df = df.copy()
        
        df['Structure'] = None
        df['StructurePrice'] = numpy.nan
        
        last_swing_high_price = None
        last_swing_low_price = None
        
        for i in range( len(df) ):
            
            # Nouveau Swing High détecté
            if df[ 'SwingHigh' ].iloc[i]:
                current_high = df[ 'High' ].iloc[i]
                
                # On a besoin d'un swing high précédent pour déterminer HH ou LH
                if last_swing_high_price is not None:
                    if current_high > last_swing_high_price:
                        # Higher High
                        df.loc[ df.index[i], 'Structure' ] = 'HH'
                    else:
                        # Lower High
                        df.loc[ df.index[i], 'Structure' ] = 'LH'
                        
                    df.loc[ df.index[i], 'StructurePrice' ] = current_high
                    
                last_swing_high_price = current_high
            
            # Nouveau Swing Low détecté
            if df['SwingLow'].iloc[i]:
                current_low = df['Low'].iloc[i]
                
                # On a besoin d'un swing low précédent pour déterminer HL ou LL
                if last_swing_low_price is not None:
                    if current_low > last_swing_low_price:
                        # Higher Low
                        df.loc[ df.index[i], 'Structure' ] = 'HL'
                    else:
                        # Lower Low
                        df.loc[ df.index[i], 'Structure' ] = 'LL'
                    
                    df.loc[ df.index[i], 'StructurePrice' ] = current_low
                    
                last_swing_low_price = current_low
        
        return df

    # ---------------------------
    # helpers: displacement check
    # ---------------------------
    def is_displacement( self, df, i ):
        # large body relative to range and body > displacement_body_ratio * (high-low)
        o = df['Open'].iat[i]
        c = df['Close'].iat[i]
        h = df['High'].iat[i]
        l = df['Low'].iat[i]
        body = abs(c - o)
        rng = h - l if (h - l) > 0 else 1e-9
        val = (body / rng)
        return val, val >= self.params.displacement_body_ratio

    # ------------------------------------
    # 3) CHoCH & BOS ICT
    # ------------------------------------
    def detect_bos_choch( self, df ):
        df = df.copy()

        df['CHOCH_UP'] = False
        df['CHOCH_DOWN'] = False
        df['CHOCH_LEVEL'] = None
        df['CHOCH_TYPE'] = None
        df['CHOCH_IDX'] = None

        df['BOS_UP'] = False
        df['BOS_DOWN'] = False
        df['BOS_LEVEL'] = None
        df['BOS_IDX'] = None

        # --- Mark displacement ---
        df['DISPLACEMENT'] = False
        df['DISPLACEMENT_VALUE'] = 0.0
        
        # --- Market State -- #
        df['MarketState'] = None

        for i in range(len(df)):
            val, result = self.is_displacement(df, i)
            if result:
                df.at[df.index[i], 'DISPLACEMENT'] = True
                df.at[df.index[i], 'DISPLACEMENT_VALUE'] = val

        state_market = MarketStockStateMachine()

        choch_level = None
        choch_direction = None   # 'UP' | 'DOWN'

        for idx in df.index:

            structure = df.at[ idx, 'Structure' ]
            close = df.at[ idx, 'Close']

            prev_state, new_state = state_market.update( structure, idx, close )
            df.at[ idx, 'MarketState'] = new_state.name # set market state for overlays_market_state

            # --- CHoCH --- #
            if new_state == MarketState.POTENTIAL_REVERSAL and prev_state == MarketState.UPTREND:
                if close < state_market.last_HL[1]:
                    df.at[ idx, 'CHOCH_LEVEL'] = state_market.last_HL[1] # price
                    df.at[ idx, 'CHOCH_IDX'] = state_market.last_HL[0] # save x position
                    df.at[ idx, 'CHOCH_TYPE'] = 'OPEN'
                    df.at[ idx, 'CHOCH_DOWN'] = True
                    choch_level = state_market.last_HL[1]
                    choch_direction = 'DOWN'

            elif new_state == MarketState.POTENTIAL_REVERSAL and prev_state == MarketState.DOWNTREND:
                if close > state_market.last_LH[1]:
                    df.at[ idx, 'CHOCH_LEVEL'] = state_market.last_LH[1] # price
                    df.at[ idx, 'CHOCH_IDX'] = state_market.last_LH[0] # save x position
                    df.at[ idx, 'CHOCH_TYPE'] = 'OPEN'
                    df.at[ idx, 'CHOCH_UP'] = True
                    choch_level = state_market.last_LH[1]
                    choch_direction = 'UP'

            # --- BOS DOWN --- #
            if ( 
                 choch_direction == 'DOWN'
                 and prev_state == MarketState.DOWNTREND
                 and close < choch_level
                 and df.at[ idx, 'DISPLACEMENT' ]
            ):
                df.at[ idx, 'BOS_DOWN' ] = True
                df.at[ idx, 'BOS_LEVEL' ] = state_market.last_LL[1]
                df.at[ idx, 'BOS_IDX' ] =  state_market.last_LL[0]
                choch_direction = None
                choch_level = None

            # --- BOS UP --- #
            if ( 
                 choch_direction == 'UP'
                 and prev_state == MarketState.UPTREND
                 and close > choch_level
                 and df.at[ idx, 'DISPLACEMENT' ]
            ):
                df.at[ idx, 'BOS_UP' ] = True
                df.at[ idx, 'BOS_LEVEL' ] = state_market.last_HH[1]
                df.at[ idx, 'BOS_IDX' ] = state_market.last_HH[0]
                choch_direction = None
                choch_level = None
            
            # --- CHoCH Closure --- #
            if choch_direction == 'DOWN' and choch_level:
                if close > state_market.last_HL[1]:
                    df.at[ idx, 'CHOCH_LEVEL'] = choch_level
                    df.at[ idx, 'CHOCH_IDX'] = idx
                    df.at[ idx, 'CHOCH_TYPE'] = 'CLOSE'
                    choch_direction = None
                    choch_level = None
                    
            elif choch_direction == 'UP' and choch_level:
                if close < state_market.last_LH[1]:
                    df.at[ idx, 'CHOCH_LEVEL'] = choch_level
                    df.at[ idx, 'CHOCH_IDX'] = idx
                    df.at[ idx, 'CHOCH_TYPE'] = 'CLOSE'
                    choch_direction = None
                    choch_level = None
                    
        return df

    # --------------------------------------------
    # 4) Liquidity detection (EQH/EQL + LIQ_TAKEN)
    # --------------------------------------------
    def detect_liquidity( self, df ):
        """
        Détection de la liquidité basée sur les pivots SwingHigh / SwingLow.
        Compatible ICT / SMC.
        """
        df = df.copy()

        # Colonnes de sortie
        df["EQH"] = False
        df["EQL"] = False
        df["BSL"] = False # Buy-Side Liquidity
        df["SSL"] = False # Sell-Side Liquidity
        df["Sweep_High"] = False
        df["Sweep_Low"] = False

        # Récupération des swings
        swing_high_idx = df.index[ df["SwingHigh"] ].tolist()
        swing_low_idx  = df.index[ df["SwingLow"] ].tolist()

        highs = df["High"]
        lows  = df["Low"]

        # ===== 1) Equal Highs / Equal Lows (liquidité reposante) =====
        # On compare chaque swing à ses voisins précédents
        for i in range( 1, len(swing_high_idx) ):
            idx_curr = swing_high_idx[i]
            idx_prev = swing_high_idx[i - 1]

            h1 = highs[idx_curr]
            h2 = highs[idx_prev]

            if abs(h1 - h2) / h1 < self.params.liquidity_threshold:
                df.at[idx_curr, "EQH"] = True
                df.at[idx_prev, "EQH"] = True
                df.at[idx_curr, "BSL"] = True
                df.at[idx_prev, "BSL"] = True

        for i in range( 1, len(swing_low_idx) ):
            idx_curr = swing_low_idx[i]
            idx_prev = swing_low_idx[i - 1]

            l1 = lows[idx_curr]
            l2 = lows[idx_prev]

            if abs(l1 - l2) / l1 < self.params.liquidity_threshold:
                df.at[idx_curr, "EQL"] = True
                df.at[idx_prev, "EQL"] = True
                df.at[idx_curr, "SSL"] = True
                df.at[idx_prev, "SSL"] = True

        # ===== 2) Sweeps (liquidité prise) =====
        # Sweep High = un nouveau swing high > précédent swing high
        for i in range(1, len(swing_high_idx)):
            idx_curr = swing_high_idx[i]
            idx_prev = swing_high_idx[i - 1]

            if highs[idx_curr] > highs[idx_prev]:
                df.at[idx_curr, "Sweep_High"] = True  # prise BSL

        # Sweep Low = un nouveau swing low < précédent swing low
        for i in range(1, len(swing_low_idx)):
            idx_curr = swing_low_idx[i]
            idx_prev = swing_low_idx[i - 1]

            if lows[idx_curr] < lows[idx_prev]:
                df.at[idx_curr, "Sweep_Low"] = True   # prise SSL

        return df

    # --------------------------
    # 5) FVG v2 (impulse driven)
    # --------------------------
    def detect_fvg( self, df ):
        df = df.copy()
        atv = atr( df, period=self.params.atr_period )
        df['FVG_UP'] = False
        df['FVG_DOWN'] = False
        for i in range(2, len(df)):
            # original rule: Low[i] > High[i-2] (up imbalance)
            if df['Low'].iat[i] > df['High'].iat[i-2]:
                # require displacement: the impulse that created it must be large enough vs ATR
                impulse_body = abs( df['Close'].iat[i-1] - df['Open'].iat[i-1] )
                if impulse_body > 0.25 * atv.iat[i-1]:
                    df.at[df.index[i], 'FVG_UP'] = True
            if df['High'].iat[i] < df['Low'].iat[i-2]:
                impulse_body = abs(df['Close'].iat[i-1] - df['Open'].iat[i-1])
                if impulse_body > 0.25 * atv.iat[i-1]:
                    df.at[df.index[i], 'FVG_DOWN'] = True
                    
        return df

    # ---------------
    # 6) Order Blocks
    # ---------------
    def detect_order_blocks( self, df, lookahead_impulse=3, violation_by='close' ):
        """
        df: DataFrame avec colonnes ['Open','High','Low','Close'], index ordonné.
        swing_lookback: pivot half-window (L).
        lookahead_impulse: nombre de barres après pivot pour confirmer impulsion.
        violation_by: 'close' or 'wick' (définit comment on détecte violation).
        """
        df = df.copy()
        df['OB_type'] = None # 'bull' ou 'bear'
        df['OB_top'] = numpy.nan
        df['OB_bottom'] = numpy.nan
        df['OB_origin_idx'] = pd.Series([ pd.NA ] * len(df), dtype="object")
        df['Broken_at'] = pd.Series([ pd.NA ] * len(df), dtype="object")
        df['Retest_at'] = pd.Series([ pd.NA ] * len(df), dtype="object")

        n = len(df)
        for i in range(n):
            # detect pivot high -> potential bearish order block
            if df['SwingHigh'].iat[i]:
                # check next lookahead for bearish impulse (ex: close lower than pivot)
                end = min( n, i + 1 + lookahead_impulse )
                impulse = df['Close'].iloc[ i+1:end ]
                if len(impulse) > 0 and impulse.min() < df['Low'].iloc[i]:  # impulsion baissière
                    # choose origin candle of OB: here la bougie juste avant l'impulsion (i)
                    origin_idx = i
                    # we define OB as the body/wick of origin candle (could be expanded)
                    top = df['High'].iloc[origin_idx]
                    bottom = df['Low'].iloc[origin_idx]
                    df.at[ df.index[origin_idx], 'OB_type' ] = 'bear'
                    df.at[ df.index[origin_idx], 'OB_top' ] = top
                    df.at[ df.index[origin_idx], 'OB_bottom' ] = bottom
                    df.at[ df.index[origin_idx], 'OB_origin_idx' ] = df.index[origin_idx]

            # detect pivot low -> potential bullish order block
            if  df['SwingLow'].iat[i]:
                end = min(n, i + 1 + lookahead_impulse)
                impulse = df['Close'].iloc[i+1:end]
                if len(impulse) > 0 and impulse.max() > df['High'].iloc[i]:  # impulsion haussière
                    origin_idx = i
                    top = df['High'].iloc[origin_idx]
                    bottom = df['Low'].iloc[origin_idx]
                    df.at[df.index[origin_idx], 'OB_type'] = 'bull'
                    df.at[df.index[origin_idx], 'OB_top'] = top
                    df.at[df.index[origin_idx], 'OB_bottom'] = bottom
                    df.at[df.index[origin_idx], 'OB_origin_idx'] = df.index[origin_idx]

        # compile list of OBs
        #
        obs = df[ df['OB_type'].notna() ][ ['OB_type','OB_top','OB_bottom'] ].copy()

        # monitor violations -> breaker creation
        #
        for obs_idx, row in obs.iterrows():
            ob_type = row['OB_type']
            top = row['OB_top']
            bottom = row['OB_bottom']
            origin_pos = df.index.get_loc(obs_idx)
            # start scanning after origin_pos + 1
            for j in range( origin_pos + 1, n ):
                high_j = df['High'].iloc[j]
                low_j = df['Low'].iloc[j]
                close_j = df['Close'].iloc[j]

                violated = False
                if ob_type == 'bull':
                    # bullish OB = was support; breaker if price closes below or wick goes below bottom
                    if violation_by == 'close' and close_j < bottom:
                        violated = True
                    if violation_by == 'wick' and low_j < bottom:
                        violated = True
                else:  # bear
                    if violation_by == 'close' and close_j > top:
                        violated = True
                    if violation_by == 'wick' and high_j > top:
                        violated = True

                if violated:
                    # mark broken_at on origin row (or store both)
                    df.at[ obs_idx, 'Broken_at' ] = df.index[j]
                    # optionally store the breaker zone as same as OB but flagged
                    df.at[ obs_idx, 'OB_type' ] = 'breaker_' + ob_type  # ex: breaker_bull
                    # detect quick retest: look next M bars for touch back into zone
                    M = 10
                    retest = None
                    for k in range( j+1, min(n, j+1+M) ):
                        if ob_type == 'bull':
                            # retest if price revisits zone (high >= bottom and low <= top maybe)
                            if df['High'].iloc[k] >= bottom and df['Low'].iloc[k] <= top:
                                retest = df.index[ k ]; break
                        else:
                            if df['High'].iloc[ k ] >= bottom and df['Low'].iloc[k] <= top:
                                retest = df.index[k]; break
                    df.at[ obs_idx, 'Retest_at' ] = retest
                    break  # stop scanning after first violation

        return df

    # ---------------------------------------------
    # 7) OTE Optimal Trade Entry / Premium-Discount
    # ---------------------------------------------
    def detect_ote(
        self, 
        df, 
        lookback=200, 
        fib_levels=(0.618, 0.79),   # zone OTE ICT
        min_move_pct=0.01
    ):
        """
        Détecte les zones OTE comme dans ICT :
        - OTE_buy : discount retracement dans un bull swing
        - OTE_sell : premium retracement dans un bear swing
        """

        df = df.copy()

        # Colonnes output
        for col in [
            'OTE_buy','OTE_sell',
            'OTE_zone_low','OTE_zone_high',
            'OTE_anchor_low','OTE_anchor_high',
            'OTE_A_idx','OTE_B_idx',        # <=== indispensable overlay
            'OTE_type'
        ]:
            df[col] = pd.NA

        df['OTE_buy'] = False
        df['OTE_sell'] = False

        # --- 1) Assure les pivots (SwingHigh / SwingLow) ---
        if 'SwingHigh' not in df.columns or 'SwingLow' not in df.columns:
            L = self.params.swing_width
            highs = df['High'].values
            lows = df['Low'].values

            swing_high = numpy.zeros(len(df), dtype=bool)
            swing_low  = numpy.zeros(len(df), dtype=bool)

            for i in range( L, len(df) - L ):
                if highs[i] == highs[i-L:i+L+1].max():
                    swing_high[i] = True
                if lows[i] == lows[i-L:i+L+1].min():
                    swing_low[i] = True

            df['SwingHigh'] = swing_high
            df['SwingLow'] = swing_low

        all_idx = list(df.index)
        n = len(all_idx)

        swing_high_idx = [i for i in df.index[df['SwingHigh']]]
        swing_low_idx  = [i for i in df.index[df['SwingLow']]]

        fib_low, fib_high = fib_levels

        def price(i, col):
            return float(df.loc[i, col])

        # --- 2) Impulsion haussière A(low) → B(high) → retracement C(low)
        for b_idx in swing_high_idx:

            pos_b = all_idx.index(b_idx)
            start_pos = max(0, pos_b - lookback)

            # pivot A
            lows_before = [
                i for i in swing_low_idx 
                if start_pos <= all_idx.index(i) < pos_b
            ]
            if not lows_before:
                continue

            a_idx = lows_before[-1]
            A = price(a_idx, 'Low')
            B = price(b_idx, 'High')

            # amplitude min
            if (B - A) / max(A, 1e-9) < min_move_pct:
                continue

            # pivot C = premier swing_low après B
            lows_after = [
                i for i in swing_low_idx 
                if pos_b < all_idx.index(i) <= pos_b + lookback
            ]
            if not lows_after:
                continue

            c_idx = lows_after[0]

            # zone ICT OTE discount : (0.618–0.79)
            ote_top = B - (B - A) * fib_low
            ote_bot = B - (B - A) * fib_high

            ote_high = max(ote_top, ote_bot)
            ote_low  = min(ote_top, ote_bot)

            c_low  = price(c_idx, 'Low')
            c_high = price(c_idx, 'High')

            if not (c_low <= ote_high and c_high >= ote_low):
                continue

            # marque C + quelques bougies
            pos_c = all_idx.index(c_idx)
            for k in range(pos_c, min(n, pos_c + 6)):
                idxk = all_idx[k]
                closek = price(idxk, 'Close')

                if ote_low <= closek <= ote_high:
                    df.at[idxk, 'OTE_buy'] = True
                    df.at[idxk, 'OTE_zone_low'] = ote_low
                    df.at[idxk, 'OTE_zone_high'] = ote_high
                    df.at[idxk, 'OTE_anchor_low'] = A
                    df.at[idxk, 'OTE_anchor_high'] = B
                    df.at[idxk, 'OTE_A_idx'] = a_idx
                    df.at[idxk, 'OTE_B_idx'] = b_idx
                    df.at[idxk, 'OTE_type'] = 'bull'
                else:
                    break

        # --- 3) Impulsion baissière A(high) → B(low) → retracement C(high)
        for b_idx in swing_low_idx:

            pos_b = all_idx.index(b_idx)
            start_pos = max(0, pos_b - lookback)

            highs_before = [
                i for i in swing_high_idx
                if start_pos <= all_idx.index(i) < pos_b
            ]
            if not highs_before:
                continue

            a_idx = highs_before[-1]
            A = price(a_idx, 'High')
            B = price(b_idx, 'Low')

            if (A - B) / max(A, 1e-9) < min_move_pct:
                continue

            highs_after = [
                i for i in swing_high_idx
                if pos_b < all_idx.index(i) <= pos_b + lookback
            ]
            if not highs_after:
                continue

            c_idx = highs_after[0]

            ote_top = B + (A - B) * fib_low
            ote_bot = B + (A - B) * fib_high

            ote_high = max(ote_top, ote_bot)
            ote_low  = min(ote_top, ote_bot)

            c_low  = price(c_idx, 'Low')
            c_high = price(c_idx, 'High')

            if not (c_low <= ote_high and c_high >= ote_low):
                continue

            pos_c = all_idx.index(c_idx)
            for k in range(pos_c, min(n, pos_c + 6)):
                idxk = all_idx[k]
                closek = price(idxk, 'Close')

                if ote_low <= closek <= ote_high:
                    df.at[idxk, 'OTE_sell'] = True
                    df.at[idxk, 'OTE_zone_low'] = ote_low
                    df.at[idxk, 'OTE_zone_high'] = ote_high
                    df.at[idxk, 'OTE_anchor_low'] = B
                    df.at[idxk, 'OTE_anchor_high'] = A
                    df.at[idxk, 'OTE_A_idx'] = a_idx
                    df.at[idxk, 'OTE_B_idx'] = b_idx
                    df.at[idxk, 'OTE_type'] = 'bear'
                else:
                    break

        return df

    # -------------------------
    # 8) Pipeline apply
    # -------------------------
    def apply( self, df ):
        df = ensure_datetime_index( df )
        self.idx_to_pos = {idx: pos for pos, idx in enumerate( df.index )}
        df1 = self.detect_swings(df)
        df2 = self.detect_structure(df1)
        df3 = self.detect_bos_choch(df2)
        df4 = self.detect_liquidity( df3 )
        df5 = self.detect_fvg(df4)
        df6 = self.detect_order_blocks(df5)
        df7 = self.detect_ote(df6)
        return df7

    # -------------------------------------------------------------------------
    
    def overlays_swings( self, df, ax, artists, key, visible_start ):

        if not all( col in df.columns for col in [ 'SwingHigh', 'SwingLow' ] ):
            return
                
        for idx in df.index:
            if df.loc[idx, 'SwingHigh']:
                pos = self.idx_to_pos[ idx ]
                y = df.loc[idx, 'High']
                sct = ax.scatter( pos, y, color='green', s=35, marker='v', visible=visible_start )
                artists[ key ].append( sct )

            if df.loc[idx, 'SwingLow']:
                pos = self.idx_to_pos[ idx ]
                y = df.loc[idx, 'Low']
                sct = ax.scatter( pos, y, color='red', s=35, marker='^', visible=visible_start )
                artists[ key ].append( sct )
           
    # -------------------------------------------------------------------------
    
    def overlays_structure( self, df, ax, artists, key, visible_start ):
        
        if 'Structure' in df.columns:
            for idx in df.index:
                s = df.loc[idx, 'Structure']
                if s in ('HH','HL','LH','LL'):
                    pos = self.idx_to_pos[ idx ]
                    y = df.loc[idx, 'High'] if s in ('HH','LH') else df.loc[idx, 'Low']
                    _y = y+0.004*y if s in ('HH', 'LH') else y-0.004*y
                    face = 'lightgreen' if s in ('HH','HL') else 'lightcoral'
                    txt = ax.text( pos, _y, s, fontsize=8, ha='center', 
                                va='bottom' if 'H' in s else 'top',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor=face, alpha=0.2),
                                visible=visible_start )
                    artists[ key ].append(txt)    

    # -------------------------------------------------------------------------

    def overlays_segments( self, df, ax, artists, key, visible_start ):

        if not all( col in df.columns for col in [ 'SwingHigh', 'SwingLow', 'MarketState' ] ):
            return

        last_pos   = None
        last_price = None
        last_structure  = None   # 'HL', 'HH', 'LH', 'LL'
        last_state = None

        for idx in df.index:

            structure = df.at[ idx, 'Structure' ]
            if structure == None:
                continue
            
            state = df.at[ idx, 'MarketState' ]
            price = df.at[ idx, 'StructurePrice' ]
            pos = self.idx_to_pos[ idx ]

            # --- tracé du segment ---
            valid = False

            if last_structure:
                if state == MarketState.UPTREND.name:
                    valid = (
                        (last_structure == 'HL' and structure == 'HH') or
                        (last_structure == 'HH' and structure == 'HL')
                    )
                    color = 'green'
                elif state == MarketState.DOWNTREND.name:
                    valid = (
                        (last_structure == 'LH' and structure == 'LL') or
                        (last_structure == 'LL' and structure == 'LH')
                    )
                    color = 'red'
                elif state == MarketState.POTENTIAL_REVERSAL.name:
                    valid = (
                        (last_structure == 'HH' and structure == 'LL') or
                        (last_structure == 'LL' and structure == 'HH')
                    )
                    color = 'grey'

            # --- tracé ---
            if valid:
                line = Line2D(
                    [last_pos, pos],
                    [last_price, price],
                    linewidth=1.3,
                    color=color,
                    alpha=0.85,
                    visible=visible_start
                )

                ax.add_line(line)
                artists[key].append(line)

            # --- mise à jour mémoire ---
            last_pos   = pos
            last_price = price
            last_structure  = structure
            last_state = state

    # -------------------------------------------------------------------------
    
    def overlays_displacement( self, df, ax, artists, key, visible_start ):
        
        if 'DISPLACEMENT' in df.columns:
            for idx in df.index:
                if df.loc[idx, 'DISPLACEMENT']:
                    pos = self.idx_to_pos[ idx ]
                    y = df.loc[idx, 'High']
                    v = "{:.2f}".format(df.loc[idx, 'DISPLACEMENT_VALUE'])
                    txt = ax.text(pos, y+0.1, v, ha='center', fontsize=7, color='red',
                                visible=visible_start)
                    sct = ax.scatter(pos, y, color='red', s=20, marker='v',
                                   visible=visible_start)
                    artists[ key ].extend( [txt, sct] )

    # -------------------------------------------------------------------------
    
    def overlays_market_state( self, df, ax, artists, key, visible_start ):

        if 'MarketState' in df.columns:
            timestamps = list( range( len( df ) ) )
            prices = df['Close'].values
            states = df['MarketState'].values
            
            # Tracer les bandes verticales pour chaque état
            current_state = states[ 0 ]
            start_idx = 0
            
            for i in range( 1, len( states ) + 1 ):
                # Changement d'état ou fin des données
                if i == len( states ) or states[i] != current_state:
                    end_idx = i if i < len( states ) else i - 1
                    
                    # Récupérer la couleur et l'alpha pour cet état
                    color, alpha = self.colors.get( current_state, ('gray', 0.15) )
                    
                    # Dessiner la bande verticale
                    vspan = ax.axvspan(
                        timestamps[ start_idx ], 
                        timestamps[ end_idx ], 
                        facecolor=color, 
                        alpha=alpha,
                        visible=visible_start, 
                        zorder=0
                    )
                    
                    # Ne pas affiche le texte pour l'état 'POTENTIAL_REVERSAL'
                    txt_visible = visible_start
                    if states[ start_idx ] == MarketState.POTENTIAL_REVERSAL.name:
                        txt_visible = False
                    
                    # Ajouter le label de l'état au milieu de la bande    
                    mid_idx = ( start_idx + end_idx ) // 2
                    label = self.state_labels.get( current_state, 'none' )
                    y_pos = prices.min() * 1.02
                    txt = ax.text(
                        timestamps[ mid_idx ], 
                        y_pos, 
                        label,
                        ha='center', va='top', fontsize=6, 
                        bbox=dict(boxstyle='square,pad=0.3', facecolor=color, alpha=0.2),
                        visible=txt_visible,
                        zorder=5
                    )
                        
                    artists[key].extend([vspan, txt])
                    
                    if i < len( states ):
                        current_state = states[i]
                        start_idx = i
                        
    # -------------------------------------------------------------------------
    
    def overlays_bos( self, df, ax, artists, key, visible_start ):
        
        if not any( col in df.columns for col in [ 'BOS_UP', 'BOS_DOWN' ] ):
            return
        
        for idx in df.index:
            pos = self.idx_to_pos[ idx ]

            # --- BOS UP --- #
            if df.at[ idx, 'BOS_UP' ]:
                y = df.at[ idx, 'BOS_LEVEL' ]
                idx_x = df.at[ idx, "BOS_IDX" ]
                pos_x = self.idx_to_pos[ idx_x ]                
                line = ax.hlines(
                    y=y, xmin=pos_x, xmax=pos,
                    linestyle='--', linewidth=0.8,
                    color='green', alpha=0.8,
                    visible=visible_start
                )
                pos_txt = (pos + pos_x) // 2
                txt = ax.text(
                    pos_txt, y * 1.003, 'BoS ↑',
                    fontsize=8, color='green',
                    bbox=dict(boxstyle='square,pad=0.3', facecolor='lightgreen', alpha=0.2),
                    visible=visible_start
                )
                artists[ key ].extend([line, txt])
                
                sct = ax.scatter( pos, y, color='green', s=30, marker='>', visible=visible_start )
                artists[ key ].append(sct)

            # --- BOS DOWN --- #
            if df.at[ idx, 'BOS_DOWN' ]:
                y = df.at[ idx, 'BOS_LEVEL' ]
                idx_x = df.at[ idx, "BOS_IDX" ]
                pos_x = self.idx_to_pos[ idx_x ]                
                line = ax.hlines(
                    y=y, xmin=pos_x, xmax=pos,
                    linestyle='--', linewidth=0.8,
                    color='red', alpha=0.8,
                    visible=visible_start
                )
                pos_txt = (pos + pos_x) // 2
                txt = ax.text(
                    pos_txt, y * 0.997, 'BoS ↓',
                    fontsize=8, color='red',
                    bbox=dict(boxstyle='square,pad=0.3', facecolor='lightcoral', alpha=0.2),
                    visible=visible_start
                )
                artists[key].extend( [line, txt] )
                
                sct = ax.scatter( pos, y, color='red', s=30, marker='>', visible=visible_start )
                artists[ key ].append(sct)


    # -------------------------------------------------------------------------
    
    def overlays_choch( self, df, ax, artists, key, visible_start ):
        
        if not any( col in df.columns for col in ['CHOCH_UP', 'CHOCH_DOWN'] ):
            return
        
        for idx in df.index:
            pos = self.idx_to_pos[ idx ]

            # --- CHOCH UP --- #
            if df.at[ idx, 'CHOCH_UP' ]:
                type = 'CHoCH ↑'
                if df.at[ idx, "CHOCH_TYPE" ] == 'CLOSE': # mark CLOSE type only
                    type += df.at[ idx, "CHOCH_TYPE" ]
                y = df.at[ idx, "CHOCH_LEVEL" ]
                idx_x = df.at[ idx, "CHOCH_IDX" ]
                pos_x = self.idx_to_pos[ idx_x ]
                line = ax.hlines(
                    y=y, xmin=pos_x, xmax=pos,
                    linestyle='--', linewidth=0.8,
                    color='dodgerblue', alpha=0.8,
                    visible=visible_start
                )
                pos_txt = (pos + pos_x) // 2
                txt = ax.text( 
                    pos_txt, y * 1.003, type,
                    fontsize=8, color='dodgerblue',
                    bbox=dict(boxstyle='square,pad=0.3', facecolor='lightblue', alpha=0.2),
                    visible=visible_start
                )
                artists[key].extend([line, txt])

            # --- CHOCH DOWN --- #
            if df.at[idx, 'CHOCH_DOWN']:
                type = 'CHoCH ↓'
                if df.at[ idx, "CHOCH_TYPE" ] == 'CLOSE': # mark CLOSE type only
                    type += df.at[ idx, "CHOCH_TYPE" ]
                y = df.at[ idx, "CHOCH_LEVEL" ]
                idx_x = df.at[ idx, "CHOCH_IDX" ]
                pos_x = self.idx_to_pos[ idx_x ]
                line = ax.hlines(
                    y=y, xmin=pos_x, xmax=pos,
                    linestyle='--', linewidth=0.8,
                    color='orange', alpha=0.8,
                    visible=visible_start
                )
                pos_txt = (pos + pos_x) // 2
                txt = ax.text(
                    pos_txt, y * 1.003, type,
                    fontsize=8, color='orange',
                    bbox=dict(boxstyle='square,pad=0.3', facecolor='moccasin', alpha=0.25),
                    visible=visible_start
                )
                artists[key].extend([line, txt])

    # -------------------------------------------------------------------------
    
    def overlays_liquidity( self, df, ax, artists, key, visible_start ):
        
        cluster_threshold = self.params.liquidity_threshold
        
        color_high = '#cc66ff33'
        color_low = '#66ddff33'
        
        highs = []
        lows = []

        for idx in df.index:
            if df.loc[idx, "EQH"]:
                highs.append((idx, df.loc[idx, "High"]))
            if df.loc[idx, "EQL"]:
                lows.append((idx, df.loc[idx, "Low"]))

        if len(highs) == 0 and len(lows) == 0:
            return

        def cluster_points( points, threshold ):
            if not points:
                return []
            points_sorted = sorted( points, key=lambda x: x[1] )
            clusters = []
            current = [points_sorted[0]]

            for idx, price in points_sorted[1:]:
                last_price = current[-1][1]
                if abs(price - last_price) / last_price < threshold:
                    current.append((idx, price))
                else:
                    clusters.append(current)
                    current = [(idx, price)]
            clusters.append(current)
            return clusters

        high_clusters = cluster_points( highs, cluster_threshold )
        low_clusters = cluster_points( lows, cluster_threshold )

        def add_liquidity_zone( cluster, color, is_high=True ):
            idx_list = [c[0] for c in cluster]
            price_list = [c[1] for c in cluster]
            price = numpy.median(price_list)

            x_left = min( self.idx_to_pos[idx] for idx in idx_list )
            x_right = max( self.idx_to_pos[idx] for idx in idx_list )

            if is_high:
                y_bottom = price
                y_top = y_bottom + y_bottom * cluster_threshold
            else:
                y_top = price
                y_bottom = y_top - y_top * cluster_threshold

            rect = patches.Rectangle( (x_left, y_bottom), x_right - x_left, y_top - y_bottom,
                                      linewidth=0, edgecolor=None, facecolor=color, visible=visible_start )
            ax.add_patch(rect)
            artists[ key ].append(rect)

            txt = ax.text( (x_left + x_right) / 2, price, "EQH" if is_high else "EQL",
                        color=color[:7], fontsize=7, ha="center",
                        va="bottom" if is_high else "top", visible=visible_start )
            artists[ key ].append( txt )

        for cluster in high_clusters:
            add_liquidity_zone(cluster, color_high, is_high=True)

        for cluster in low_clusters:
            add_liquidity_zone(cluster, color_low, is_high=False)

        # Sweeps
        for idx in df.index:
            x = self.idx_to_pos[ idx ]
            if df.loc[idx, "Sweep_High"]:
                txt = ax.annotate( "▼", (x, df.loc[idx, "High"] + 0.01*df.loc[idx, "High"]), color=color_high,
                                fontsize=7, ha="center", va="bottom", visible=visible_start )
                artists[ key ].append( txt )

            if df.loc[idx, "Sweep_Low"]:
                txt = ax.annotate("▲", (x, df.loc[idx, "Low"] + 0.01*df.loc[idx, "Low"]), color=color_low,
                                fontsize=7, ha="center", va="top", visible=visible_start )
                artists[ key ].append( txt )                    

    # -------------------------------------------------------------------------
    
    def overlays_fvg( self, df, ax, artists, key, visible_start ):
        
        if 'FVG_UP' not in df.columns:
            return
            
        for i, idx in enumerate( df.index[2:], start=2 ):
            pos = self.idx_to_pos[ idx ]

            if df.loc[idx, 'FVG_UP']:
                top = df.loc[idx, 'Low']
                bot = df.iloc[i-2]['High']
                rect = Rectangle( (pos-2, bot), 2, top-bot, facecolor='green',
                                    alpha=0.2, edgecolor=None,
                                    visible=visible_start )
                ax.add_patch(rect)
                artists[ key ].append(rect)
                
            if df.loc[idx, 'FVG_DOWN']:
                top = df.iloc[i-2]['Low']
                bot = df.loc[idx, 'High']
                rect = Rectangle( (pos-2, bot), 2, top-bot, facecolor='red',
                                    alpha=0.2, edgecolor=None,
                                    visible=visible_start )
                ax.add_patch(rect)
                artists[ key ].append(rect)

    # -------------------------------------------------------------------------
    
    def overlays_order_blocs( self, df, ax, artists, key, visible_start ):
        
        pos_start = None
        ob_type_reel = None
        
        for idx in df.index:
            pos = self.idx_to_pos[ idx ]
            ob_type = df.loc[idx, 'OB_type']
            
            if ob_type is None or not str(ob_type).startswith( 'breaker_' ):
                continue

            top = df.loc[idx, 'OB_top']
            bottom = df.loc[idx, 'OB_bottom']

            if pd.isna( top ) or pd.isna( bottom ):
                continue

            pos_broken = None
            pos_retest = None
            if pos_start == None:
                pos_start = pos
            if ob_type_reel == None:
                ob_type_reel = ob_type
            
            # Broken at
            broken = df.loc[idx, 'Broken_at']
            if pd.notna( broken ):
                pos_broken = self.idx_to_pos[ broken ]
                price = df.loc[broken, 'Close']
                sct = ax.scatter( pos_broken, price, color='darkblue', s=40, marker='v',
                               visible=visible_start)
                artists[ key ].append(sct)

            # Retest at
            retest = df.loc[idx, 'Retest_at']
            if pd.notna(retest):
                pos_retest = self.idx_to_pos[retest]
                price = df.loc[retest, 'Close']
                sct = ax.scatter( pos_retest, price, color='darkorange', s=40, marker='^',
                               visible=visible_start)
                artists[ key ].append(sct)


            if pos_broken != None or pos_retest != None:
                color = 'green' if 'bull' in ob_type_reel else 'red'
                if pos_broken != None:
                    _with = pos_broken - pos_start
                if pos_retest != None:
                    _with = pos_retest - pos_start
                                        
                rect = patches.Rectangle( (pos_start-0.4, bottom), width=_with,
                                        height=top - bottom, facecolor=color, edgecolor='darkblue',
                                        alpha=0.3, visible=visible_start )
                ax.add_patch(rect)
                artists[ key ].append(rect)
                
                pos_broken = None
                pos_retest = None
                pos_start = None
                ob_type_reel = None      
				

    # -------------------------------------------------------------------------
    
    def overlays_ote( self, df, ax, artists, key, visible_start ):

        if 'OTE_zone_low' not in df.columns:
            return

        color_line = "#1361ff80"
        
        fib_levels = {
            "0.618": 0.618,
            "0.705": 0.705,
            "0.790": 0.790,
        }

        for idx in df.index[ df['OTE_buy'] | df['OTE_sell'] ]:

            if df.at[ idx, 'OTE_buy']:
                color_zone = "#4cfc4640"
            
            if df.at[ idx, 'OTE_sell']:
                color_zone = "#fc464632"
            
            # récupérer A, B (pivots)
            A_idx = df.at[ idx, 'OTE_A_idx' ]
            B_idx = df.at[ idx, 'OTE_B_idx' ]

            if pd.isna(A_idx) or pd.isna(B_idx):
                continue

            zone_low  = df.at[ idx, 'OTE_zone_low' ]
            zone_high = df.at[ idx, 'OTE_zone_high' ]

            if pd.isna(zone_low) or pd.isna(zone_high):
                continue

            pos_A = self.idx_to_pos[ A_idx ]
            pos_B = self.idx_to_pos[ B_idx ]

            pos_start = min( pos_A, pos_B )
            pos_end   = max( pos_A, pos_B )

            # --------------------------
            # 1) ZONE HORIZONTALE A => B
            # --------------------------
            rect = patches.Rectangle(
                (pos_start, zone_low),
                width = pos_end - pos_start,
                height = zone_high - zone_low,
                facecolor = color_zone,
                edgecolor = 'none',
                alpha = 0.25,
                visible = visible_start
            )
            ax.add_patch(rect)
            artists[key].append(rect)

            # -----------------------------------
            # 2) LIGNES FIBS (0.618, 0.705, 0.79)
            # -----------------------------------
            total_range = zone_high - zone_low

            for label, level in fib_levels.items():
                y = zone_low + total_range * (level - 0.618) / (0.79 - 0.618)
                # y doit être recalculé dans l'échelle réelle
                # (on assume zone_low = fib0.618, zone_high = fib0.79)

                line = ax.hlines(
                    y, xmin=pos_start, xmax=pos_end,
                    linestyle='--', linewidth=0.8,
                    color=color_line, alpha=0.7,
                    visible=visible_start
                )
                artists[key].append(line)

                txt_f = ax.text(
                    pos_end, y, f"{label}",
                    ha='left', va='center',
                    fontsize=6, color=color_line,
                    visible=visible_start
                )
                artists[key].append(txt_f)

